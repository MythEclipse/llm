//! The transformer: embeddings → decoder blocks → logits.
//!
//! Two block types share the loop. A dense block is attention; a `qwen35` block
//! is either attention (with a gate and per-head q/k norms) or a gated delta net,
//! decided per layer from the header. The feed-forward half is the same for both.

use crate::compute::{Activation, AttentionSpec, Compute, GroupMember, Workspace};
use crate::cuda::Graph;
use crate::deltanet;
use crate::error::EngineError;
use crate::gguf::{DType, GgufFile};
use crate::ops::RopeTable;
use crate::tensor::{Weight, as_f32};
use crate::workspace::{self, BufferPlan};
use tracing::{debug, info, instrument};

/// Geometry taken from the GGUF header.
#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub n_layers: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_rot: usize,
    pub rope_base: f32,
    pub eps: f32,
    pub n_vocab: usize,
    /// Experts per layer; zero means a dense feed-forward.
    pub expert_count: usize,
    /// Experts each token is routed to. Ignored when `expert_count` is zero.
    pub expert_used: usize,
    /// Hidden width inside one expert. Equal to `n_ff` when dense.
    pub expert_ff: usize,
    /// `full_attention_interval`: every Nth block is a full attention layer and
    /// the rest are recurrent. Zero means every block is attention.
    pub full_attention_interval: usize,
    /// Shapes of the gated delta net blocks, when the model has any.
    pub ssm: Option<SsmConfig>,
    /// Whether attention layers normalise q and k per head before rope.
    pub qk_norm: bool,
    /// Whether the query projection packs a sigmoid gate beside each query.
    pub q_gate: bool,
}

/// The shapes a gated delta net block works in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SsmConfig {
    pub conv_kernel: usize,
    /// Width of one key and one value head.
    pub state_size: usize,
    /// Total width of all value heads.
    pub inner: usize,
    /// Value heads (`ssm.time_step_rank`).
    pub v_heads: usize,
    /// Key heads (`ssm.group_count`).
    pub k_heads: usize,
}

impl Config {
    /// Geometry from the GGUF header.
    ///
    /// Keys are read under the model's own architecture (`qwen2.block_count`,
    /// `llama.block_count`, …) because that is where a GGUF puts them; the
    /// `llama.` spelling stays as a fallback for files that keep it. Fields the
    /// format leaves optional (head counts, key length, rope base, epsilon,
    /// vocabulary) fall back the way llama.cpp does, and a model whose block
    /// type this engine does not implement is refused here rather than failing
    /// halfway through a load.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, EngineError> {
        let architecture = gguf.meta_str("general.architecture").unwrap_or("llama");
        let read = |name: &str| {
            gguf.meta_u64_opt(&format!("{architecture}.{name}"))
                .or_else(|| gguf.meta_u64_opt(&format!("llama.{name}")))
        };
        let read_float = |name: &str| {
            gguf.meta_f32_opt(&format!("{architecture}.{name}"))
                .or_else(|| gguf.meta_f32_opt(&format!("llama.{name}")))
        };
        let required = |name: &str| {
            read(name).ok_or_else(|| EngineError::MissingMetadata(format!("{architecture}.{name}")))
        };

        let n_layers = required("block_count")? as usize;
        let n_embd = required("embedding_length")? as usize;
        let n_head = required("attention.head_count")? as usize;
        let n_ff = required("feed_forward_length")? as usize;
        let n_head_kv = read("attention.head_count_kv").unwrap_or(n_head as u64) as usize;
        let head_dim = read("attention.key_length")
            .map(|value| value as usize)
            .unwrap_or_else(|| n_embd / n_head.max(1));
        let n_rot = read("rope.dimension_count")
            .map(|value| value as usize)
            .unwrap_or(head_dim);
        let rope_base = read_float("rope.freq_base").unwrap_or(10_000.0);
        let eps = read_float("attention.layer_norm_rms_epsilon").unwrap_or(1e-6);
        let n_vocab = match read("vocab_size") {
            Some(value) => value as usize,
            None => gguf.meta_i32_array("tokenizer.ggml.token_type")?.len(),
        };

        let expert_count = read("expert_count").unwrap_or(0) as usize;
        let expert_used = read("expert_used_count").unwrap_or(0) as usize;
        let expert_ff = read("expert_feed_forward_length")
            .map(|value| value as usize)
            .unwrap_or(n_ff);
        if expert_count > 0 && (expert_used == 0 || expert_used > expert_count) {
            return Err(EngineError::Format(format!(
                "expert_count {expert_count} with expert_used_count {expert_used}"
            )));
        }

        // A model that interleaves two block types says how often attention
        // appears; the rest of the blocks are recurrent. llama.cpp derives the
        // same way when the file carries no explicit per-layer array.
        let full_attention_interval = read("full_attention_interval").unwrap_or(0) as usize;
        let ssm = if gguf.has_key_prefix(&format!("{architecture}.ssm.")) {
            Some(SsmConfig {
                conv_kernel: required("ssm.conv_kernel")? as usize,
                state_size: required("ssm.state_size")? as usize,
                inner: required("ssm.inner_size")? as usize,
                v_heads: required("ssm.time_step_rank")? as usize,
                k_heads: required("ssm.group_count")? as usize,
            })
        } else {
            None
        };
        if ssm.is_some() && full_attention_interval == 0 {
            return Err(EngineError::Format(
                "state-space blocks without full_attention_interval: cannot tell which \
                 layers are recurrent"
                    .into(),
            ));
        }

        if !n_head.is_multiple_of(n_head_kv.max(1)) {
            return Err(EngineError::Format(
                "query head count is not a multiple of kv head count".into(),
            ));
        }
        check_supported(architecture, head_dim)?;
        // Per-head q/k norms are host code, so their presence decides the path
        // before any layer is loaded.
        let first_attention = (0..n_layers)
            .find(|index| {
                !(ssm.is_some()
                    && full_attention_interval > 0
                    && !(index + 1).is_multiple_of(full_attention_interval))
            })
            .unwrap_or(0);
        let qk_norm = gguf
            .data(&format!("blk.{first_attention}.attn_q_norm.weight"))
            .is_ok();
        // A query projection twice as wide as the heads it serves carries a gate
        // beside each query.
        let q_gate = gguf
            .tensor(&format!("blk.{first_attention}.attn_q.weight"))
            .map(|tensor| tensor.dims[1] as usize == 2 * n_head * head_dim)
            .unwrap_or(false);

        Ok(Self {
            n_layers,
            n_embd,
            n_ff,
            n_head,
            n_head_kv,
            head_dim,
            n_rot,
            rope_base,
            eps,
            n_vocab,
            expert_count,
            expert_used,
            expert_ff,
            full_attention_interval,
            ssm,
            qk_norm,
            q_gate,
        })
    }

    /// Whether block `index` runs a gated delta net instead of attention.
    pub fn is_recurrent(&self, index: usize) -> bool {
        self.ssm.is_some()
            && self.full_attention_interval > 0
            && !(index + 1).is_multiple_of(self.full_attention_interval)
    }

    /// How many blocks keep keys and values.
    pub fn attention_layers(&self) -> usize {
        (0..self.n_layers)
            .filter(|index| !self.is_recurrent(*index))
            .count()
    }

    /// Whether this model needs the host code paths: the delta net runs on the
    /// host, and the attention kernels are written for a head dimension of at
    /// most 128.
    /// Whether some blocks hand work to the host even when a device is present.
    /// The delta rule has no kernel, so a recurrent block reads its projections
    /// back and uploads its output; that also rules out capturing a graph.
    pub fn has_host_blocks(&self) -> bool {
        self.ssm.is_some()
    }

    pub fn kv_dim(&self) -> usize {
        self.n_head_kv * self.head_dim
    }

    #[allow(dead_code)]
    fn group_size(&self) -> usize {
        self.n_head / self.n_head_kv
    }
}

/// Whether this engine can run the model at all, decided from its header.
///
/// Two block types are implemented: attention and the gated delta net. The
/// attention kernels stage up to 256 elements per head; anything wider has no
/// kernel and no host path either.
fn check_supported(architecture: &str, head_dim: usize) -> Result<(), EngineError> {
    let _ = architecture;
    if head_dim > 256 {
        return Err(EngineError::Unsupported(format!(
            "head_dim {head_dim} exceeds the 256 the attention kernels stage"
        )));
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct ModelDescription {
    pub config: Config,
    /// The architecture the file declares, so a caller can report what it
    /// actually loaded instead of assuming every file is llama-shaped.
    pub architecture: String,
    pub tensors: usize,
    pub quantized_tensors: usize,
    pub weight_bytes: u64,
}

struct Layer<'a> {
    attn_norm: &'a [f32],
    /// The norm the feed-forward reads. A model with a post-attention norm uses
    /// that one, and its residual is the value before the norm; a llama-shaped
    /// model uses `ffn_norm`. The block arithmetic is the same either way.
    ffn_norm: &'a [f32],
    attn: Attention<'a>,
    feed_forward: FeedForward<'a>,
}

/// The mixing half of a block: attention, or a gated delta net.
enum Attention<'a> {
    Dense(DenseAttention<'a>),
    Recurrent(RecurrentAttention<'a>),
}

struct DenseAttention<'a> {
    q: Weight<'a>,
    k: Weight<'a>,
    v: Weight<'a>,
    out: Weight<'a>,
    /// Per-head RMS norms over the head dimension, applied before rope.
    q_norm: Option<&'a [f32]>,
    k_norm: Option<&'a [f32]>,
    /// The query projection packs a sigmoid gate beside every query, which is
    /// multiplied into the attention output before the output projection.
    gated: bool,
}

struct RecurrentAttention<'a> {
    qkv: Weight<'a>,
    gate: Weight<'a>,
    alpha: Weight<'a>,
    beta: Weight<'a>,
    out: Weight<'a>,
    conv: &'a [f32],
    decay: &'a [f32],
    dt: &'a [f32],
    norm: &'a [f32],
}

/// The feed-forward half of a layer: one dense block, or a router over a stack
/// of experts.
enum FeedForward<'a> {
    Dense {
        gate: Weight<'a>,
        up: Weight<'a>,
        down: Weight<'a>,
    },
    Sparse {
        /// One row per expert: `expert_count × n_embd`.
        router: Weight<'a>,
        gate: Experts<'a>,
        up: Experts<'a>,
        down: Experts<'a>,
    },
}

/// Experts stored as one tensor, one contiguous block each.
///
/// GGUF lists dimensions fastest-first, so a `[k, out, n_expert]` tensor holds
/// expert `i` as `i * byte_len(k * out) ..`. That makes an expert a plain slice,
/// which is what the matmul kernels want.
struct Experts<'a> {
    dtype: DType,
    k: usize,
    out: usize,
    data: &'a [u8],
}

impl<'a> Experts<'a> {
    fn expert(&self, index: usize) -> Weight<'a> {
        let bytes = self.dtype.byte_len((self.k * self.out) as u64) as usize;
        let start = index * bytes;
        Weight::new(
            self.dtype,
            self.k,
            self.out,
            &self.data[start..start + bytes],
        )
    }

    fn len(&self) -> usize {
        self.data.len() / self.dtype.byte_len((self.k * self.out) as u64) as usize
    }
}

/// The experts one token uses, with a softmax over the chosen logits.
///
/// The router is trained with a softmax over the top-k, not over every expert,
/// so the weights of the chosen ones sum to one.
fn top_experts(logits: &[f32], used: usize) -> Vec<(usize, f32)> {
    let mut ranked: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    ranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    ranked.truncate(used.clamp(1, ranked.len()));
    let peak = ranked[0].1;
    let mut total = 0.0f32;
    for (_, value) in ranked.iter_mut() {
        *value = (*value - peak).exp();
        total += *value;
    }
    for (_, value) in ranked.iter_mut() {
        *value /= total;
    }
    ranked
}

/// Key/value history for one conversation.
pub struct KvCache {
    capacity: usize,
    kv_dim: usize,
    keys: Vec<Activation>,
    values: Vec<Activation>,
    /// Which slot each block keeps its keys in. A recurrent block keeps none,
    /// and holding a cache for it would be dead memory at context length.
    slots: Vec<Option<usize>>,
    len: usize,
    entries_f16: bool,
}

impl KvCache {
    pub fn new(compute: &Compute, config: &Config, capacity: usize) -> Result<Self, EngineError> {
        let per_layer = capacity * config.kv_dim();
        // On the device the cache is f16 (llama.cpp's precision), so an entry is
        // two bytes and the buffer holds half as many f32 slots as positions ×
        // dims. The host path keeps f32 entries because its attention reads the
        // cache as plain slices.
        let device = compute.device_name().is_some();
        let slots = if device {
            per_layer.div_ceil(2)
        } else {
            per_layer
        };
        let mut keys = Vec::with_capacity(config.attention_layers());
        let mut values = Vec::with_capacity(config.attention_layers());
        let mut layer_slots = Vec::with_capacity(config.n_layers);
        for index in 0..config.n_layers {
            if config.is_recurrent(index) {
                layer_slots.push(None);
                continue;
            }
            layer_slots.push(Some(keys.len()));
            keys.push(compute.alloc(slots)?);
            values.push(compute.alloc(slots)?);
        }
        Ok(Self {
            capacity,
            kv_dim: config.kv_dim(),
            keys,
            values,
            slots: layer_slots,
            len: 0,
            entries_f16: device,
        })
    }

    /// The cache slot block `layer` writes to, or `None` when it keeps no keys.
    pub fn slot(&self, layer: usize) -> Option<usize> {
        self.slots.get(layer).copied().flatten()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn bytes(&self) -> usize {
        // k + v per layer; f16 entries are two bytes, f32 four.
        let per_entry = if self.entries_f16 { 2 } else { 4 };
        self.keys.len() * self.capacity * self.kv_dim * 2 * per_entry
    }

    pub fn reset(&mut self) {
        self.len = 0;
    }

    pub fn keys_mut(&mut self, layer: usize) -> Option<(&mut Activation, &mut Activation)> {
        let slot = self.slot(layer)?;
        Some((&mut self.keys[slot], &mut self.values[slot]))
    }

    /// Moves the position forward without a pass. A replayed graph has already
    /// written the keys and values; the counter still has to follow.
    pub fn advance(&mut self, tokens: usize) {
        self.len += tokens;
    }
}

/// The state the recurrent blocks carry between tokens.
///
/// It lives with the session, not with the model: two conversations have
/// different histories, and this is history. Held on the host because the delta
/// rule runs there — see `deltanet`.
pub struct RecurrentCache {
    /// One entry per block; empty for the attention blocks.
    layers: Vec<Vec<f32>>,
    geometry: Option<deltanet::Geometry>,
}

impl RecurrentCache {
    pub fn new(config: &Config) -> Self {
        let geometry = deltanet::Geometry::from_config(config);
        let layers = (0..config.n_layers)
            .map(|index| match geometry {
                Some(geometry) if config.is_recurrent(index) => {
                    vec![0.0; geometry.conv_state_len() + geometry.ssm_state_len()]
                }
                _ => Vec::new(),
            })
            .collect();
        Self { layers, geometry }
    }

    pub fn bytes(&self) -> usize {
        self.layers.iter().map(Vec::len).sum::<usize>() * std::mem::size_of::<f32>()
    }

    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.fill(0.0);
        }
    }

    fn layer_mut(&mut self, index: usize) -> Option<(&mut [f32], &mut [f32])> {
        let geometry = self.geometry?;
        let (conv_len, ssm_len) = (geometry.conv_state_len(), geometry.ssm_state_len());
        let layer = self.layers.get_mut(index)?;
        if layer.len() != conv_len + ssm_len {
            return None;
        }
        let (conv, ssm) = layer.split_at_mut(conv_len);
        Some((conv, ssm))
    }
}

pub struct Model<'a> {
    pub config: Config,
    gguf: &'a GgufFile,
    token_embd: Weight<'a>,
    output: Weight<'a>,
    output_norm: &'a [f32],
    layers: Vec<Layer<'a>>,
    compute: Compute,
}

impl<'a> Model<'a> {
    /// Device running the matmuls: `None` means the CPU.
    pub fn device_name(&self) -> Option<&str> {
        self.compute.device_name()
    }

    pub fn load(gguf: &'a GgufFile) -> Result<Self, EngineError> {
        Self::load_with_device(gguf, true)
    }

    /// `use_gpu = false` pins the CPU backend, which is what the tests that
    /// compare backends need.
    #[instrument(skip_all, fields(use_gpu, tensors = gguf.tensors().len()))]
    pub fn load_with_device(gguf: &'a GgufFile, use_gpu: bool) -> Result<Self, EngineError> {
        let config = Config::from_gguf(gguf)?;
        let weight_bytes: usize = gguf
            .tensors()
            .iter()
            .map(|tensor| tensor.byte_len as usize)
            .sum();
        let mut layers = Vec::with_capacity(config.n_layers);
        for index in 0..config.n_layers {
            layers.push(load_layer(gguf, &config, index)?);
        }
        let token_embd = weight(gguf, "token_embd.weight", config.n_embd, config.n_vocab)?;
        // A file may tie the output head to the token embedding, in which case
        // there is no `output.weight` and the embedding matrix is the head.
        let output = match gguf.tensor("output.weight") {
            Ok(_) => weight(gguf, "output.weight", config.n_embd, config.n_vocab)?,
            Err(_) => weight(gguf, "token_embd.weight", config.n_embd, config.n_vocab)?,
        };

        // Every block runs on the device when there is one. A recurrent block
        // hands its four projections back to the host for the state update and
        // takes the result back, because the delta rule has no kernel yet; the
        // projections are what the weights cost, so that island is cheap.
        let compute = if use_gpu {
            Compute::auto(&config, weight_bytes)
        } else {
            Compute::cpu(&config)
        };
        let mut all_weights: Vec<&Weight<'a>> = Vec::with_capacity(layers.len() * 7 + 2);
        all_weights.push(&token_embd);
        all_weights.push(&output);
        // Every expert is a separate slice, so a routed model reserves each one;
        // a dense layer reserves its three matrices.
        let mut expert_views: Vec<Vec<Weight<'a>>> = Vec::with_capacity(layers.len());
        for layer in &layers {
            match &layer.attn {
                Attention::Dense(attn) => {
                    all_weights.extend([&attn.q, &attn.k, &attn.v, &attn.out])
                }
                Attention::Recurrent(attn) => {
                    all_weights.extend([&attn.qkv, &attn.gate, &attn.alpha, &attn.beta, &attn.out])
                }
            }
            match &layer.feed_forward {
                FeedForward::Dense { gate, up, down } => all_weights.extend([gate, up, down]),
                FeedForward::Sparse { gate, up, down, .. } => {
                    let mut views = Vec::with_capacity(gate.len() * 3);
                    for index in 0..gate.len() {
                        views.push(gate.expert(index));
                        views.push(up.expert(index));
                        views.push(down.expert(index));
                    }
                    expert_views.push(views);
                }
            }
        }
        for views in &expert_views {
            all_weights.extend(views.iter());
        }
        // A device that cannot hold every weight falls back to the CPU here
        // rather than failing halfway through the first reply.
        let compute = match compute.preload(&all_weights) {
            Ok(()) => compute,
            Err(error) => Compute::with_fallback(error, &config),
        };
        info!(
            layers = config.n_layers,
            embedding = config.n_embd,
            vocab = config.n_vocab,
            weight_bytes,
            device = compute.device_name().unwrap_or("cpu"),
            "model loaded"
        );

        Ok(Self {
            token_embd,
            output,
            output_norm: as_f32(gguf.data("output_norm.weight")?),
            layers,
            config,
            compute,
            gguf,
        })
    }

    /// Facts the model-info endpoint reports: geometry, quantisation mix, and
    /// how much of the file is weights.
    pub fn describe(&self) -> ModelDescription {
        let mut tensors = 0usize;
        let mut quantized = 0usize;
        let mut weight_bytes = 0u64;
        for tensor in self.gguf.tensors() {
            tensors += 1;
            weight_bytes += tensor.byte_len;
            if tensor.dtype == crate::gguf::DType::Q8_0 {
                quantized += 1;
            }
        }
        ModelDescription {
            architecture: self
                .gguf
                .meta_str("general.architecture")
                .unwrap_or("llama")
                .to_string(),
            config: self.config,
            tensors,
            quantized_tensors: quantized,
            weight_bytes,
        }
    }

    /// The scratch layout this model plans for `tokens` at `capacity`, without
    /// allocating it: the endpoint reports these numbers and [`Scratch::new`]
    /// allocates the same plan.
    pub fn scratch_plan(&self, tokens: usize, capacity: usize) -> BufferPlan {
        let score_grid = self.compute.plan().attention(tokens).needs_score_grid();
        BufferPlan::plan(workspace::scratch_requests(
            &self.config,
            tokens,
            capacity,
            score_grid,
        ))
    }

    pub fn rope_table(&self, positions: usize) -> RopeTable {
        RopeTable::new(self.config.n_rot, self.config.rope_base, positions)
    }

    /// The device running this model, for reporting.
    pub fn compute(&self) -> &Compute {
        &self.compute
    }

    /// Why the CPU is doing the work, when a GPU was expected.
    pub fn device_note(&self) -> Option<&str> {
        self.compute.note()
    }

    /// Logits for the last token of `tokens`, with keys/values appended at
    /// `kv.len()`.
    ///
    /// `scratch` is sized for `tokens` by [`Scratch::new`], so the pass itself
    /// allocates nothing: a decode step reuses the same buffers every token.
    #[instrument(skip_all, fields(tokens = tokens.len(), position = kv.len()))]
    pub fn forward(
        &self,
        tokens: &[u32],
        kv: &mut KvCache,
        recurrent: &mut RecurrentCache,
        rope: &RopeTable,
        scratch: &mut Scratch,
    ) -> Result<Vec<f32>, EngineError> {
        self.set_position(kv, tokens.len())?;
        self.embed(tokens, scratch)?;
        if let Ok(dir) = std::env::var("LLM_DUMP_LAYERS") {
            dump_activation(
                &self.compute,
                &dir,
                "inp_embd",
                &scratch.x,
                tokens.len() * self.config.n_embd,
            );
        }
        self.body(tokens, kv, recurrent, rope, scratch)?;
        self.read_logits(scratch)
    }

    /// One decode step, captured into a graph once this session has run enough
    /// of them for the capture to pay for itself.
    ///
    /// A captured step is the same launches on the same pointers; only the
    /// position changes, and the kernels read that from device memory. The first
    /// steps run plainly for a second reason too: they are what grows the
    /// lazily-sized buffers, and nothing may allocate while a capture is open.
    pub fn forward_decode(
        &self,
        token: u32,
        kv: &mut KvCache,
        recurrent: &mut RecurrentCache,
        rope: &RopeTable,
        scratch: &mut Scratch,
        graphs: &mut DecodeGraphs,
    ) -> Result<Vec<f32>, EngineError> {
        graphs.steps += 1;
        // The position and the embedding are outside the graph: both change with
        // the token, and both need a host copy, which a capture cannot contain.
        self.set_position(kv, 1)?;
        self.embed(&[token], scratch)?;
        // The grid the attention kernel launches with depends on how many
        // positions are visible, so the cache is keyed on it: the same graph is
        // replayed until the context crosses into a wider grid.
        let shape = self.compute.decode_splits(kv.len() + 1);
        if let Some(graph) = graphs.captured.get(&shape) {
            self.compute.replay(graph)?;
            kv.advance(1);
            return self.read_logits(scratch);
        }
        // A host block (the delta net) touches memory the graph cannot see, so
        // only a model that runs entirely on the device is captured.
        let capture = graphs.steps >= self.compute.plan().graph_after_steps
            && self.compute.can_capture()
            && !self.config.has_host_blocks();
        if capture {
            self.compute.begin_capture()?;
        }
        let outcome = self.body(&[token], kv, recurrent, rope, scratch);
        if capture {
            match outcome {
                Ok(()) => {
                    let captured = self.compute.end_capture()?;
                    self.compute.replay(&captured)?;
                    graphs.captured.insert(shape, captured);
                }
                Err(error) => {
                    self.compute.cancel_capture();
                    return Err(error);
                }
            }
        } else {
            outcome?;
        }
        self.read_logits(scratch)
    }

    /// Writes the position the kernels read, before anything is launched.
    fn set_position(&self, kv: &KvCache, tokens: usize) -> Result<(), EngineError> {
        // A captured decode step bakes its arguments, so the one value that moves
        // every token arrives through a pointer instead.
        self.compute.set_position(kv.len(), kv.len() + tokens)
    }

    /// The embedding lookup, which needs a host copy of the token ids and so
    /// stays outside a captured step.
    fn embed(&self, tokens: &[u32], scratch: &mut Scratch) -> Result<(), EngineError> {
        for token in tokens {
            if *token as usize >= self.config.n_vocab {
                return Err(EngineError::Format(format!(
                    "token {token} outside vocabulary"
                )));
            }
        }
        self.compute
            .embeddings(&self.token_embd, tokens, &mut scratch.x, 0)
    }

    /// The launches of one pass, and nothing that waits on the host: this is
    /// exactly what a decode step records into a graph.
    fn body(
        &self,
        tokens: &[u32],
        kv: &mut KvCache,
        recurrent: &mut RecurrentCache,
        rope: &RopeTable,
        scratch: &mut Scratch,
    ) -> Result<(), EngineError> {
        let n = tokens.len();
        if n == 0 {
            return Err(EngineError::Format("empty prompt".into()));
        }
        if kv.len() + n > kv.capacity() {
            return Err(EngineError::ContextFull {
                used: kv.len() + n,
                capacity: kv.capacity(),
            });
        }
        let config = &self.config;
        let embd = config.n_embd;
        let start = kv.len();
        let capacity = kv.capacity();
        debug!(tokens = n, start, "forward pass started");

        if scratch.tokens() < n {
            return Err(EngineError::Format(format!(
                "scratch holds {} tokens, {n} requested",
                scratch.tokens()
            )));
        }

        let dump = std::env::var("LLM_DUMP_LAYERS").ok();
        for (index, layer) in self.layers.iter().enumerate() {
            let _layer_span =
                tracing::debug_span!("transformer_layer", layer = index, tokens = n, start)
                    .entered();
            self.layer_forward(
                layer, index, n, start, kv, recurrent, rope, scratch, capacity,
            )?;
            if let Some(dir) = &dump {
                dump_activation(
                    &self.compute,
                    dir,
                    &format!("l_out-{index}-at{start}"),
                    &scratch.x,
                    n * config.n_embd,
                );
            }
        }
        kv.len = start + n;

        self.compute.rmsnorm(
            &scratch.x,
            (n - 1) * embd,
            self.output_norm,
            config.eps,
            1,
            &mut scratch.output_norm,
            0,
        )?;
        self.compute.matmul(
            &self.output,
            &scratch.output_norm,
            0,
            1,
            &mut scratch.logits,
            0,
        )?;
        debug!(tokens = n, position = start + n, "forward pass complete");
        Ok(())
    }

    /// The logits the pass just wrote, on the host.
    fn read_logits(&self, scratch: &Scratch) -> Result<Vec<f32>, EngineError> {
        let mut values = vec![0.0f32; self.config.n_vocab];
        self.compute.download(&scratch.logits, &mut values)?;
        Ok(values)
    }

    #[allow(clippy::too_many_arguments)]
    fn layer_forward(
        &self,
        layer: &Layer<'a>,
        index: usize,
        n: usize,
        start: usize,
        kv: &mut KvCache,
        recurrent: &mut RecurrentCache,
        rope: &RopeTable,
        scratch: &mut Scratch,
        capacity: usize,
    ) -> Result<(), EngineError> {
        match &layer.attn {
            Attention::Dense(attn) => self.dense_attention(
                attn,
                layer.attn_norm,
                index,
                n,
                start,
                kv,
                rope,
                scratch,
                capacity,
            )?,
            Attention::Recurrent(attn) => {
                self.recurrent_attention(attn, layer.attn_norm, index, n, recurrent, scratch)?
            }
        }
        self.feed_forward(layer, n, scratch)
    }

    /// Attention, including the two things a `qwen35` block adds: a gate packed
    /// beside every query, and per-head norms on q and k before rope.
    #[allow(clippy::too_many_arguments)]
    fn dense_attention(
        &self,
        attn: &DenseAttention<'a>,
        attn_norm: &[f32],
        index: usize,
        n: usize,
        start: usize,
        kv: &mut KvCache,
        rope: &RopeTable,
        scratch: &mut Scratch,
        capacity: usize,
    ) -> Result<(), EngineError> {
        let config = &self.config;
        let kv_dim = config.kv_dim();

        // q, k and v read the same normalised rows, so the norm is quantised
        // straight from the hidden state — one kernel instead of a write-out,
        // a read-back, and a quantise pass.
        let normed = self.compute.norm_prepare(
            &scratch.x,
            0,
            attn_norm,
            config.eps,
            n,
            &mut scratch.normed_attn,
            0,
        )?;
        let query = if attn.gated {
            self.compute.matmul_prepared_group(
                &mut [
                    GroupMember {
                        weight: &attn.q,
                        out: &mut scratch.q_full,
                        out_offset: 0,
                    },
                    GroupMember {
                        weight: &attn.k,
                        out: &mut scratch.k,
                        out_offset: 0,
                    },
                    GroupMember {
                        weight: &attn.v,
                        out: &mut scratch.v,
                        out_offset: 0,
                    },
                ],
                &scratch.normed_attn,
                0,
                &normed,
                n,
            )?;
            self.compute.split_query_gate(
                &scratch.q_full,
                &mut scratch.q,
                &mut scratch.attn_gate,
                n,
                config.n_head,
                config.head_dim,
            )?;
            &mut scratch.q
        } else {
            self.compute.matmul_prepared_group(
                &mut [
                    GroupMember {
                        weight: &attn.q,
                        out: &mut scratch.q,
                        out_offset: 0,
                    },
                    GroupMember {
                        weight: &attn.k,
                        out: &mut scratch.k,
                        out_offset: 0,
                    },
                    GroupMember {
                        weight: &attn.v,
                        out: &mut scratch.v,
                        out_offset: 0,
                    },
                ],
                &scratch.normed_attn,
                0,
                &normed,
                n,
            )?;
            &mut scratch.q
        };
        if let Some(weight) = attn.q_norm {
            self.compute
                .rmsnorm_rows_in_place(query, 0, weight, config.eps, n * config.n_head)?;
        }
        if let Some(weight) = attn.k_norm {
            self.compute.rmsnorm_rows_in_place(
                &mut scratch.k,
                0,
                weight,
                config.eps,
                n * config.n_head_kv,
            )?;
        }

        let slot = kv
            .slot(index)
            .ok_or_else(|| EngineError::Format(format!("block {index} has no key cache")))?;
        self.compute.rope_append(
            query,
            &mut scratch.k,
            &scratch.v,
            &mut kv.keys[slot],
            &mut kv.values[slot],
            rope,
            n,
            config.n_head,
            config.n_head_kv,
            config.head_dim,
            start,
        )?;

        let spec = AttentionSpec {
            n_head: config.n_head,
            n_head_kv: config.n_head_kv,
            head_dim: config.head_dim,
            kv_dim,
            scores_stride: start + n,
            scale: 1.0 / (config.head_dim as f32).sqrt(),
            capacity,
        };
        self.compute.attend(
            &scratch.q,
            &kv.keys[slot],
            &kv.values[slot],
            &mut scratch.attn,
            &mut scratch.scores,
            &spec,
            start,
            n,
        )?;
        if let Ok(dir) = std::env::var("LLM_DUMP_LAYERS") {
            let width = n * config.n_head * config.head_dim;
            dump_activation(
                &self.compute,
                &dir,
                &format!("a_pregate-{index}-n{n}"),
                &scratch.attn,
                width,
            );
        }
        if attn.gated {
            let width = n * config.n_head * config.head_dim;
            self.compute
                .sigmoid_mul(&mut scratch.attn, &scratch.attn_gate, width)?;
            if let Ok(dir) = std::env::var("LLM_DUMP_LAYERS") {
                let width = n * config.n_head * config.head_dim;
                dump_activation(
                    &self.compute,
                    &dir,
                    &format!("a_gated-{index}-n{n}"),
                    &scratch.attn,
                    width,
                );
            }
        }

        self.compute.matmul_add(
            &attn.out,
            &scratch.attn,
            0,
            None,
            n,
            &mut scratch.x,
            0,
            &mut scratch.projected,
        )
    }

    /// A gated delta net block: four projections, then the recurrent arithmetic
    /// on the host, then the output projection.
    fn recurrent_attention(
        &self,
        attn: &RecurrentAttention<'a>,
        attn_norm: &[f32],
        index: usize,
        n: usize,
        recurrent: &mut RecurrentCache,
        scratch: &mut Scratch,
    ) -> Result<(), EngineError> {
        let config = &self.config;
        let geometry = deltanet::Geometry::from_config(config).ok_or_else(|| {
            EngineError::Format("recurrent block without state-space shapes".into())
        })?;

        let normed = self.compute.norm_prepare(
            &scratch.x,
            0,
            attn_norm,
            config.eps,
            n,
            &mut scratch.normed_attn,
            0,
        )?;
        self.compute.matmul_prepared_group(
            &mut [
                GroupMember {
                    weight: &attn.qkv,
                    out: &mut scratch.recurrent_qkv,
                    out_offset: 0,
                },
                GroupMember {
                    weight: &attn.gate,
                    out: &mut scratch.recurrent_z,
                    out_offset: 0,
                },
                GroupMember {
                    weight: &attn.alpha,
                    out: &mut scratch.recurrent_alpha,
                    out_offset: 0,
                },
                GroupMember {
                    weight: &attn.beta,
                    out: &mut scratch.recurrent_beta,
                    out_offset: 0,
                },
            ],
            &scratch.normed_attn,
            0,
            &normed,
            n,
        )?;

        let buffers = scratch.deltanet.as_mut().ok_or_else(|| {
            EngineError::Format("recurrent scratch missing from the workspace".into())
        })?;
        self.compute
            .download(&scratch.recurrent_qkv, &mut buffers.qkv)?;
        self.compute
            .download(&scratch.recurrent_z, &mut buffers.z)?;
        self.compute
            .download(&scratch.recurrent_alpha, &mut buffers.alpha)?;
        self.compute
            .download(&scratch.recurrent_beta, &mut buffers.beta)?;
        // Both gates are elementwise, so they belong with the projection that
        // produced them rather than with the state update.
        deltanet::activate_alpha(&mut buffers.alpha, attn.dt, geometry.v_heads);
        deltanet::activate_beta(&mut buffers.beta);

        let weights = deltanet::Weights {
            conv: attn.conv,
            decay: attn.decay,
            dt: attn.dt,
            norm: attn.norm,
            eps: config.eps,
        };
        let (conv, ssm) = recurrent
            .layer_mut(index)
            .ok_or_else(|| EngineError::Format(format!("block {index} has no recurrent state")))?;
        deltanet::block(
            &geometry,
            &weights,
            &mut deltanet::LayerState { conv, ssm },
            buffers,
            n,
        );

        let produced = &buffers.out;
        if let Ok(dir) = std::env::var("LLM_DUMP_LAYERS") {
            for (name, values) in [
                ("r_norm", scratch.normed_attn.host()),
                ("r_qkv", scratch.recurrent_qkv.host()),
                ("r_z", scratch.recurrent_z.host()),
                ("r_alpha", scratch.recurrent_alpha.host()),
                ("r_beta", scratch.recurrent_beta.host()),
            ] {
                if let Some(values) = values {
                    dump_slice(&dir, &format!("{name}-{index}-n{n}"), values);
                }
            }
            dump_slice(&dir, &format!("r_raw-{index}-n{n}"), &buffers.raw);
            dump_slice(&dir, &format!("r_conv-{index}-last"), &buffers.conv);
            dump_slice(&dir, &format!("r_out-{index}-n{n}"), produced);
        }
        self.compute.upload(&mut scratch.recurrent_out, produced)?;
        self.compute.matmul_add(
            &attn.out,
            &scratch.recurrent_out,
            0,
            None,
            n,
            &mut scratch.x,
            0,
            &mut scratch.projected,
        )
    }

    /// The feed-forward half of a block, which both block types share.
    fn feed_forward(
        &self,
        layer: &Layer<'a>,
        n: usize,
        scratch: &mut Scratch,
    ) -> Result<(), EngineError> {
        let config = &self.config;
        let ffn = config.n_ff;
        let normed = self.compute.norm_prepare(
            &scratch.x,
            0,
            layer.ffn_norm,
            config.eps,
            n,
            &mut scratch.normed_ffn,
            0,
        )?;
        match &layer.feed_forward {
            FeedForward::Dense { gate, up, down } => {
                self.compute.matmul_prepared_group(
                    &mut [
                        GroupMember {
                            weight: gate,
                            out: &mut scratch.gate,
                            out_offset: 0,
                        },
                        GroupMember {
                            weight: up,
                            out: &mut scratch.up,
                            out_offset: 0,
                        },
                    ],
                    &scratch.normed_ffn,
                    0,
                    &normed,
                    n,
                )?;
                let activated = self.compute.silu_mul_prepare(
                    &mut scratch.activated,
                    &scratch.gate,
                    &scratch.up,
                    n,
                    ffn,
                )?;
                self.compute.matmul_add(
                    down,
                    &scratch.activated,
                    0,
                    Some(&activated),
                    n,
                    &mut scratch.x,
                    0,
                    &mut scratch.projected,
                )?;
            }
            FeedForward::Sparse {
                router,
                gate,
                up,
                down,
            } => {
                self.expert_forward(router, gate, up, down, n, scratch)?;
            }
        }
        Ok(())
    }

    /// One token at a time through the experts it was routed to.
    ///
    /// The router is a single matmul per token; its logits go back to the host
    /// to choose the experts, because which weights to launch cannot be decided
    /// on the device. Each chosen expert then runs the same three matmuls a dense
    /// block does, with its output scaled by the routing weight before it is
    /// accumulated into the hidden state.
    fn expert_forward(
        &self,
        router: &Weight<'a>,
        gate: &Experts<'a>,
        up: &Experts<'a>,
        down: &Experts<'a>,
        n: usize,
        scratch: &mut Scratch,
    ) -> Result<(), EngineError> {
        let config = &self.config;
        let embd = config.n_embd;
        let used = config.expert_used;
        let mut logits = vec![0.0f32; config.expert_count];
        for token in 0..n {
            self.compute.matmul(
                router,
                &scratch.normed_ffn,
                token * embd,
                1,
                &mut scratch.router,
                0,
            )?;
            self.compute.download(&scratch.router, &mut logits)?;
            for (index, weight) in top_experts(&logits, used) {
                let expert_gate = gate.expert(index);
                let expert_up = up.expert(index);
                let expert_down = down.expert(index);
                self.compute.matmul(
                    &expert_gate,
                    &scratch.normed_ffn,
                    token * embd,
                    1,
                    &mut scratch.gate,
                    0,
                )?;
                self.compute.matmul(
                    &expert_up,
                    &scratch.normed_ffn,
                    token * embd,
                    1,
                    &mut scratch.up,
                    0,
                )?;
                self.compute.silu_mul(
                    &mut scratch.activated,
                    &scratch.gate,
                    &scratch.up,
                    config.expert_ff,
                )?;
                // Scale before the down projection: the projection is linear, so
                // this is the same as weighting its output afterwards, without a
                // second pass over the hidden state.
                self.compute
                    .scale(&mut scratch.activated, weight, config.expert_ff)?;
                self.compute.matmul_add(
                    &expert_down,
                    &scratch.activated,
                    0,
                    None,
                    1,
                    &mut scratch.x,
                    token * embd,
                    &mut scratch.projected,
                )?;
            }
        }
        Ok(())
    }
}

/// The decode steps this session has captured, one per grid shape.
///
/// A captured step bakes its launch grid, so a wider context needs a new one;
/// the graph is replayed until the context crosses into the next grid, which is
/// what keeps the capture cost off the hot path.
#[derive(Default)]
pub struct DecodeGraphs {
    captured: std::collections::HashMap<usize, Graph>,
    steps: usize,
}

impl DecodeGraphs {
    /// Graphs captured so far.
    pub fn len(&self) -> usize {
        self.captured.len()
    }

    pub fn is_empty(&self) -> bool {
        self.captured.is_empty()
    }
}

/// Buffers one forward pass reuses.
///
/// The plan decides the sizes and the offsets: the kernel plan says whether a
/// score grid is needed at all, and the lifetime table in [`crate::workspace`]
/// says which buffers can share a region. One allocation per scratch, so a
/// decode step allocates nothing at all.
pub struct Scratch {
    workspace: Workspace,
    /// Tokens this scratch is sized for.
    tokens: usize,
    x: Activation,
    logits: Activation,
    output_norm: Activation,
    normed_attn: Activation,
    normed_ffn: Activation,
    q: Activation,
    k: Activation,
    v: Activation,
    attn: Activation,
    projected: Activation,
    gate: Activation,
    up: Activation,
    activated: Activation,
    /// Router logits, one per expert. Empty when the model is dense.
    router: Activation,
    scores: Activation,
    /// The packed query of a gated attention block, before its gate is split off.
    q_full: Activation,
    /// That block's gate, applied to the attention output.
    attn_gate: Activation,
    /// Delta net projection outputs, one buffer per projection.
    recurrent_qkv: Activation,
    recurrent_z: Activation,
    recurrent_alpha: Activation,
    recurrent_beta: Activation,
    recurrent_out: Activation,
    /// Host scratch for the delta net's convolution and state update.
    deltanet: Option<deltanet::Buffers>,
}

impl Scratch {
    pub fn new(
        compute: &Compute,
        config: &Config,
        tokens: usize,
        capacity: usize,
    ) -> Result<Self, EngineError> {
        let score_grid = compute.plan().attention(tokens).needs_score_grid();
        let plan = BufferPlan::plan(workspace::scratch_requests(
            config, tokens, capacity, score_grid,
        ));
        let workspace = compute.workspace(plan)?;
        let x = workspace.view("x");
        let logits = workspace.view("logits");
        let output_norm = workspace.view("output_norm");
        let normed_attn = workspace.view("normed_attn");
        let normed_ffn = workspace.view("normed_ffn");
        let q = workspace.view("q");
        let k = workspace.view("k");
        let v = workspace.view("v");
        let attn = workspace.view("attn");
        let projected = workspace.view("projected");
        let gate = workspace.view("gate");
        let up = workspace.view("up");
        let activated = workspace.view("activated");
        let router = workspace.view("router");
        let scores = workspace.view("scores");
        let q_full = workspace.view("q_full");
        let attn_gate = workspace.view("attn_gate");
        let recurrent_qkv = workspace.view("recurrent_qkv");
        let recurrent_z = workspace.view("recurrent_z");
        let recurrent_alpha = workspace.view("recurrent_alpha");
        let recurrent_beta = workspace.view("recurrent_beta");
        let recurrent_out = workspace.view("recurrent_out");
        let deltanet = deltanet::Geometry::from_config(config)
            .map(|geometry| deltanet::Buffers::new(&geometry, tokens));
        Ok(Self {
            workspace,
            tokens,
            x,
            logits,
            output_norm,
            normed_attn,
            normed_ffn,
            q,
            k,
            v,
            attn,
            projected,
            gate,
            up,
            activated,
            router,
            scores,
            q_full,
            attn_gate,
            recurrent_qkv,
            recurrent_z,
            recurrent_alpha,
            recurrent_beta,
            recurrent_out,
            deltanet,
        })
    }

    /// Tokens this scratch can carry in one call.
    pub fn tokens(&self) -> usize {
        self.tokens
    }

    /// Bytes the plan holds.
    pub fn bytes(&self) -> usize {
        self.workspace.bytes()
    }
}

/// Write a plain slice as raw little-endian f32.
fn dump_slice(dir: &str, name: &str, values: &[f32]) {
    use std::io::Write;
    let path = std::path::Path::new(dir).join(format!("{name}.f32"));
    if let Ok(mut file) = std::fs::File::create(path) {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let _ = file.write_all(&bytes);
    }
}

/// Write an activation as raw little-endian f32, so a reference implementation's
/// dump can be compared value by value. Set `LLM_DUMP_LAYERS` to a directory.
fn dump_activation(compute: &Compute, dir: &str, name: &str, values: &Activation, len: usize) {
    use std::io::Write;
    let len = len.min(values.len());
    let mut buffer = vec![0.0f32; len];
    if compute.download(values, &mut buffer).is_err() {
        return;
    }
    let path = std::path::Path::new(dir).join(format!("{name}.f32"));
    if let Ok(mut file) = std::fs::File::create(path) {
        let mut bytes = Vec::with_capacity(len * 4);
        for value in &buffer {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let _ = file.write_all(&bytes);
    }
}

/// Read one block's tensors, in whichever shape the file declares.
///
/// Which half is recurrent comes from the header's `full_attention_interval`, and
/// whether the query projection carries a gate comes from its own width — a
/// `2 × n_head × head_dim` output is query-then-gate per head. Nothing here
/// checks the architecture's name.
fn load_layer<'a>(
    gguf: &'a GgufFile,
    config: &Config,
    index: usize,
) -> Result<Layer<'a>, EngineError> {
    let prefix = format!("blk.{index}");
    let attn_norm = as_f32(gguf.data(&format!("{prefix}.attn_norm.weight"))?);
    let ffn_norm = match gguf.data(&format!("{prefix}.post_attention_norm.weight")) {
        Ok(data) => as_f32(data),
        Err(_) => as_f32(gguf.data(&format!("{prefix}.ffn_norm.weight"))?),
    };
    let attn = if config.is_recurrent(index) {
        let ssm = config
            .ssm
            .ok_or_else(|| EngineError::Format("recurrent block without ssm keys".into()))?;
        let key_dim = ssm.state_size * ssm.k_heads;
        let value_dim = ssm.inner;
        Attention::Recurrent(RecurrentAttention {
            qkv: weight(
                gguf,
                &format!("{prefix}.attn_qkv.weight"),
                config.n_embd,
                2 * key_dim + value_dim,
            )?,
            gate: weight(
                gguf,
                &format!("{prefix}.attn_gate.weight"),
                config.n_embd,
                value_dim,
            )?,
            alpha: weight(
                gguf,
                &format!("{prefix}.ssm_alpha.weight"),
                config.n_embd,
                ssm.v_heads,
            )?,
            beta: weight(
                gguf,
                &format!("{prefix}.ssm_beta.weight"),
                config.n_embd,
                ssm.v_heads,
            )?,
            out: weight(
                gguf,
                &format!("{prefix}.ssm_out.weight"),
                value_dim,
                config.n_embd,
            )?,
            conv: as_f32(gguf.data(&format!("{prefix}.ssm_conv1d.weight"))?),
            decay: as_f32(gguf.data(&format!("{prefix}.ssm_a"))?),
            dt: as_f32(gguf.data(&format!("{prefix}.ssm_dt.bias"))?),
            norm: as_f32(gguf.data(&format!("{prefix}.ssm_norm.weight"))?),
        })
    } else {
        let plain = config.n_head * config.head_dim;
        let q_width = gguf.tensor(&format!("{prefix}.attn_q.weight"))?.dims[1] as usize;
        let gated = q_width == plain * 2;
        if q_width != plain && !gated {
            return Err(EngineError::Format(format!(
                "{prefix}.attn_q has width {q_width}, expected {plain} or {}",
                plain * 2
            )));
        }
        Attention::Dense(DenseAttention {
            q: weight(
                gguf,
                &format!("{prefix}.attn_q.weight"),
                config.n_embd,
                q_width,
            )?,
            k: weight(
                gguf,
                &format!("{prefix}.attn_k.weight"),
                config.n_embd,
                config.kv_dim(),
            )?,
            v: weight(
                gguf,
                &format!("{prefix}.attn_v.weight"),
                config.n_embd,
                config.kv_dim(),
            )?,
            out: weight(
                gguf,
                &format!("{prefix}.attn_output.weight"),
                plain,
                config.n_embd,
            )?,
            q_norm: optional_f32(gguf, &format!("{prefix}.attn_q_norm.weight")),
            k_norm: optional_f32(gguf, &format!("{prefix}.attn_k_norm.weight")),
            gated,
        })
    };
    let feed_forward = if config.expert_count == 0 {
        FeedForward::Dense {
            gate: weight(
                gguf,
                &format!("{prefix}.ffn_gate.weight"),
                config.n_embd,
                config.n_ff,
            )?,
            up: weight(
                gguf,
                &format!("{prefix}.ffn_up.weight"),
                config.n_embd,
                config.n_ff,
            )?,
            down: weight(
                gguf,
                &format!("{prefix}.ffn_down.weight"),
                config.n_ff,
                config.n_embd,
            )?,
        }
    } else {
        FeedForward::Sparse {
            router: weight(
                gguf,
                &format!("{prefix}.ffn_gate_inp.weight"),
                config.n_embd,
                config.expert_count,
            )?,
            gate: experts(
                gguf,
                &format!("{prefix}.ffn_gate_exps.weight"),
                config.n_embd,
                config.expert_ff,
                config.expert_count,
            )?,
            up: experts(
                gguf,
                &format!("{prefix}.ffn_up_exps.weight"),
                config.n_embd,
                config.expert_ff,
                config.expert_count,
            )?,
            down: experts(
                gguf,
                &format!("{prefix}.ffn_down_exps.weight"),
                config.expert_ff,
                config.n_embd,
                config.expert_count,
            )?,
        }
    };
    Ok(Layer {
        attn_norm,
        ffn_norm,
        attn,
        feed_forward,
    })
}

fn optional_f32<'a>(gguf: &'a GgufFile, name: &str) -> Option<&'a [f32]> {
    gguf.data(name).ok().map(as_f32)
}

/// A stacked expert tensor: `[k, out, n_expert]`, validated against the header.
fn experts<'a>(
    gguf: &'a GgufFile,
    name: &str,
    k: usize,
    out: usize,
    count: usize,
) -> Result<Experts<'a>, EngineError> {
    let tensor = gguf.tensor(name)?;
    if tensor.dims != vec![k as u64, out as u64, count as u64] {
        return Err(EngineError::Format(format!(
            "{name} has shape {:?}, expected [{k}, {out}, {count}]",
            tensor.dims
        )));
    }
    let data = gguf.data(name)?;
    let expected = tensor.dtype.byte_len((k * out * count) as u64) as usize;
    if data.len() != expected {
        return Err(EngineError::Format(format!(
            "{name} holds {} bytes, expected {expected}",
            data.len()
        )));
    }
    Ok(Experts {
        dtype: tensor.dtype,
        k,
        out,
        data,
    })
}

fn weight<'a>(
    gguf: &'a GgufFile,
    name: &str,
    k: usize,
    out: usize,
) -> Result<Weight<'a>, EngineError> {
    let tensor = gguf.tensor(name)?;
    if tensor.dims != vec![k as u64, out as u64] {
        return Err(EngineError::Format(format!(
            "{name} has shape {:?}, expected [{k}, {out}]",
            tensor.dims
        )));
    }
    Ok(Weight::new(tensor.dtype, k, out, gguf.data(name)?))
}

#[cfg(test)]
pub mod tests {
    use super::{Config, check_supported};
    use crate::error::EngineError;

    #[test]
    fn the_router_takes_the_highest_logits_and_normalises_them() {
        let picks = super::top_experts(&[1.0, 5.0, 3.0, 2.0], 2);
        assert_eq!(picks.len(), 2);
        assert_eq!(picks[0].0, 1, "highest logit wins");
        assert_eq!(picks[1].0, 2, "second highest follows");
        let total: f32 = picks.iter().map(|(_, weight)| weight).sum();
        assert!((total - 1.0).abs() < 1e-6, "weights sum to {total}");
        assert!(
            picks[0].1 > picks[1].1,
            "the higher logit carries more weight"
        );
    }

    #[test]
    fn the_router_never_asks_for_more_experts_than_there_are() {
        let picks = super::top_experts(&[0.0, 0.0], 8);
        assert_eq!(picks.len(), 2);
        let total: f32 = picks.iter().map(|(_, weight)| weight).sum();
        assert!((total - 1.0).abs() < 1e-6);
    }

    #[test]
    fn a_dense_attention_block_is_supported() {
        assert!(check_supported("qwen2", 128).is_ok());
        assert!(check_supported("llama", 64).is_ok());
    }

    #[test]
    fn a_head_dimension_beyond_what_the_engine_handles_is_refused_by_name() {
        // 256 is a `qwen35` head and runs on the host; 512 is past what any
        // kernel or host path here is written for.
        assert!(check_supported("qwen35", 256).is_ok());
        let error = check_supported("qwen35", 512).unwrap_err();
        match error {
            EngineError::Unsupported(message) => {
                assert!(message.contains("head_dim 512"), "{message}");
            }
            other => panic!("expected an unsupported error, got {other:?}"),
        }
    }

    #[test]
    fn a_state_space_model_keeps_attention_every_nth_block() {
        let config = Config {
            n_layers: 8,
            full_attention_interval: 4,
            ssm: Some(super::SsmConfig {
                conv_kernel: 4,
                state_size: 128,
                inner: 4096,
                v_heads: 32,
                k_heads: 16,
            }),
            ..minicpm()
        };
        let recurrent: Vec<usize> = (0..config.n_layers)
            .filter(|i| config.is_recurrent(*i))
            .collect();
        assert_eq!(recurrent, vec![0, 1, 2, 4, 5, 6]);
        assert_eq!(config.attention_layers(), 2);
        assert!(config.has_host_blocks());
    }

    #[test]
    fn a_dense_model_needs_no_host_path() {
        assert!(!minicpm().has_host_blocks());
    }

    /// The geometry of the model this engine ships with: 42 layers, 16 query
    /// heads over 2 kv heads, head_dim 128.
    pub fn minicpm() -> Config {
        Config {
            n_layers: 42,
            n_embd: 2048,
            n_ff: 6144,
            n_head: 16,
            n_head_kv: 2,
            head_dim: 128,
            n_rot: 128,
            rope_base: 5e6,
            eps: 1e-6,
            n_vocab: 130_560,
            expert_count: 0,
            expert_used: 0,
            expert_ff: 6144,
            full_attention_interval: 0,
            ssm: None,
            qk_norm: false,
            q_gate: false,
        }
    }
}
