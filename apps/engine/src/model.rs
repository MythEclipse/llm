//! The MiniCPM5 transformer: embeddings → 42 decoder layers → logits.

use crate::compute::{Activation, AttentionSpec, Compute, GroupMember, Workspace};
use crate::cuda::Graph;
use crate::error::EngineError;
use crate::gguf::GgufFile;
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

        if !n_head.is_multiple_of(n_head_kv.max(1)) {
            return Err(EngineError::Format(
                "query head count is not a multiple of kv head count".into(),
            ));
        }
        check_supported(
            architecture,
            head_dim,
            gguf.has_key_prefix(&format!("{architecture}.ssm.")),
        )?;

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
        })
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
/// Every kernel here is written for a dense attention block with a head
/// dimension of at most 128. A model that interleaves another block type (a
/// state-space layer, for instance) needs a forward pass this engine does not
/// have, and saying so at load is better than half-loading it.
fn check_supported(
    architecture: &str,
    head_dim: usize,
    has_state_space: bool,
) -> Result<(), EngineError> {
    if has_state_space {
        return Err(EngineError::Unsupported(format!(
            "architecture {architecture} interleaves state-space blocks with attention; \
             this engine implements dense attention blocks only"
        )));
    }
    if head_dim > 128 {
        return Err(EngineError::Unsupported(format!(
            "head_dim {head_dim} exceeds the 128 the attention kernels are written for"
        )));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
pub struct ModelDescription {
    pub config: Config,
    pub tensors: usize,
    pub quantized_tensors: usize,
    pub weight_bytes: u64,
}

struct Layer<'a> {
    attn_norm: &'a [f32],
    ffn_norm: &'a [f32],
    wq: Weight<'a>,
    wk: Weight<'a>,
    wv: Weight<'a>,
    wo: Weight<'a>,
    w_gate: Weight<'a>,
    w_up: Weight<'a>,
    w_down: Weight<'a>,
}

/// Key/value history for one conversation.
pub struct KvCache {
    capacity: usize,
    kv_dim: usize,
    keys: Vec<Activation>,
    values: Vec<Activation>,
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
        let mut keys = Vec::with_capacity(config.n_layers);
        let mut values = Vec::with_capacity(config.n_layers);
        for _ in 0..config.n_layers {
            keys.push(compute.alloc(slots)?);
            values.push(compute.alloc(slots)?);
        }
        Ok(Self {
            capacity,
            kv_dim: config.kv_dim(),
            keys,
            values,
            len: 0,
            entries_f16: device,
        })
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

    /// Moves the position forward without a pass. A replayed graph has already
    /// written the keys and values; the counter still has to follow.
    pub fn advance(&mut self, tokens: usize) {
        self.len += tokens;
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
            let prefix = format!("blk.{index}");
            layers.push(Layer {
                attn_norm: as_f32(gguf.data(&format!("{prefix}.attn_norm.weight"))?),
                ffn_norm: as_f32(gguf.data(&format!("{prefix}.ffn_norm.weight"))?),
                wq: weight(
                    gguf,
                    &format!("{prefix}.attn_q.weight"),
                    config.n_embd,
                    config.n_embd,
                )?,
                wk: weight(
                    gguf,
                    &format!("{prefix}.attn_k.weight"),
                    config.n_embd,
                    config.kv_dim(),
                )?,
                wv: weight(
                    gguf,
                    &format!("{prefix}.attn_v.weight"),
                    config.n_embd,
                    config.kv_dim(),
                )?,
                wo: weight(
                    gguf,
                    &format!("{prefix}.attn_output.weight"),
                    config.n_embd,
                    config.n_embd,
                )?,
                w_gate: weight(
                    gguf,
                    &format!("{prefix}.ffn_gate.weight"),
                    config.n_embd,
                    config.n_ff,
                )?,
                w_up: weight(
                    gguf,
                    &format!("{prefix}.ffn_up.weight"),
                    config.n_embd,
                    config.n_ff,
                )?,
                w_down: weight(
                    gguf,
                    &format!("{prefix}.ffn_down.weight"),
                    config.n_ff,
                    config.n_embd,
                )?,
            });
        }
        let token_embd = weight(gguf, "token_embd.weight", config.n_embd, config.n_vocab)?;
        let output = weight(gguf, "output.weight", config.n_embd, config.n_vocab)?;

        let compute = if use_gpu {
            Compute::auto(&config, weight_bytes)
        } else {
            Compute::cpu(&config)
        };
        let mut all_weights: Vec<&Weight<'a>> = Vec::with_capacity(layers.len() * 7 + 2);
        all_weights.push(&token_embd);
        all_weights.push(&output);
        for layer in &layers {
            all_weights.extend([
                &layer.wq,
                &layer.wk,
                &layer.wv,
                &layer.wo,
                &layer.w_gate,
                &layer.w_up,
                &layer.w_down,
            ]);
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
        rope: &RopeTable,
        scratch: &mut Scratch,
    ) -> Result<Vec<f32>, EngineError> {
        self.set_position(kv, tokens.len())?;
        self.embed(tokens, scratch)?;
        self.body(tokens, kv, rope, scratch)?;
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
        let capture =
            graphs.steps >= self.compute.plan().graph_after_steps && self.compute.can_capture();
        if capture {
            self.compute.begin_capture()?;
        }
        let outcome = self.body(&[token], kv, rope, scratch);
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

        for (index, layer) in self.layers.iter().enumerate() {
            let _layer_span =
                tracing::debug_span!("transformer_layer", layer = index, tokens = n, start)
                    .entered();
            self.layer_forward(layer, index, n, start, kv, rope, scratch, capacity)?;
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
        rope: &RopeTable,
        scratch: &mut Scratch,
        capacity: usize,
    ) -> Result<(), EngineError> {
        let config = &self.config;
        let kv_dim = config.kv_dim();
        let ffn = config.n_ff;

        // q, k and v read the same normalised rows, so the norm is quantised
        // straight from the hidden state — one kernel instead of a write-out,
        // a read-back, and a quantise pass.
        let normed = self.compute.norm_prepare(
            &scratch.x,
            0,
            layer.attn_norm,
            config.eps,
            n,
            &mut scratch.normed_attn,
            0,
        )?;
        self.compute.matmul_prepared_group(
            &mut [
                GroupMember {
                    weight: &layer.wq,
                    out: &mut scratch.q,
                    out_offset: 0,
                },
                GroupMember {
                    weight: &layer.wk,
                    out: &mut scratch.k,
                    out_offset: 0,
                },
                GroupMember {
                    weight: &layer.wv,
                    out: &mut scratch.v,
                    out_offset: 0,
                },
            ],
            &scratch.normed_attn,
            0,
            &normed,
            n,
        )?;

        // q, k and v leave the projections together: the query is rotated in
        // place, the key is rotated into the cache, and the value is copied
        // beside it, all in one launch.
        self.compute.rope_append(
            &mut scratch.q,
            &mut scratch.k,
            &scratch.v,
            &mut kv.keys[index],
            &mut kv.values[index],
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
            &kv.keys[index],
            &kv.values[index],
            &mut scratch.attn,
            &mut scratch.scores,
            &spec,
            start,
            n,
        )?;

        self.compute.matmul_add(
            &layer.wo,
            &scratch.attn,
            0,
            None,
            n,
            &mut scratch.x,
            0,
            &mut scratch.projected,
        )?;

        let normed = self.compute.norm_prepare(
            &scratch.x,
            0,
            layer.ffn_norm,
            config.eps,
            n,
            &mut scratch.normed_ffn,
            0,
        )?;
        self.compute.matmul_prepared_group(
            &mut [
                GroupMember {
                    weight: &layer.w_gate,
                    out: &mut scratch.gate,
                    out_offset: 0,
                },
                GroupMember {
                    weight: &layer.w_up,
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
            &layer.w_down,
            &scratch.activated,
            0,
            Some(&activated),
            n,
            &mut scratch.x,
            0,
            &mut scratch.projected,
        )?;
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
    scores: Activation,
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
        let scores = workspace.view("scores");
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
            scores,
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
    fn a_dense_attention_block_is_supported() {
        assert!(check_supported("qwen2", 128, false).is_ok());
        assert!(check_supported("llama", 64, false).is_ok());
    }

    #[test]
    fn a_head_dimension_no_kernel_covers_is_refused_by_name() {
        let error = check_supported("qwen35", 256, false).unwrap_err();
        match error {
            EngineError::Unsupported(message) => {
                assert!(message.contains("head_dim 256"), "{message}");
            }
            other => panic!("expected an unsupported error, got {other:?}"),
        }
    }

    #[test]
    fn a_state_space_block_is_refused_with_the_reason() {
        let error = check_supported("qwen35", 128, true).unwrap_err();
        match error {
            EngineError::Unsupported(message) => {
                assert!(message.contains("state-space"), "{message}");
            }
            other => panic!("expected an unsupported error, got {other:?}"),
        }
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
        }
    }
}
