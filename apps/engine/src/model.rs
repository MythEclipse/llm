//! The MiniCPM5 transformer: embeddings → 42 decoder layers → logits.

use crate::compute::{Activation, AttentionSpec, Compute, GroupMember};
use crate::error::EngineError;
use crate::gguf::GgufFile;
use crate::ops::RopeTable;
use crate::tensor::{Weight, as_f32};
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
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, EngineError> {
        let config = Self {
            n_layers: gguf.meta_u64("llama.block_count")? as usize,
            n_embd: gguf.meta_u64("llama.embedding_length")? as usize,
            n_ff: gguf.meta_u64("llama.feed_forward_length")? as usize,
            n_head: gguf.meta_u64("llama.attention.head_count")? as usize,
            n_head_kv: gguf.meta_u64("llama.attention.head_count_kv")? as usize,
            head_dim: gguf.meta_u64("llama.attention.key_length")? as usize,
            n_rot: gguf.meta_u64("llama.rope.dimension_count")? as usize,
            rope_base: gguf.meta_f32("llama.rope.freq_base")?,
            eps: gguf.meta_f32("llama.attention.layer_norm_rms_epsilon")?,
            n_vocab: gguf.meta_u64("llama.vocab_size")? as usize,
        };
        if !config.n_head.is_multiple_of(config.n_head_kv) {
            return Err(EngineError::Format(
                "query head count is not a multiple of kv head count".into(),
            ));
        }
        Ok(config)
    }

    pub fn kv_dim(&self) -> usize {
        self.n_head_kv * self.head_dim
    }

    #[allow(dead_code)]
    fn group_size(&self) -> usize {
        self.n_head / self.n_head_kv
    }
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
    #[instrument(skip_all, fields(tokens = tokens.len(), position = kv.len()))]
    pub fn forward(
        &self,
        tokens: &[u32],
        kv: &mut KvCache,
        rope: &RopeTable,
    ) -> Result<Vec<f32>, EngineError> {
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
        debug!(tokens = n, start, "forward pass started");

        let mut x = self.compute.alloc(n * embd)?;
        for token in tokens {
            if *token as usize >= config.n_vocab {
                return Err(EngineError::Format(format!(
                    "token {token} outside vocabulary"
                )));
            }
        }
        self.compute
            .embeddings(&self.token_embd, tokens, &mut x, 0)?;

        let mut scratch = Scratch::new(&self.compute, config, n)?;
        for (index, layer) in self.layers.iter().enumerate() {
            let _layer_span =
                tracing::debug_span!("transformer_layer", layer = index, tokens = n, start)
                    .entered();
            self.layer_forward(layer, index, &mut x, n, start, kv, rope, &mut scratch)?;
        }
        kv.len = start + n;

        self.compute.rmsnorm(
            &x,
            (n - 1) * embd,
            self.output_norm,
            config.eps,
            1,
            &mut scratch.normed,
            0,
        )?;
        let mut logits = self.compute.alloc(config.n_vocab)?;
        self.compute
            .matmul(&self.output, &scratch.normed, 0, 1, &mut logits, 0)?;

        let mut values = vec![0.0f32; config.n_vocab];
        self.compute.download(&logits, &mut values)?;
        debug!(tokens = n, position = start + n, "forward pass complete");
        Ok(values)
    }

    #[allow(clippy::too_many_arguments)]
    fn layer_forward(
        &self,
        layer: &Layer<'a>,
        index: usize,
        x: &mut Activation,
        n: usize,
        start: usize,
        kv: &mut KvCache,
        rope: &RopeTable,
        scratch: &mut Scratch,
    ) -> Result<(), EngineError> {
        let config = &self.config;
        let kv_dim = config.kv_dim();
        let ffn = config.n_ff;

        // q, k and v read the same normalised rows, so the norm is quantised
        // straight from the hidden state — one kernel instead of a write-out,
        // a read-back, and a quantise pass.
        let normed = self.compute.norm_prepare(
            x,
            0,
            layer.attn_norm,
            config.eps,
            n,
            &mut scratch.normed,
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
            &scratch.normed,
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
        };
        let needed = self.compute.attention_scores(&spec, n);
        scratch.ensure_scores(&self.compute, needed)?;
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
            x,
            0,
            &mut scratch.projected,
        )?;

        let normed = self.compute.norm_prepare(
            x,
            0,
            layer.ffn_norm,
            config.eps,
            n,
            &mut scratch.normed,
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
            &scratch.normed,
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
            x,
            0,
            &mut scratch.projected,
        )?;
        Ok(())
    }
}

struct Scratch {
    normed: Activation,
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
    fn new(compute: &Compute, config: &Config, n: usize) -> Result<Self, EngineError> {
        Ok(Self {
            normed: compute.alloc(n * config.n_embd)?,
            q: compute.alloc(n * config.n_embd)?,
            k: compute.alloc(n * config.kv_dim())?,
            v: compute.alloc(n * config.kv_dim())?,
            attn: compute.alloc(n * config.n_embd)?,
            projected: compute.alloc(n * config.n_embd)?,
            gate: compute.alloc(n * config.n_ff)?,
            up: compute.alloc(n * config.n_ff)?,
            activated: compute.alloc(n * config.n_ff)?,
            scores: compute.alloc(1)?,
        })
    }

    /// Attention scratch, grown in powers of two so a long decode does not
    /// reallocate on every token.
    fn ensure_scores(&mut self, compute: &Compute, needed: usize) -> Result<(), EngineError> {
        if self.scores.len() < needed {
            self.scores = compute.alloc(needed.next_power_of_two())?;
        }
        Ok(())
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
