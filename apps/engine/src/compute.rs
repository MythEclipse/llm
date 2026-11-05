//! Where the arithmetic runs.
//!
//! One forward pass, two devices: every tensor the model touches is an
//! [`Activation`], which is either host memory or a device allocation. The
//! choice is made once, when the model loads, so the layer loop below in
//! `model.rs` never asks which backend it is talking to.

use crate::cuda::{AttentionGeometry, DeviceBuffer, Gpu};
use crate::error::EngineError;
use crate::gguf::DType;
use crate::model::Config;
use crate::ops::{self, RopeTable};
use crate::tactics::{AttentionPath, KernelPlan};
use crate::tensor::{Weight, dequant_row, matmul};
use crate::workspace::BufferPlan;

/// A tensor the forward pass reads or writes.
pub enum Activation {
    Host(Vec<f32>),
    Device(DeviceBuffer),
}

impl Activation {
    pub fn len(&self) -> usize {
        match self {
            Self::Host(values) => values.len(),
            Self::Device(buffer) => buffer.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Activations already quantised for the matmuls that read them. `buffer` is
/// `None` on the CPU, where the dot product quantises per row and there is nothing
/// to prepare.
pub struct Prepared {
    buffer: Option<crate::cuda::Prepared>,
}

/// One matmul of a group that shares its activations with its neighbours.
pub struct GroupMember<'a> {
    pub weight: &'a Weight<'a>,
    pub out: &'a mut Activation,
    pub out_offset: usize,
}

pub struct AttentionSpec {
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub kv_dim: usize,
    pub scores_stride: usize,
    pub scale: f32,
    /// Context capacity: the decode grid and its shared window are fixed from
    /// this, so they do not move as the context fills.
    pub capacity: usize,
}

/// The buffers one session reuses, laid out by the plan and allocated once.
///
/// On the device they are windows into a single arena, so a token allocates
/// nothing; on the host each is its own vector, because a `Vec` cannot be a view.
pub struct Workspace {
    arena: Option<std::sync::Arc<crate::cuda::Arena>>,
    plan: BufferPlan,
}

impl Workspace {
    /// The buffer the plan calls `name`.
    pub fn view(&self, name: &str) -> Activation {
        let index = self
            .plan
            .index_of(name)
            .expect("every requested buffer is in the plan");
        match &self.arena {
            Some(arena) => Activation::Device(crate::cuda::view(
                arena,
                self.plan.offset(index),
                self.plan.size(index),
            )),
            None => Activation::Host(vec![0.0; self.plan.size(index)]),
        }
    }

    /// Bytes the whole plan holds.
    pub fn bytes(&self) -> usize {
        self.plan.bytes()
    }
}

/// The device used for the whole forward pass, and the kernel plan resolved for
/// it once, when the model loaded.
pub struct Compute {
    gpu: Option<Gpu>,
    note: Option<String>,
    plan: KernelPlan,
}

impl Compute {
    /// GPU when one is present, not disabled by `LLM_GPU=0`, and has room for
    /// `required_bytes` of weights; otherwise the CPU.
    pub fn auto(config: &Config, required_bytes: usize) -> Self {
        if std::env::var("LLM_GPU").is_ok_and(|value| value == "0") {
            return Self::with_note(None, Some("LLM_GPU=0".to_string()), config);
        }
        let gpu = match Gpu::open() {
            Ok(gpu) => gpu,
            Err(error) => {
                return Self::with_note(None, Some(format!("no CUDA device ({error})")), config);
            }
        };
        // Leave room for the KV cache and activation scratch on top of the weights.
        let headroom = 512 << 20;
        match gpu.free_bytes() {
            Ok(free) if free >= required_bytes + headroom => {
                Self::with_note(Some(gpu), None, config)
            }
            Ok(free) => Self::with_note(
                None,
                Some(format!(
                    "device has {} MiB free, model needs {} MiB",
                    free >> 20,
                    (required_bytes + headroom) >> 20
                )),
                config,
            ),
            Err(error) => Self::with_note(
                None,
                Some(format!("device memory unknown ({error})")),
                config,
            ),
        }
    }

    pub fn cpu(config: &Config) -> Self {
        Self::with_note(None, None, config)
    }

    fn with_note(gpu: Option<Gpu>, note: Option<String>, config: &Config) -> Self {
        let plan = KernelPlan::for_model(config, gpu.as_ref().map(Gpu::name));
        Self { gpu, note, plan }
    }

    /// CPU backend that remembers why the device was abandoned mid-load.
    pub fn with_fallback(error: EngineError, config: &Config) -> Self {
        Self::with_note(None, Some(error.to_string()), config)
    }

    /// Writes the position the kernels read. Called once per forward pass,
    /// before the layer loop: a captured decode step bakes its arguments, so the
    /// position has to arrive through memory.
    pub fn set_position(&self, position: usize, visible: usize) -> Result<(), EngineError> {
        match &self.gpu {
            Some(gpu) => gpu.set_position(position, visible),
            None => Ok(()),
        }
    }

    /// Whether a decode step can be captured at all.
    pub fn can_capture(&self) -> bool {
        self.gpu.is_some()
    }

    /// The decode grid's shape key for `visible` positions. A captured step is
    /// only valid for the grid it was recorded with, so the caller caches one
    /// graph per value.
    pub fn decode_splits(&self, visible: usize) -> usize {
        crate::cuda::split_count(visible)
    }

    pub fn begin_capture(&self) -> Result<(), EngineError> {
        match &self.gpu {
            Some(gpu) => gpu.begin_capture(),
            None => Err(EngineError::Unsupported("no device to capture on".into())),
        }
    }

    pub fn end_capture(&self) -> Result<crate::cuda::Graph, EngineError> {
        match &self.gpu {
            Some(gpu) => gpu.end_capture(),
            None => Err(EngineError::Unsupported("no device to capture on".into())),
        }
    }

    pub fn cancel_capture(&self) {
        if let Some(gpu) = &self.gpu {
            gpu.cancel_capture();
        }
    }

    pub fn replay(&self, graph: &crate::cuda::Graph) -> Result<(), EngineError> {
        match &self.gpu {
            Some(gpu) => gpu.replay(graph),
            None => Ok(()),
        }
    }

    /// The kernels this model resolved to, and why.
    pub fn plan(&self) -> &KernelPlan {
        &self.plan
    }

    /// Allocate the plan in one go: one arena on the device, one vector per
    /// buffer on the host.
    pub fn workspace(&self, plan: BufferPlan) -> Result<Workspace, EngineError> {
        match &self.gpu {
            Some(gpu) => Ok(Workspace {
                arena: Some(gpu.arena(plan.floats())?),
                plan,
            }),
            None => Ok(Workspace { arena: None, plan }),
        }
    }

    /// Why the CPU backend is in use, when it was not a free choice.
    pub fn note(&self) -> Option<&str> {
        self.note.as_deref()
    }

    /// `Some("NVIDIA GeForce RTX 4050 Laptop GPU")` when the GPU path is live.
    pub fn device_name(&self) -> Option<&str> {
        self.gpu.as_ref().map(Gpu::name)
    }

    /// Bytes the engine holds on the device now and at peak, so a benchmark can
    /// report the footprint next to the speed.
    pub fn device_footprint(&self) -> Option<(usize, usize)> {
        self.gpu.as_ref().map(Gpu::device_bytes)
    }

    /// Whether the live backend is the device, so a host timer around a launch
    /// measures GPU work instead of the queueing of it.
    pub fn is_device(&self) -> bool {
        self.gpu.is_some()
    }

    /// Blocks until every launch already queued has finished.
    pub fn synchronize(&self) {
        if let Some(gpu) = &self.gpu {
            gpu.synchronize();
        }
    }

    pub fn device_bytes(&self) -> Option<usize> {
        self.gpu.as_ref().map(Gpu::total_bytes)
    }

    /// Copies every weight to the device up front, so the first request does
    /// not pay for the upload and a device that cannot hold the model is
    /// discovered at load time instead of mid-generation.
    pub fn preload(&self, weights: &[&Weight<'_>]) -> Result<(), EngineError> {
        let Some(gpu) = &self.gpu else {
            return Ok(());
        };
        for weight in weights {
            if weight.dtype == DType::Q8_0 {
                gpu.reserve(weight.identity(), weight.bytes())?;
            }
        }
        Ok(())
    }

    pub fn alloc(&self, len: usize) -> Result<Activation, EngineError> {
        match &self.gpu {
            Some(gpu) => Ok(Activation::Device(gpu.alloc(len)?)),
            None => Ok(Activation::Host(vec![0.0; len])),
        }
    }

    pub fn upload(&self, destination: &mut Activation, source: &[f32]) -> Result<(), EngineError> {
        match (&self.gpu, destination) {
            (Some(gpu), Activation::Device(buffer)) => gpu.upload(buffer, source),
            (None, Activation::Host(values)) => {
                values[..source.len()].copy_from_slice(source);
                Ok(())
            }
            _ => unreachable!("activation belongs to the other backend"),
        }
    }

    pub fn download(
        &self,
        source: &Activation,
        destination: &mut [f32],
    ) -> Result<(), EngineError> {
        match (&self.gpu, source) {
            (Some(gpu), Activation::Device(buffer)) => gpu.download(buffer, destination),
            (None, Activation::Host(values)) => {
                destination.copy_from_slice(&values[..destination.len()]);
                Ok(())
            }
            _ => unreachable!("activation belongs to the other backend"),
        }
    }

    pub fn embedding(
        &self,
        weight: &Weight<'_>,
        token: u32,
        out: &mut Activation,
        out_offset: usize,
    ) -> Result<(), EngineError> {
        match (&self.gpu, out) {
            (Some(gpu), Activation::Device(buffer)) => gpu.embedding(
                weight.identity(),
                weight.bytes(),
                token,
                buffer,
                out_offset,
                weight.k,
            ),
            (None, Activation::Host(values)) => {
                dequant_row(
                    weight,
                    token as usize,
                    &mut values[out_offset..out_offset + weight.k],
                );
                Ok(())
            }
            _ => unreachable!("activation belongs to the other backend"),
        }
    }

    /// Embeddings for a whole prompt in one launch, instead of one per token.
    pub fn embeddings(
        &self,
        weight: &Weight<'_>,
        tokens: &[u32],
        out: &mut Activation,
        out_offset: usize,
    ) -> Result<(), EngineError> {
        match (&self.gpu, out) {
            (Some(gpu), Activation::Device(buffer)) => gpu.embeddings(
                weight.identity(),
                weight.bytes(),
                tokens,
                buffer,
                out_offset,
                weight.k,
            ),
            (None, Activation::Host(values)) => {
                for (index, token) in tokens.iter().enumerate() {
                    let from = out_offset + index * weight.k;
                    dequant_row(weight, *token as usize, &mut values[from..from + weight.k]);
                }
                Ok(())
            }
            _ => unreachable!("activation belongs to the other backend"),
        }
    }

    /// Activations quantised once for a group of matmuls that share them. On the
    /// CPU there is nothing to do: the dot product quantises per row as it goes,
    /// so `buffer` stays `None` and the caller falls back to `matmul`.
    pub fn prepare(
        &self,
        x: &Activation,
        x_offset: usize,
        n: usize,
        k: usize,
    ) -> Result<Prepared, EngineError> {
        // Below the prompt threshold the matmul takes the row-at-a-time kernel,
        // which quantises as it goes and never reads the buffer, so preparing one
        // would be a launch per matmul for nothing.
        if n < self.plan.prepare_min_tokens {
            return Ok(Prepared { buffer: None });
        }
        match (&self.gpu, x) {
            (Some(gpu), Activation::Device(buffer)) => Ok(Prepared {
                buffer: Some(gpu.prepare(buffer, x_offset, n, k)?),
            }),
            _ => Ok(Prepared { buffer: None }),
        }
    }

    /// `matmul` on activations `prepare` already quantised.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_prepared(
        &self,
        weight: &Weight<'_>,
        x: &Activation,
        x_offset: usize,
        prepared: &Prepared,
        n: usize,
        y: &mut Activation,
        y_offset: usize,
    ) -> Result<(), EngineError> {
        self.matmul_with(
            weight,
            x,
            x_offset,
            n,
            y,
            y_offset,
            prepared.buffer.as_ref(),
            None,
        )
    }

    pub fn matmul(
        &self,
        weight: &Weight<'_>,
        x: &Activation,
        x_offset: usize,
        n: usize,
        y: &mut Activation,
        y_offset: usize,
    ) -> Result<(), EngineError> {
        self.matmul_with(weight, x, x_offset, n, y, y_offset, None, None)
    }

    /// Several matmuls that read the same activations, in one launch where the
    /// device allows it. Their row tiles then share a wave: at short prompts a
    /// single projection leaves part of the device idle, and q, k and v are three
    /// of those in a row. On the CPU they simply run one after another.
    pub fn matmul_prepared_group(
        &self,
        members: &mut [GroupMember<'_>],
        x: &Activation,
        x_offset: usize,
        prepared: &Prepared,
        n: usize,
    ) -> Result<(), EngineError> {
        let grouped = self.plan.group_matmul(members.len(), n);
        if let (true, Some(gpu), Activation::Device(source)) = (grouped, &self.gpu, x) {
            let mut entries = Vec::with_capacity(members.len());
            for member in members.iter() {
                let Activation::Device(target) = &member.out else {
                    unreachable!("activation belongs to the other backend");
                };
                entries.push(crate::cuda::GroupEntry {
                    identity: member.weight.identity(),
                    bytes: member.weight.bytes(),
                    out: member.weight.out,
                    y: target,
                    y_offset: member.out_offset,
                });
            }
            let k = members[0].weight.k;
            if n == 1 {
                return gpu.matmul_decode_group(&entries, source, x_offset, k, None);
            }
            if let Some(buffer) = prepared.buffer.as_ref() {
                return gpu.matmul_prompt_group(&entries, buffer, n, k, None);
            }
        }
        for member in members.iter_mut() {
            self.matmul(member.weight, x, x_offset, n, member.out, member.out_offset)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn matmul_with(
        &self,
        weight: &Weight<'_>,
        x: &Activation,
        x_offset: usize,
        n: usize,
        y: &mut Activation,
        y_offset: usize,
        prepared: Option<&crate::cuda::Prepared>,
        bias: Option<(&Activation, usize)>,
    ) -> Result<(), EngineError> {
        match (&self.gpu, x, y) {
            (Some(gpu), Activation::Device(source), Activation::Device(target))
                if weight.dtype == DType::Q8_0 =>
            {
                let bias = match bias {
                    Some((Activation::Device(bias), bias_offset)) => {
                        Some((bias.pointer(), bias_offset))
                    }
                    _ => None,
                };
                gpu.matmul(
                    weight.identity(),
                    weight.bytes(),
                    source,
                    x_offset,
                    n,
                    weight.k,
                    weight.out,
                    target,
                    y_offset,
                    prepared,
                    bias,
                )
            }
            (None, Activation::Host(source), Activation::Host(target)) => {
                matmul(
                    weight,
                    &source[x_offset..x_offset + n * weight.k],
                    n,
                    &mut target[y_offset..y_offset + n * weight.out],
                );
                if let Some((Activation::Host(bias), bias_offset)) = bias {
                    for index in 0..n * weight.out {
                        target[y_offset + index] += bias[bias_offset + index];
                    }
                }
                Ok(())
            }
            (_, Activation::Device(_), _) | (_, _, Activation::Device(_)) => Err(
                EngineError::Unsupported(format!("{} is not quantized as Q8_0", weight.k)),
            ),
            _ => unreachable!("activation belongs to the other backend"),
        }
    }

    /// `y[t·out + o] += Σᵢ x[t·k + i] · W[o·k + i]` — the matmul result added
    /// into `y` in place. Single-token decode fuses the add into the matmul
    /// epilogue; batches keep a separate residual pass, because the prompt
    /// kernel's scattered epilogue loads measured slower than one coalesced
    /// read-modify-write pass. `temp` is the scratch both unfused paths need.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_add(
        &self,
        weight: &Weight<'_>,
        x: &Activation,
        x_offset: usize,
        prepared: Option<&Prepared>,
        n: usize,
        y: &mut Activation,
        y_offset: usize,
        temp: &mut Activation,
    ) -> Result<(), EngineError> {
        // The residual add rides in the matmul epilogue on every device path,
        // decode and prompt alike: writing the product to a scratch buffer and
        // adding it afterwards costs a second pass over the whole hidden state.
        if let (Some(gpu), Activation::Device(source), Activation::Device(target)) =
            (&self.gpu, x, &mut *y)
            && weight.dtype == DType::Q8_0
        {
            let pointer = target.pointer();
            return gpu.matmul(
                weight.identity(),
                weight.bytes(),
                source,
                x_offset,
                n,
                weight.k,
                weight.out,
                target,
                y_offset,
                prepared.and_then(|p| p.buffer.as_ref()),
                Some((pointer, y_offset)),
            );
        }
        self.matmul_with(
            weight,
            x,
            x_offset,
            n,
            temp,
            y_offset,
            prepared.and_then(|p| p.buffer.as_ref()),
            None,
        )?;
        self.residual_add(y, y_offset, temp, y_offset, n * weight.out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rmsnorm(
        &self,
        x: &Activation,
        x_offset: usize,
        weight: &[f32],
        eps: f32,
        rows: usize,
        out: &mut Activation,
        out_offset: usize,
    ) -> Result<(), EngineError> {
        let n = weight.len();
        match (&self.gpu, x, out) {
            (Some(gpu), Activation::Device(source), Activation::Device(target)) => {
                gpu.rmsnorm_rows(source, x_offset, target, out_offset, weight, n, rows, eps)
            }
            (None, Activation::Host(source), Activation::Host(target)) => {
                for row in 0..rows {
                    let from = x_offset + row * n;
                    let to = out_offset + row * n;
                    ops::rmsnorm(
                        &source[from..from + n],
                        weight,
                        eps,
                        &mut target[to..to + n],
                    );
                }
                Ok(())
            }
            _ => unreachable!("activation belongs to the other backend"),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rope(
        &self,
        x: &mut Activation,
        offset: usize,
        table: &RopeTable,
        tokens: usize,
        heads: usize,
        head_dim: usize,
        start: usize,
    ) -> Result<(), EngineError> {
        match (&self.gpu, x) {
            (Some(gpu), Activation::Device(buffer)) => gpu.rope(
                buffer,
                offset,
                tokens,
                heads,
                head_dim,
                head_dim / 2,
                table.n_rot(),
                table.base(),
                start,
            ),
            (None, Activation::Host(values)) => {
                let stride = heads * head_dim;
                for token in 0..tokens {
                    let from = offset + token * stride;
                    table.apply(
                        &mut values[from..from + stride],
                        heads,
                        head_dim,
                        start + token,
                    );
                }
                Ok(())
            }
            _ => unreachable!("activation belongs to the other backend"),
        }
    }

    /// RoPE on the query, RoPE on the key written straight into the cache, and
    /// the value copied beside it. One launch on the device; the host backend
    /// runs the same three passes it would have run anyway.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_append(
        &self,
        query: &mut Activation,
        key: &mut Activation,
        value: &Activation,
        keys: &mut Activation,
        values: &mut Activation,
        table: &RopeTable,
        tokens: usize,
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        start: usize,
    ) -> Result<(), EngineError> {
        if let (
            Some(gpu),
            Activation::Device(query),
            Activation::Device(key),
            Activation::Device(value),
            Activation::Device(keys),
            Activation::Device(values),
        ) = (
            &self.gpu,
            &mut *query,
            &mut *key,
            value,
            &mut *keys,
            &mut *values,
        ) {
            let kv_dim = n_head_kv * head_dim;
            return gpu.rope_append(
                query,
                key,
                value,
                keys,
                values,
                tokens,
                n_head,
                n_head_kv,
                head_dim,
                table.n_rot(),
                table.base(),
                kv_dim,
            );
        }
        let kv_dim = n_head_kv * head_dim;
        self.rope(query, 0, table, tokens, n_head, head_dim, start)?;
        self.rope(key, 0, table, tokens, n_head_kv, head_dim, start)?;
        self.kv_append(keys, values, key, value, tokens, kv_dim, start)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn kv_append(
        &self,
        keys: &mut Activation,
        values: &mut Activation,
        source_keys: &Activation,
        source_values: &Activation,
        tokens: usize,
        kv_dim: usize,
        start: usize,
    ) -> Result<(), EngineError> {
        match (&self.gpu, keys, values, source_keys, source_values) {
            (
                Some(gpu),
                Activation::Device(cache_keys),
                Activation::Device(cache_values),
                Activation::Device(keys_in),
                Activation::Device(values_in),
            ) => gpu.kv_append(
                cache_keys,
                cache_values,
                keys_in,
                values_in,
                tokens,
                kv_dim,
                start,
            ),
            (
                None,
                Activation::Host(cache_keys),
                Activation::Host(cache_values),
                Activation::Host(keys_in),
                Activation::Host(values_in),
            ) => {
                for token in 0..tokens {
                    let source = token * kv_dim;
                    let target = (start + token) * kv_dim;
                    cache_keys[target..target + kv_dim]
                        .copy_from_slice(&keys_in[source..source + kv_dim]);
                    cache_values[target..target + kv_dim]
                        .copy_from_slice(&values_in[source..source + kv_dim]);
                }
                Ok(())
            }
            _ => unreachable!("activation belongs to the other backend"),
        }
    }

    /// Score elements the attention kernel of this backend reads. The tiled
    /// kernels keep their probabilities in registers and shared memory, so they
    /// ask for none; only the serial kernel and the host path write a full grid
    /// of `n_head × tokens × scores_stride`. Sizing the scratch from this instead
    /// of from the grid leaves the largest buffer in the engine unallocated.
    pub fn attention_scores(&self, spec: &AttentionSpec, tokens: usize) -> usize {
        self.plan
            .score_floats(spec.n_head, tokens, spec.scores_stride)
    }

    /// Which attention kernel this call runs. The decision was made at load;
    /// this only picks the regime.
    fn attention_path(&self, tokens: usize) -> AttentionPath {
        self.plan.attention(tokens)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attend(
        &self,
        query: &Activation,
        keys: &Activation,
        values: &Activation,
        out: &mut Activation,
        scores: &mut Activation,
        spec: &AttentionSpec,
        start: usize,
        tokens: usize,
    ) -> Result<(), EngineError> {
        match (&self.gpu, query, keys, values, out, scores) {
            (
                Some(gpu),
                Activation::Device(q),
                Activation::Device(k),
                Activation::Device(v),
                Activation::Device(target),
                Activation::Device(scratch),
            ) => {
                let geometry = AttentionGeometry {
                    n_head: spec.n_head,
                    n_head_kv: spec.n_head_kv,
                    head_dim: spec.head_dim,
                    scores_stride: spec.scores_stride,
                    scale: spec.scale,
                };
                match self.attention_path(tokens) {
                    // Decode: one token has no batch to hide latency behind, so
                    // the positions are spread over blocks instead.
                    AttentionPath::DecodeSplit => gpu.attention_split(
                        q,
                        k,
                        v,
                        target,
                        geometry,
                        start + tokens,
                        spec.capacity,
                    ),
                    // Prefill: sixteen queries per block, keys and values staged
                    // in shared memory, softmax kept online.
                    AttentionPath::PrefillTiled => {
                        gpu.attention_prefill(q, k, v, target, geometry, start, tokens)
                    }
                    // Prefill: eight heads that share one KV head, sixteen
                    // queries, scores on the f16 tensor cores.
                    AttentionPath::PrefillMma => {
                        gpu.attention_prefill_mma(q, k, v, target, geometry, start, tokens)
                    }
                    // The tiled kernel fixes four dimensions per lane, which is
                    // what head_dim 128 gives; anything else keeps the serial one.
                    AttentionPath::Serial => {
                        gpu.attention(q, k, v, target, scratch, geometry, start, tokens)
                    }
                    AttentionPath::Host => unreachable!("device buffers on the host path"),
                }
            }
            (
                None,
                Activation::Host(q),
                Activation::Host(k),
                Activation::Host(v),
                Activation::Host(target),
                Activation::Host(scratch),
            ) => {
                attend_host(q, k, v, target, scratch, spec, start, tokens);
                Ok(())
            }
            _ => unreachable!("activation belongs to the other backend"),
        }
    }

    pub fn silu_mul(
        &self,
        act: &mut Activation,
        gate: &Activation,
        up: &Activation,
        len: usize,
    ) -> Result<(), EngineError> {
        match (&self.gpu, act, gate, up) {
            (
                Some(gpu),
                Activation::Device(target),
                Activation::Device(gate),
                Activation::Device(up),
            ) => gpu.silu_mul(target, gate, up, len),
            (None, Activation::Host(target), Activation::Host(gate), Activation::Host(up)) => {
                for index in 0..len {
                    target[index] = ops::silu(gate[index]) * up[index];
                }
                Ok(())
            }
            _ => unreachable!("activation belongs to the other backend"),
        }
    }

    /// `prepare(rmsnorm(x))` in one kernel: the normalised rows are quantised
    /// where they are computed. Below the prompt threshold the decode matmuls
    /// read the f32 norm, so it is written out and `buffer` stays `None`.
    #[allow(clippy::too_many_arguments)]
    pub fn norm_prepare(
        &self,
        x: &Activation,
        x_offset: usize,
        weight: &[f32],
        eps: f32,
        rows: usize,
        out: &mut Activation,
        out_offset: usize,
    ) -> Result<Prepared, EngineError> {
        let n = weight.len();
        if rows < self.plan.prepare_min_tokens {
            self.rmsnorm(x, x_offset, weight, eps, rows, out, out_offset)?;
            return Ok(Prepared { buffer: None });
        }
        match (&self.gpu, x) {
            (Some(gpu), Activation::Device(source)) => Ok(Prepared {
                buffer: Some(gpu.rmsnorm_prepare(source, x_offset, weight, n, rows, eps)?),
            }),
            (None, Activation::Host(_)) => {
                self.rmsnorm(x, x_offset, weight, eps, rows, out, out_offset)?;
                Ok(Prepared { buffer: None })
            }
            _ => unreachable!("activation belongs to the other backend"),
        }
    }

    /// Activations quantised straight from `silu(gate) * up` for the down
    /// projection, so the f32 product is never written to global memory. The
    /// CPU path computes the product into `act` as before.
    ///
    /// Below the prompt threshold the decode matmul reads the f32 activation
    /// directly, never the prepared buffer — so the plain product is written
    /// and `buffer` stays `None`, exactly like [`Self::prepare`].
    pub fn silu_mul_prepare(
        &self,
        act: &mut Activation,
        gate: &Activation,
        up: &Activation,
        n: usize,
        k: usize,
    ) -> Result<Prepared, EngineError> {
        if n < self.plan.prepare_min_tokens {
            self.silu_mul(act, gate, up, n * k)?;
            return Ok(Prepared { buffer: None });
        }
        match (&self.gpu, act, gate, up) {
            (Some(gpu), Activation::Device(_), Activation::Device(g), Activation::Device(u)) => {
                Ok(Prepared {
                    buffer: Some(gpu.silu_mul_quantize(g, u, n, k)?),
                })
            }
            (None, Activation::Host(target), Activation::Host(g), Activation::Host(u)) => {
                for index in 0..n * k {
                    target[index] = ops::silu(g[index]) * u[index];
                }
                Ok(Prepared { buffer: None })
            }
            _ => unreachable!("activation belongs to the other backend"),
        }
    }

    /// `act[..len] *= factor`, used by routed experts to weight their output
    /// before it is accumulated into the hidden state.
    pub fn scale(&self, act: &mut Activation, factor: f32, len: usize) -> Result<(), EngineError> {
        match (&self.gpu, act) {
            (Some(gpu), Activation::Device(target)) => {
                gpu.axpby_f32(target, 0, target, 0, factor, 0.0, len)
            }
            (None, Activation::Host(values)) => {
                for value in values[..len].iter_mut() {
                    *value *= factor;
                }
                Ok(())
            }
            _ => unreachable!("activation belongs to the other backend"),
        }
    }

    pub fn residual_add(
        &self,
        x: &mut Activation,
        x_offset: usize,
        delta: &Activation,
        delta_offset: usize,
        len: usize,
    ) -> Result<(), EngineError> {
        match (&self.gpu, x, delta) {
            (Some(gpu), Activation::Device(target), Activation::Device(source)) => {
                gpu.residual_add(target, x_offset, source, delta_offset, len)
            }
            (None, Activation::Host(target), Activation::Host(source)) => {
                for index in 0..len {
                    target[x_offset + index] += source[delta_offset + index];
                }
                Ok(())
            }
            _ => unreachable!("activation belongs to the other backend"),
        }
    }
}

impl std::fmt::Debug for Compute {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Compute")
            .field("device", &self.device_name().unwrap_or("cpu"))
            .finish()
    }
}

/// Grouped-query causal attention on the host.
///
/// One pass over the keys per KV head feeds all of its query heads, which is
/// what keeps the key traffic at 1/group of the naive loop.
#[allow(clippy::too_many_arguments)]
fn attend_host(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    out: &mut [f32],
    scores: &mut [f32],
    spec: &AttentionSpec,
    start: usize,
    tokens: usize,
) {
    let group = spec.n_head / spec.n_head_kv;
    let stride = spec.scores_stride;
    for token in 0..tokens {
        let visible = start + token + 1;
        let query_row = &query[token * spec.n_head * spec.head_dim..];
        let out_row = &mut out[token * spec.n_head * spec.head_dim..];

        for kv_head in 0..spec.n_head_kv {
            let first_head = kv_head * group;
            let row_base = (token * spec.n_head + first_head) * stride;
            let group_scores = &mut scores[row_base..row_base + group * stride];

            for offset in 0..visible {
                let key_start = offset * spec.kv_dim + kv_head * spec.head_dim;
                let key = &keys[key_start..key_start + spec.head_dim];
                for head in 0..group {
                    let query_head = (first_head + head) * spec.head_dim;
                    group_scores[head * stride + offset] =
                        ops::dot_f32(&query_row[query_head..query_head + spec.head_dim], key)
                            * spec.scale;
                }
            }

            let out_group =
                &mut out_row[first_head * spec.head_dim..(first_head + group) * spec.head_dim];
            out_group.fill(0.0);
            for head in 0..group {
                ops::softmax(&mut group_scores[head * stride..head * stride + visible]);
            }
            for offset in 0..visible {
                let value_start = offset * spec.kv_dim + kv_head * spec.head_dim;
                let value = &values[value_start..value_start + spec.head_dim];
                for (head, slot) in out_group.chunks_mut(spec.head_dim).enumerate() {
                    let weight = group_scores[head * stride + offset];
                    for (cell, weight_value) in slot.iter_mut().zip(value) {
                        *cell += weight * weight_value;
                    }
                }
            }
        }
    }
}
