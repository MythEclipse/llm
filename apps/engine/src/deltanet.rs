//! Gated delta net blocks: the recurrent half of a `qwen35` model.
//!
//! One token at a time, the state is a `head_dim × head_dim` matrix per value
//! head, decayed by a gate and corrected by the delta rule. The reference is
//! llama.cpp `d812350493` (`src/models/delta-net-base.cpp` for the graph,
//! `ggml/src/ggml-cpu/ops.cpp` for the arithmetic); the formulas and the layout
//! decisions are written down in `docs/research/qwen35-deltanet.md`.
//!
//! This runs on the host. The projections around it are matmuls and go through
//! [`crate::compute::Compute`] like every other block; the convolution, the
//! delta rule and the gated norm are a few hundred thousand multiply-adds per
//! token, far below the cost of the projections, so a device kernel for them
//! would be measured before it is written.

use crate::model::Config;

/// The shapes a delta net block works in, derived once from the header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Geometry {
    pub conv_kernel: usize,
    /// Key heads: `ssm.group_count`.
    pub k_heads: usize,
    /// Value heads: `ssm.time_step_rank`.
    pub v_heads: usize,
    /// `ssm.state_size`, the width of one key and one value head.
    pub head_dim: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub conv_dim: usize,
}

impl Geometry {
    pub fn from_config(config: &Config) -> Option<Self> {
        let ssm = config.ssm?;
        let head_dim = ssm.state_size;
        if ssm.v_heads == 0 || ssm.k_heads == 0 || head_dim == 0 {
            return None;
        }
        if !ssm.v_heads.is_multiple_of(ssm.k_heads) || !ssm.inner.is_multiple_of(ssm.v_heads) {
            return None;
        }
        let value_dim = ssm.inner;
        let key_dim = head_dim * ssm.k_heads;
        Some(Self {
            conv_kernel: ssm.conv_kernel,
            k_heads: ssm.k_heads,
            v_heads: ssm.v_heads,
            head_dim,
            key_dim,
            value_dim,
            conv_dim: 2 * key_dim + value_dim,
        })
    }

    /// Floats of convolution history one layer carries.
    pub fn conv_state_len(&self) -> usize {
        self.conv_kernel.saturating_sub(1) * self.conv_dim
    }

    /// Floats of delta rule state one layer carries.
    pub fn ssm_state_len(&self) -> usize {
        self.v_heads * self.head_dim * self.head_dim
    }
}

/// Weights of one recurrent block that are read as plain f32.
///
/// The projections are `Weight`s and stay in [`crate::model`]; what is left is
/// the convolution kernel, the two per-head vectors that decide the decay, and
/// the norm that gates the output.
pub struct Weights<'a> {
    pub conv: &'a [f32],
    /// `ssm_a`, already `-exp(A_log)` in the file: negative, so the gate decays.
    pub decay: &'a [f32],
    pub dt: &'a [f32],
    pub norm: &'a [f32],
    /// The model's RMS norm epsilon.
    pub eps: f32,
}

/// The state one layer carries between tokens.
pub struct LayerState<'a> {
    /// `(conv_kernel - 1) × conv_dim`: the inputs the convolution still needs.
    pub conv: &'a mut [f32],
    /// `v_heads × head_dim × head_dim`, each head stored transposed so a row is
    /// the contiguous axis the reference reads (`M[j][i] = S[i][j]`).
    pub ssm: &'a mut [f32],
}

/// Host scratch for one pass, sized for the chunk being processed.
pub struct Buffers {
    /// Projection outputs, `tokens × conv_dim`: the convolution input.
    pub qkv: Vec<f32>,
    /// Gate for the final norm, `tokens × value_dim`.
    pub z: Vec<f32>,
    /// `tokens × v_heads`, before softplus.
    pub alpha: Vec<f32>,
    /// `tokens × v_heads`, after sigmoid.
    pub beta: Vec<f32>,
    /// What the projections feed, `tokens × value_dim`.
    pub out: Vec<f32>,
    /// The delta rule's output before the gated norm, `tokens × value_dim`.
    pub raw: Vec<f32>,
    /// One token of convolved, activated channels.
    pub conv: Vec<f32>,
    /// One token, one head of the delta rule's correction.
    delta: Vec<f32>,
}

impl Buffers {
    pub fn new(geometry: &Geometry, tokens: usize) -> Self {
        Self {
            qkv: vec![0.0; tokens * geometry.conv_dim],
            z: vec![0.0; tokens * geometry.value_dim],
            alpha: vec![0.0; tokens * geometry.v_heads],
            beta: vec![0.0; tokens * geometry.v_heads],
            out: vec![0.0; tokens * geometry.value_dim],
            raw: vec![0.0; tokens * geometry.value_dim],
            conv: vec![0.0; geometry.conv_dim],
            delta: vec![0.0; geometry.head_dim],
        }
    }

    pub fn tokens(&self, geometry: &Geometry) -> usize {
        if geometry.conv_dim == 0 {
            return 0;
        }
        self.qkv.len() / geometry.conv_dim
    }
}

/// Run `tokens` tokens through the block: convolution, delta rule, gated norm.
///
/// Reads [`Buffers::qkv`], [`Buffers::z`], [`Buffers::alpha`], [`Buffers::beta`];
/// writes [`Buffers::out`]. `alpha` must already have been through softplus and
/// `beta` through sigmoid, because both are elementwise and belong with the
/// projection that produced them.
pub fn block(
    geometry: &Geometry,
    weights: &Weights<'_>,
    state: &mut LayerState<'_>,
    buffers: &mut Buffers,
    tokens: usize,
) {
    for token in 0..tokens {
        convolve(geometry, weights, state, buffers, token);
        // The reference L2-normalises the convolved query and key per head —
        // `rms_norm(x, eps/n) * 1/sqrt(n)`, which is `x / sqrt(Σx² + eps)`. The
        // delta rule compares directions, so without it every state update is
        // scaled by the length of the convolved vector.
        l2_normalise(geometry, buffers, weights.eps);
        delta_rule(geometry, weights, state, buffers, token);
    }
}

/// Depthwise causal convolution over the packed `q|k|v` channels, then SiLU.
///
/// The reference sums `input[t + k] * w[k]` — forward, not reversed — over the
/// state's `conv_kernel - 1` rows followed by this token's row, and then keeps the
/// last `conv_kernel - 1` rows as the new state.
fn convolve(
    geometry: &Geometry,
    weights: &Weights<'_>,
    state: &mut LayerState<'_>,
    buffers: &mut Buffers,
    token: usize,
) {
    let channels = geometry.conv_dim;
    let kernel = geometry.conv_kernel;
    let history = kernel.saturating_sub(1);
    let row = &buffers.qkv[token * channels..(token + 1) * channels];

    // The weight is `[channels, kernel]` in memory — the kernel axis is the
    // fastest one, because the tensor is declared as `(kernel, channels)` and
    // ggml's row for a channel is contiguous. Reading it as `[kernel][channel]`
    // gives a model that runs and produces plausible nonsense.
    for (channel, current) in row.iter().enumerate() {
        let weights_for_channel = &weights.conv[channel * kernel..(channel + 1) * kernel];
        let mut sum = 0.0f32;
        for (tap, weight) in weights_for_channel.iter().take(history).enumerate() {
            sum += state.conv[tap * channels + channel] * weight;
        }
        sum += current * weights_for_channel[history];
        buffers.conv[channel] = silu(sum);
    }

    // The window slides by one: drop the oldest row, append this token.
    for tap in 0..history.saturating_sub(1) {
        let (from, to) = (tap + 1, tap);
        state
            .conv
            .copy_within(from * channels..(from + 1) * channels, to * channels);
    }
    if history > 0 {
        state.conv[(history - 1) * channels..].copy_from_slice(row);
    }
}

/// Normalise the convolved query and key heads in place.
fn l2_normalise(geometry: &Geometry, buffers: &mut Buffers, eps: f32) {
    let head_dim = geometry.head_dim;
    for head in 0..geometry.k_heads {
        for start in [
            head * head_dim,
            geometry.key_dim + head * head_dim,
        ] {
            let values = &mut buffers.conv[start..start + head_dim];
            let sum: f32 = values.iter().map(|value| value * value).sum();
            let denom = (sum + eps).sqrt();
            for value in values.iter_mut() {
                *value /= denom;
            }
        }
    }
}

/// The delta rule, one value head at a time.
fn delta_rule(
    geometry: &Geometry,
    weights: &Weights<'_>,
    state: &mut LayerState<'_>,
    buffers: &mut Buffers,
    token: usize,
) {
    let head_dim = geometry.head_dim;
    let key_dim = geometry.key_dim;
    let value_dim = geometry.value_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let convolved = &buffers.conv;

    for head in 0..geometry.v_heads {
        // The keys have fewer heads than the values; the reference repeats them,
        // so value head `h` reads key head `h % k_heads`.
        let key_head = head % geometry.k_heads;
        let q = &convolved[key_head * head_dim..(key_head + 1) * head_dim];
        let k = &convolved[key_dim + key_head * head_dim..key_dim + (key_head + 1) * head_dim];
        let v = &convolved[2 * key_dim + head * head_dim..2 * key_dim + (head + 1) * head_dim];

        let alpha = buffers.alpha[token * geometry.v_heads + head];
        let beta = buffers.beta[token * geometry.v_heads + head];
        let decay = (weights.decay[head] * alpha).exp();

        let matrix = &mut state.ssm[head * head_dim * head_dim..(head + 1) * head_dim * head_dim];
        let delta = &mut buffers.delta;
        let out = &mut buffers.raw
            [token * value_dim + head * head_dim..token * value_dim + (head + 1) * head_dim];

        for value in matrix.iter_mut() {
            *value *= decay;
        }
        // The prediction the state already makes for this value, corrected by
        // beta: what the delta rule adds is the error, not the value itself.
        for row in 0..head_dim {
            let weights_row = &matrix[row * head_dim..(row + 1) * head_dim];
            let predicted: f32 = weights_row.iter().zip(k).map(|(a, b)| a * b).sum();
            delta[row] = (v[row] - predicted) * beta;
        }
        for row in 0..head_dim {
            let weights_row = &mut matrix[row * head_dim..(row + 1) * head_dim];
            let correction = delta[row];
            for (slot, key) in weights_row.iter_mut().zip(k) {
                *slot += key * correction;
            }
        }
        for row in 0..head_dim {
            let weights_row = &matrix[row * head_dim..(row + 1) * head_dim];
            out[row] = weights_row.iter().zip(q).map(|(a, b)| a * b).sum::<f32>() * scale;
        }

        // Gated RMS norm over the head, gated by the projection's own silu.
        let z = &buffers.z[token * value_dim + head * head_dim..token * value_dim + (head + 1) * head_dim];
        let variance: f32 = out.iter().map(|value| value * value).sum::<f32>() / head_dim as f32;
        let denom = (variance + weights.eps).sqrt();
        let destination = &mut buffers.out
            [token * value_dim + head * head_dim..token * value_dim + (head + 1) * head_dim];
        for index in 0..head_dim {
            destination[index] = out[index] / denom * weights.norm[index] * silu(z[index]);
        }
    }
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

/// `log(1 + exp(x))`, guarded the way ggml does: past 20 the two are equal to
/// f32 precision and `exp` would only lose bits.
pub fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        value.exp().ln_1p()
    }
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

/// The decay input: `softplus(alpha + dt)`, per value head.
///
/// The bias is part of the projection, not of the recurrence, which is why it is
/// added here rather than inside the block.
pub fn activate_alpha(values: &mut [f32], dt: &[f32], heads: usize) {
    for (index, value) in values.iter_mut().enumerate() {
        *value = softplus(*value + dt[index % heads]);
    }
}

pub fn activate_beta(values: &mut [f32]) {
    for value in values {
        *value = sigmoid(*value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry() -> Geometry {
        Geometry {
            conv_kernel: 2,
            k_heads: 1,
            v_heads: 1,
            head_dim: 2,
            key_dim: 2,
            value_dim: 2,
            conv_dim: 6,
        }
    }

    #[test]
    fn a_kernel_that_reads_only_the_current_token_passes_it_through() {
        // The weight is `[channel, tap]`: with two taps, tap 0 is the carried
        // state and tap 1 is the token being processed. A kernel of [0, 1] for
        // every channel therefore ignores the state entirely.
        let geometry = Geometry {
            conv_kernel: 2,
            ..geometry()
        };
        let mut conv = vec![0.0; geometry.conv_state_len()];
        let mut ssm = vec![0.0; geometry.ssm_state_len()];
        let weights = Weights {
            conv: &[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            decay: &[-1.0],
            dt: &[0.0],
            norm: &[1.0, 1.0],
            eps: 1e-6,
        };
        let mut buffers = Buffers::new(&geometry, 1);
        buffers.qkv.copy_from_slice(&[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
        block(
            &geometry,
            &weights,
            &mut LayerState {
                conv: &mut conv,
                ssm: &mut ssm,
            },
            &mut buffers,
            1,
        );
        // Channels 0..4 are the query and key heads, which the block normalises;
        // channels 4..6 are the value head, which it does not.
        assert!((buffers.conv[4] - silu(0.5)).abs() < 1e-6, "{:?}", buffers.conv);
        assert!((conv[0] - 0.5).abs() < 1e-6, "state keeps the input row");
    }

    #[test]
    fn the_delta_rule_writes_the_value_into_an_empty_state() {
        // With a zero state the first token's output is the value, scaled by
        // beta and read back through the query.
        let geometry = geometry();
        let mut conv = vec![0.0; geometry.conv_state_len()];
        let mut ssm = vec![0.0; geometry.ssm_state_len()];
        let weights = Weights {
            conv: &[1.0; 6 * 2],
            decay: &[0.0],
            dt: &[0.0],
            norm: &[1.0, 1.0],
            eps: 1e-6,
        };
        let mut buffers = Buffers::new(&geometry, 1);
        // q = [1, 0], k = [1, 0], v = [1, 0] in the packed layout.
        buffers.qkv.copy_from_slice(&[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
        buffers.alpha[0] = 1.0;
        buffers.beta[0] = 1.0;
        buffers.z[0] = 1.0;
        block(
            &geometry,
            &weights,
            &mut LayerState {
                conv: &mut conv,
                ssm: &mut ssm,
            },
            &mut buffers,
            1,
        );
        // The convolution passes the value through silu, the state starts at
        // zero so the correction is the whole value, and the query reads it back
        // with a 1/sqrt(head_dim) scale.
        let value = silu(1.0);
        let expected = value / (geometry.head_dim as f32).sqrt();
        assert!((buffers.raw[0] - expected).abs() < 1e-6, "{:?}", buffers.raw);
        assert!(buffers.raw[1].abs() < 1e-6, "the second row of the state is empty");

        // The gated norm then normalises the head and multiplies by silu(z).
        let normalised = expected / (expected * expected / 2.0 + 1e-6).sqrt();
        let expected_out = normalised * silu(1.0);
        assert!((buffers.out[0] - expected_out).abs() < 1e-5, "{:?}", buffers.out);
    }
}
