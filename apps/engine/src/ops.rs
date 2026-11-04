//! Numeric primitives of the transformer, kept separate so each one can be
//! tested against a hand-computed value.

/// RMSNorm: `x / sqrt(mean(x²) + eps) * weight`.
pub fn rmsnorm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    assert_eq!(x.len(), weight.len(), "rmsnorm weight length mismatch");
    assert_eq!(out.len(), x.len(), "rmsnorm output length mismatch");
    // ggml accumulates the sum of squares in double; doing it in f32 shifts the
    // scale enough to move logits after 42 layers.
    let mean_square =
        (x.iter().map(|value| (value * value) as f64).sum::<f64>() / x.len() as f64) as f32;
    let scale = 1.0 / (mean_square + eps).sqrt();
    for ((slot, value), w) in out.iter_mut().zip(x).zip(weight) {
        *slot = value * scale * w;
    }
}

/// SiLU, the activation inside the SwiGLU feed-forward block.
pub fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

/// Dot product of two f32 slices.
///
/// Attention is the only caller, and its dots are 128 elements long: a plain
/// `iter().zip().map().sum()` folds strictly left to right, so every FMA waits
/// for the previous one and the loop runs at one multiply-add per latency
/// instead of per cycle. Four accumulators plus AVX2 keep the pipeline full.
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot product length mismatch");

    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: both features were checked above.
        return unsafe { dot_f32_avx2(a, b) };
    }

    dot_f32_scalar(a, b)
}

fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sums = [0.0f32; 4];
    for (index, (x, y)) in a.iter().zip(b).enumerate() {
        sums[index % 4] += x * y;
    }
    sums.iter().sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    // SAFETY: the caller guarantees equal lengths, and every offset below is
    // derived from those lengths, so all reads stay in bounds.
    unsafe {
        let lanes = 32;
        let blocks = a.len() / lanes;
        let mut accumulators = [_mm256_setzero_ps(); 4];
        for block in 0..blocks {
            let base = block * lanes;
            for (lane, accumulator) in accumulators.iter_mut().enumerate() {
                let offset = base + lane * 8;
                let lhs = _mm256_loadu_ps(a.as_ptr().add(offset));
                let rhs = _mm256_loadu_ps(b.as_ptr().add(offset));
                *accumulator = _mm256_fmadd_ps(lhs, rhs, *accumulator);
            }
        }
        let paired = _mm256_add_ps(
            _mm256_add_ps(accumulators[0], accumulators[1]),
            _mm256_add_ps(accumulators[2], accumulators[3]),
        );
        let mut total = crate::quant::horizontal_sum(paired);
        for index in blocks * lanes..a.len() {
            total += *a.get_unchecked(index) * *b.get_unchecked(index);
        }
        total
    }
}

pub fn softmax(values: &mut [f32]) {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    let inverse = 1.0 / sum;
    for value in values.iter_mut() {
        *value *= inverse;
    }
}

/// Precomputed rotary tables: cos/sin per position and rotation pair.
///
/// The table is built once per session so the forward pass never calls a
/// transcendental function per token.
pub struct RopeTable {
    half: usize,
    n_rot: usize,
    base: f32,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl RopeTable {
    pub fn new(n_rot: usize, base: f32, max_positions: usize) -> Self {
        assert!(n_rot.is_multiple_of(2), "rope dimension must be even");
        let half = n_rot / 2;
        let mut cos = vec![0.0f32; max_positions * half];
        let mut sin = vec![0.0f32; max_positions * half];
        for position in 0..max_positions {
            for pair in 0..half {
                let frequency = base.powf(-2.0 * pair as f32 / n_rot as f32);
                let angle = position as f32 * frequency;
                cos[position * half + pair] = angle.cos();
                sin[position * half + pair] = angle.sin();
            }
        }
        Self {
            half,
            n_rot,
            base,
            cos,
            sin,
        }
    }

    /// Rotate consecutive `(2i, 2i+1)` pairs inside every head.
    pub fn n_rot(&self) -> usize {
        self.n_rot
    }

    pub fn base(&self) -> f32 {
        self.base
    }

    pub fn apply(&self, x: &mut [f32], n_heads: usize, head_dim: usize, position: usize) {
        let offset = position * self.half;
        for head in 0..n_heads {
            let base = head * head_dim;
            for pair in 0..self.half {
                let cos = self.cos[offset + pair];
                let sin = self.sin[offset + pair];
                let low = x[base + 2 * pair];
                let high = x[base + 2 * pair + 1];
                x[base + 2 * pair] = low * cos - high * sin;
                x[base + 2 * pair + 1] = low * sin + high * cos;
            }
        }
    }
}
