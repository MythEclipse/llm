//! Q8_0 kernels.
//!
//! A block is 34 bytes: one `f16` scale followed by 32 `i8` values. The dot
//! product is `d · Σ qᵢ·xᵢ` accumulated in f32, which is what ggml does.

use half::f16;

pub const Q8_0_BLOCK_ELEMS: usize = 32;
pub const Q8_0_BLOCK_BYTES: usize = 34;

/// Dot product of a quantized row against an f32 vector of the same length.
pub fn dot_q8_0(row: &[u8], x: &[f32]) -> f32 {
    assert_eq!(
        x.len() % Q8_0_BLOCK_ELEMS,
        0,
        "vector length must be a multiple of 32"
    );
    let blocks = x.len() / Q8_0_BLOCK_ELEMS;
    assert_eq!(
        row.len(),
        blocks * Q8_0_BLOCK_BYTES,
        "row byte length mismatch"
    );

    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: both features were checked above.
        return unsafe { dot_q8_0_avx2(row, x, blocks) };
    }

    dot_q8_0_scalar(row, x, blocks)
}

fn block_scale(row: &[u8], block: usize) -> f32 {
    let base = block * Q8_0_BLOCK_BYTES;
    f16::from_bits(u16::from_le_bytes([row[base], row[base + 1]])).to_f32()
}

fn dot_q8_0_scalar(row: &[u8], x: &[f32], blocks: usize) -> f32 {
    let mut total = 0.0f32;
    for block in 0..blocks {
        let base = block * Q8_0_BLOCK_BYTES;
        let quants = &row[base + 2..base + Q8_0_BLOCK_BYTES];
        let values = &x[block * Q8_0_BLOCK_ELEMS..(block + 1) * Q8_0_BLOCK_ELEMS];
        let mut sum = 0.0f32;
        for (q, v) in quants.iter().zip(values) {
            sum += *q as i8 as f32 * *v;
        }
        total += block_scale(row, block) * sum;
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q8_0_avx2(row: &[u8], x: &[f32], blocks: usize) -> f32 {
    use std::arch::x86_64::*;

    let mut total = 0.0f32;
    for block in 0..blocks {
        let base = block * Q8_0_BLOCK_BYTES;
        let mut acc = _mm256_setzero_ps();
        for chunk in 0..4 {
            // SAFETY: the caller bounds-checked 32 quants and 32 f32 values per block.
            // `_mm256_cvtepi8_epi32` widens only 8 bytes, so each step consumes 8 quants.
            unsafe {
                let packed = _mm_loadl_epi64(row.as_ptr().add(base + 2 + chunk * 8).cast());
                let widened = _mm256_cvtepi8_epi32(packed);
                let as_float = _mm256_cvtepi32_ps(widened);
                let vector = _mm256_loadu_ps(x.as_ptr().add(block * Q8_0_BLOCK_ELEMS + chunk * 8));
                acc = _mm256_fmadd_ps(as_float, vector, acc);
            }
        }
        // SAFETY: pure register arithmetic on a value produced above.
        total += block_scale(row, block) * unsafe { horizontal_sum(acc) };
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn horizontal_sum(vector: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;

    let low = _mm256_castps256_ps128(vector);
    let high = _mm256_extractf128_ps(vector, 1);
    let sum = _mm_add_ps(low, high);
    let sum = _mm_hadd_ps(sum, sum);
    let sum = _mm_hadd_ps(sum, sum);
    _mm_cvtss_f32(sum)
}

/// Dequantize a whole row into `out` (used by tests and by the embedding lookup).
pub fn dequant_q8_0(row: &[u8], out: &mut [f32]) {
    assert_eq!(
        out.len() % Q8_0_BLOCK_ELEMS,
        0,
        "output length must be a multiple of 32"
    );
    let blocks = out.len() / Q8_0_BLOCK_ELEMS;
    assert_eq!(
        row.len(),
        blocks * Q8_0_BLOCK_BYTES,
        "row byte length mismatch"
    );
    for block in 0..blocks {
        let base = block * Q8_0_BLOCK_BYTES;
        let scale = block_scale(row, block);
        for i in 0..Q8_0_BLOCK_ELEMS {
            out[block * Q8_0_BLOCK_ELEMS + i] = scale * row[base + 2 + i] as i8 as f32;
        }
    }
}
