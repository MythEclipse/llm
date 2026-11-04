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

/// Quantize an f32 vector to Q8_0: one f32 scale per 32 elements.
///
/// This is what makes an int8 dot possible at all — ggml does the same for the
/// activation side, and it is the reason its prefill is not limited by the
/// int8-to-f32 widening our f32 path needs.
pub fn quantize_q8_0(x: &[f32], quants: &mut [i8], scales: &mut [f32]) {
    assert_eq!(
        x.len() % Q8_0_BLOCK_ELEMS,
        0,
        "vector length must be a multiple of 32"
    );
    let blocks = x.len() / Q8_0_BLOCK_ELEMS;
    assert_eq!(quants.len(), x.len(), "quant buffer length mismatch");
    assert_eq!(scales.len(), blocks, "scale buffer length mismatch");

    for block in 0..blocks {
        let values = &x[block * Q8_0_BLOCK_ELEMS..(block + 1) * Q8_0_BLOCK_ELEMS];
        let peak = values
            .iter()
            .fold(0.0f32, |peak, value| peak.max(value.abs()));
        let scale = peak / 127.0;
        let inverse = if peak > 0.0 { 127.0 / peak } else { 0.0 };
        scales[block] = scale;
        let base = block * Q8_0_BLOCK_ELEMS;
        for (slot, value) in quants[base..base + Q8_0_BLOCK_ELEMS].iter_mut().zip(values) {
            *slot = (value * inverse).round() as i8;
        }
    }
}

/// Dot product of a Q8_0 row against an already-quantized activation vector.
///
/// Both sides are int8 here, so the multiply is `maddubs` instead of a widening
/// to f32: 32 products per instruction pair rather than 8.
pub fn dot_q8_0_q8_0(row: &[u8], quants: &[i8], scales: &[f32]) -> f32 {
    let blocks = quants.len() / Q8_0_BLOCK_ELEMS;
    assert_eq!(
        row.len(),
        blocks * Q8_0_BLOCK_BYTES,
        "row byte length mismatch"
    );
    assert_eq!(scales.len(), blocks, "scale buffer length mismatch");

    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: both features were checked above.
        return unsafe { dot_q8_0_q8_0_avx2(row, quants, scales, blocks) };
    }

    let mut total = 0.0f32;
    for (block, scale) in scales.iter().enumerate() {
        let base = block * Q8_0_BLOCK_BYTES;
        let weights = &row[base + 2..base + Q8_0_BLOCK_BYTES];
        let values = &quants[block * Q8_0_BLOCK_ELEMS..(block + 1) * Q8_0_BLOCK_ELEMS];
        let mut sum = 0i32;
        for (weight, value) in weights.iter().zip(values) {
            sum += *weight as i8 as i32 * *value as i32;
        }
        total += block_scale(row, block) * scale * sum as f32;
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q8_0_q8_0_avx2(row: &[u8], quants: &[i8], scales: &[f32], blocks: usize) -> f32 {
    use std::arch::x86_64::*;

    let mut acc = _mm256_setzero_ps();
    for (block, scale) in scales.iter().enumerate().take(blocks) {
        let base = block * Q8_0_BLOCK_BYTES;
        // SAFETY: the caller bounds-checked 32 quants and 32 weight bytes per
        // block, and `maddubs` needs the unsigned side bounded — |q| <= 127 and
        // |w| <= 127 keep the i16 pair sum under 32767, which is why ggml takes
        // the absolute value here and folds the sign into the other operand.
        unsafe {
            let weight = _mm256_loadu_si256(row.as_ptr().add(base + 2).cast());
            let value = _mm256_loadu_si256(quants.as_ptr().add(block * Q8_0_BLOCK_ELEMS).cast());
            let magnitude = _mm256_sign_epi8(value, value);
            let signed_weight = _mm256_sign_epi8(weight, value);
            let pairs = _mm256_maddubs_epi16(magnitude, signed_weight);
            let sums = _mm256_madd_epi16(pairs, _mm256_set1_epi16(1));
            let combined = _mm256_set1_ps(block_scale(row, block) * scale);
            acc = _mm256_fmadd_ps(combined, _mm256_cvtepi32_ps(sums), acc);
        }
    }
    // SAFETY: pure register arithmetic on a value produced above.
    unsafe { horizontal_sum(acc) }
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
