//! Batched matrix-vector multiplication over quantized weights.
//!
//! One kernel serves both decode (`n = 1`) and prompt prefill (`n = chunk`).
//! Parallelism is over output rows so each weight row is read once per call
//! regardless of `n`; a per-token parallel loop would re-read the whole matrix
//! for every prompt token.

use half::f16;
use rayon::prelude::*;

use crate::gguf::DType;
use crate::quant::{Q8_0_BLOCK_ELEMS, dequant_q8_0, dot_q8_0, dot_q8_0_q8_0, quantize_q8_0};

/// A weight matrix `out × k` still in its quantized form.
pub struct Weight<'a> {
    pub dtype: DType,
    pub k: usize,
    pub out: usize,
    data: &'a [u8],
}

impl<'a> Weight<'a> {
    pub fn new(dtype: DType, k: usize, out: usize, data: &'a [u8]) -> Self {
        let expected = dtype.byte_len((k * out) as u64) as usize;
        assert_eq!(data.len(), expected, "weight byte length mismatch");
        Self {
            dtype,
            k,
            out,
            data,
        }
    }

    /// Raw quantized payload, for backends that copy weights to a device.
    pub fn bytes(&self) -> &'a [u8] {
        self.data
    }

    /// Stable key for this weight slice: (address, length).
    pub fn identity(&self) -> (usize, usize) {
        (self.data.as_ptr() as usize, self.data.len())
    }

    fn row_bytes(&self) -> usize {
        self.dtype.byte_len(self.k as u64) as usize
    }

    fn row(&self, o: usize) -> &'a [u8] {
        let len = self.row_bytes();
        &self.data[o * len..(o + 1) * len]
    }

    /// Dot product of output row `o` against one activation vector.
    /// Dot of a Q8_0 row against a pre-quantized activation, so both sides stay
    /// int8. Non-quantized weights fall back to the f32 path.
    fn dot_row_quantized(&self, o: usize, quants: &[i8], scales: &[f32]) -> f32 {
        match self.dtype {
            DType::Q8_0 => dot_q8_0_q8_0(self.row(o), quants, scales),
            _ => {
                let mut dequantized = vec![0.0f32; self.k];
                for (slot, (value, scale)) in dequantized.iter_mut().zip(
                    quants.iter().zip(
                        scales
                            .iter()
                            .flat_map(|scale| std::iter::repeat_n(*scale, 32)),
                    ),
                ) {
                    *slot = *value as f32 * scale;
                }
                self.dot_row(o, &dequantized)
            }
        }
    }

    fn dot_row(&self, o: usize, x: &[f32]) -> f32 {
        let row = self.row(o);
        match self.dtype {
            DType::Q8_0 => dot_q8_0(row, x),
            DType::F32 => dot_f32(as_f32(row), x),
            DType::F16 => {
                let mut dequantized = vec![0.0f32; self.k];
                for (slot, pair) in dequantized.iter_mut().zip(row.as_chunks::<2>().0) {
                    *slot = f16::from_bits(u16::from_le_bytes(*pair)).to_f32();
                }
                dot_f32(&dequantized, x)
            }
        }
    }
}

/// `y[t·out + o] = Σᵢ x[t·k + i] · W[o·k + i]`.
pub fn matmul(weight: &Weight<'_>, x: &[f32], n: usize, y: &mut [f32]) {
    assert_eq!(x.len(), n * weight.k, "activation shape mismatch");
    assert_eq!(y.len(), n * weight.out, "output shape mismatch");

    if n == 1 {
        // Decode writes straight into `y`: no scratch buffer, no transpose.
        y.par_iter_mut()
            .enumerate()
            .for_each(|(o, slot)| *slot = weight.dot_row(o, x));
        return;
    }

    // Quantize each token's activation once, then every output row dots int8
    // against int8 — the same trick ggml uses to keep prefill off the
    // int8-to-f32 widening path.
    let blocks = weight.k / Q8_0_BLOCK_ELEMS;
    let mut quants = vec![0i8; n * weight.k];
    let mut scales = vec![0.0f32; n * blocks];
    if n >= 8 {
        x.par_chunks(weight.k)
            .zip(quants.par_chunks_mut(weight.k))
            .zip(scales.par_chunks_mut(blocks))
            .for_each(|((values, token_quants), token_scales)| {
                quantize_q8_0(values, token_quants, token_scales);
            });
    } else {
        for t in 0..n {
            quantize_q8_0(
                &x[t * weight.k..(t + 1) * weight.k],
                &mut quants[t * weight.k..(t + 1) * weight.k],
                &mut scales[t * blocks..(t + 1) * blocks],
            );
        }
    }

    // Transposed scratch so the parallel loop can own one output row at a time.
    let mut scratch = vec![0.0f32; weight.out * n];
    scratch
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(o, row_out)| {
            for (t, slot) in row_out.iter_mut().enumerate() {
                *slot = weight.dot_row_quantized(
                    o,
                    &quants[t * weight.k..(t + 1) * weight.k],
                    &scales[t * blocks..(t + 1) * blocks],
                );
            }
        });

    y.par_chunks_mut(weight.out)
        .enumerate()
        .for_each(|(t, row_out)| {
            for (o, slot) in row_out.iter_mut().enumerate() {
                *slot = scratch[o * n + t];
            }
        });
}

fn dot_f32(weights: &[f32], x: &[f32]) -> f32 {
    weights.iter().zip(x).map(|(w, v)| w * v).sum()
}

/// Reinterpret an f32 tensor payload borrowed from the mmap.
pub fn as_f32(bytes: &[u8]) -> &[f32] {
    // SAFETY: f32 rows come straight from the mmap and the slice length is a
    // multiple of 4 because the tensor was validated against its dtype.
    unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<f32>(), bytes.len() / 4) }
}

/// Row of an embedding matrix dequantized into `out`.
pub fn dequant_row(weight: &Weight<'_>, o: usize, out: &mut [f32]) {
    assert_eq!(out.len(), weight.k, "embedding row length mismatch");
    let row = weight.row(o);
    match weight.dtype {
        DType::Q8_0 => dequant_q8_0(row, out),
        DType::F32 => out.copy_from_slice(as_f32(row)),
        DType::F16 => {
            for (slot, pair) in out.iter_mut().zip(row.as_chunks::<2>().0) {
                *slot = f16::from_bits(u16::from_le_bytes(*pair)).to_f32();
            }
        }
    }
}
