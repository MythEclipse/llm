//! Batched matrix-vector multiplication over quantized weights.
//!
//! One kernel serves both decode (`n = 1`) and prompt prefill (`n = chunk`).
//! Parallelism is over output rows so each weight row is read once per call
//! regardless of `n`; a per-token parallel loop would re-read the whole matrix
//! for every prompt token.

use half::f16;
use rayon::prelude::*;

use crate::gguf::DType;
use crate::quant::{dequant_q8_0, dot_q8_0};

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

    // Transposed scratch so the parallel loop can own one output row at a time.
    let mut scratch = vec![0.0f32; weight.out * n];
    scratch
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(o, row_out)| {
            for (t, slot) in row_out.iter_mut().enumerate() {
                *slot = weight.dot_row(o, &x[t * weight.k..(t + 1) * weight.k]);
            }
        });

    for o in 0..weight.out {
        for t in 0..n {
            y[t * weight.out + o] = scratch[o * n + t];
        }
    }
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
