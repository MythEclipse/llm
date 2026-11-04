//! Batched matmul: decode (n = 1) and prefill (n > 1) against a naive reference.

use half::f16;
use llm_engine::gguf::DType;
use llm_engine::quant::Q8_0_BLOCK_BYTES;
use llm_engine::tensor::{Weight, matmul};

const K: usize = 32;
const OUT: usize = 4;

/// Deterministic Q8_0 matrix, `out × k`, one block per row.
fn quantized_matrix() -> Vec<u8> {
    let mut bytes = Vec::with_capacity(OUT * Q8_0_BLOCK_BYTES);
    for o in 0..OUT {
        bytes.extend_from_slice(
            &f16::from_f32(0.125 * (o as f32 + 1.0))
                .to_bits()
                .to_le_bytes(),
        );
        for i in 0..K {
            bytes.push((((o * 7 + i * 3) % 21) as i32 - 10) as i8 as u8);
        }
    }
    bytes
}

fn reference(data: &[u8], x: &[f32], n: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; n * OUT];
    for t in 0..n {
        for o in 0..OUT {
            let base = o * Q8_0_BLOCK_BYTES;
            let scale = f16::from_bits(u16::from_le_bytes([data[base], data[base + 1]])).to_f32();
            let mut sum = 0.0f64;
            for i in 0..K {
                let q = data[base + 2 + i] as i8 as f64;
                sum += q * x[t * K + i] as f64;
            }
            y[t * OUT + o] = (scale as f64 * sum) as f32;
        }
    }
    y
}

fn activations(n: usize) -> Vec<f32> {
    (0..n * K).map(|i| (i as f32 * 0.017).sin()).collect()
}

/// The batched path quantizes each activation row to int8 first, exactly as ggml
/// does, so the reference has to quantize too. Comparing it against the f32 dot
/// instead would only be measuring that difference.
fn quantized_reference(data: &[u8], x: &[f32], n: usize) -> Vec<f32> {
    let blocks = K / 32;
    let mut quants = vec![0i8; K];
    let mut scales = vec![0.0f32; blocks];
    let mut y = vec![0.0f32; n * OUT];
    for t in 0..n {
        llm_engine::quant::quantize_q8_0(&x[t * K..(t + 1) * K], &mut quants, &mut scales);
        for o in 0..OUT {
            let base = o * Q8_0_BLOCK_BYTES;
            let scale = f16::from_bits(u16::from_le_bytes([data[base], data[base + 1]])).to_f32();
            let mut sum = 0.0f64;
            for block in 0..blocks {
                let mut inner = 0i64;
                for i in 0..32 {
                    let q = data[base + 2 + block * 32 + i] as i8 as i64;
                    inner += q * quants[block * 32 + i] as i64;
                }
                sum += inner as f64 * scales[block] as f64;
            }
            y[t * OUT + o] = (scale as f64 * sum) as f32;
        }
    }
    y
}

#[test]
fn decode_step_matches_reference() {
    let data = quantized_matrix();
    let weight = Weight::new(DType::Q8_0, K, OUT, &data);
    let x = activations(1);
    let mut y = vec![0.0f32; OUT];

    matmul(&weight, &x, 1, &mut y);

    let expected = reference(&data, &x, 1);
    for (actual, want) in y.iter().zip(&expected) {
        assert!((actual - want).abs() < 1e-3, "got {actual}, want {want}");
    }
}

#[test]
fn prefill_batch_matches_reference() {
    let data = quantized_matrix();
    let weight = Weight::new(DType::Q8_0, K, OUT, &data);
    let n = 5;
    let x = activations(n);
    let mut y = vec![0.0f32; n * OUT];

    matmul(&weight, &x, n, &mut y);

    let expected = quantized_reference(&data, &x, n);
    for (actual, want) in y.iter().zip(&expected) {
        assert!((actual - want).abs() < 1e-3, "got {actual}, want {want}");
    }
}

#[test]
fn batch_and_single_agree_up_to_activation_quantization() {
    let data = quantized_matrix();
    let weight = Weight::new(DType::Q8_0, K, OUT, &data);
    let n = 3;
    let x = activations(n);

    let mut batched = vec![0.0f32; n * OUT];
    matmul(&weight, &x, n, &mut batched);

    for t in 0..n {
        let mut single = vec![0.0f32; OUT];
        matmul(&weight, &x[t * K..(t + 1) * K], 1, &mut single);
        // Decode keeps the activation in f32 and prefill rounds it to int8, so
        // these differ by one quantization step — the same asymmetry llama.cpp
        // has between its gemv and batched paths. The step scales with the row,
        // hence a tolerance taken from the largest output of that token.
        let row = single
            .iter()
            .fold(0.0f32, |peak, value| peak.max(value.abs()));
        for o in 0..OUT {
            let tolerance = 0.02 * row.max(1.0);
            assert!(
                (single[o] - batched[t * OUT + o]).abs() < tolerance,
                "token {t} output {o}: single {single:?} batched {batched:?}"
            );
        }
    }
}

#[test]
fn f32_weights_take_the_same_path() {
    let values: Vec<f32> = (0..OUT * K).map(|i| (i as f32) * 0.01).collect();
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let weight = Weight::new(DType::F32, K, OUT, &bytes);
    let x = activations(1);
    let mut y = vec![0.0f32; OUT];

    matmul(&weight, &x, 1, &mut y);

    for o in 0..OUT {
        let expected: f32 = (0..K).map(|i| values[o * K + i] * x[i]).sum();
        assert!((y[o] - expected).abs() < 1e-3);
    }
}
