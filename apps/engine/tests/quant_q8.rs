//! Q8_0 kernel correctness: scalar and AVX2 paths against a f64 reference.

use half::f16;
use llm_engine::quant::{Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMS, dequant_q8_0, dot_q8_0};

fn block(scale: f32, quants: &[i8; 32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(Q8_0_BLOCK_BYTES);
    bytes.extend_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
    bytes.extend(quants.iter().map(|q| *q as u8));
    bytes
}

#[test]
fn dot_matches_f64_reference() {
    let quants: [i8; 32] = std::array::from_fn(|i| (i as i8) - 16);
    let row = block(0.5, &quants);
    let x: Vec<f32> = (0..Q8_0_BLOCK_ELEMS)
        .map(|i| (i as f32) * 0.125 - 1.0)
        .collect();

    let expected: f64 = quants
        .iter()
        .zip(&x)
        .map(|(q, v)| *q as f64 * *v as f64)
        .sum::<f64>()
        * 0.5;

    let actual = dot_q8_0(&row, &x);
    assert!(
        (actual as f64 - expected).abs() < 1e-4,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn dot_spans_multiple_blocks_with_per_block_scale() {
    let quants: [i8; 32] = [1; 32];
    let mut row = block(0.25, &quants);
    row.extend(block(2.0, &quants));
    let x = vec![1.0f32; 64];

    // 0.25·32 + 2.0·32
    assert!((dot_q8_0(&row, &x) - 72.0).abs() < 1e-4);
}

#[test]
fn dequant_row_round_trips_through_dot() {
    let quants: [i8; 32] = std::array::from_fn(|i| if i % 2 == 0 { 3 } else { -7 });
    let row = block(1.5, &quants);

    let mut values = vec![0.0f32; Q8_0_BLOCK_ELEMS];
    dequant_q8_0(&row, &mut values);

    let expected: f32 = quants
        .iter()
        .map(|q| *q as f32 * 1.5)
        .collect::<Vec<_>>()
        .iter()
        .sum();
    let actual: f32 = values.iter().sum();
    assert!((actual - expected).abs() < 1e-4);
    assert!((dot_q8_0(&row, &values) - values.iter().map(|v| v * v).sum::<f32>()).abs() < 1e-2);
}
