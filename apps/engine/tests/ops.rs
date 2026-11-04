//! Numeric primitives against hand-computed values.

use llm_engine::ops::{RopeTable, dot_f32, rmsnorm, silu, softmax};

#[test]
fn rmsnorm_scales_by_root_mean_square() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let weight = [1.0, 1.0, 1.0, 1.0];
    let mut out = [0.0; 4];

    rmsnorm(&x, &weight, 0.0, &mut out);

    // mean square = 7.5, scale = 1/sqrt(7.5)
    let scale = 1.0 / 7.5f32.sqrt();
    for (actual, value) in out.iter().zip(&x) {
        assert!((actual - value * scale).abs() < 1e-6);
    }
}

#[test]
fn rmsnorm_applies_the_weight_after_normalizing() {
    let x = [2.0, 2.0];
    let weight = [3.0, 0.5];
    let mut out = [0.0; 2];

    rmsnorm(&x, &weight, 0.0, &mut out);

    assert!((out[0] - 3.0).abs() < 1e-6);
    assert!((out[1] - 0.5).abs() < 1e-6);
}

#[test]
fn silu_matches_reference_points() {
    assert!((silu(0.0) - 0.0).abs() < 1e-6);
    assert!((silu(1.0) - 0.731_058_6).abs() < 1e-6);
    assert!((silu(-1.0) + 0.268_941_4).abs() < 1e-6);
}

#[test]
fn softmax_sums_to_one_and_preserves_order() {
    let mut values = [1.0, 2.0, 3.0];
    softmax(&mut values);

    let sum: f32 = values.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
    assert!(values[0] < values[1] && values[1] < values[2]);
}

#[test]
fn softmax_is_uniform_for_equal_logits() {
    let mut values = [0.0; 4];
    softmax(&mut values);
    for value in values {
        assert!((value - 0.25).abs() < 1e-6);
    }
}

#[test]
fn rope_at_position_zero_is_identity() {
    let table = RopeTable::new(4, 10_000.0, 4);
    let mut x = [0.5, -1.5, 2.0, -0.25];
    let original = x;

    table.apply(&mut x, 1, 4, 0);

    for (actual, expected) in x.iter().zip(&original) {
        assert!((actual - expected).abs() < 1e-6);
    }
}

#[test]
fn rope_rotates_the_consecutive_pair() {
    // base^(0) = 1, so at position 1 the first pair rotates by exactly 1 radian.
    let table = RopeTable::new(2, 10_000.0, 2);
    let mut x = [1.0, 0.0];

    table.apply(&mut x, 1, 2, 1);

    assert!((x[0] - 1.0f32.cos()).abs() < 1e-5);
    assert!((x[1] - 1.0f32.sin()).abs() < 1e-5);
}

#[test]
fn rope_preserves_pair_norm() {
    let table = RopeTable::new(8, 5_000_000.0, 16);
    let mut x = [0.3, -0.7, 1.1, 0.2, -0.4, 0.9, 0.05, -1.3];
    let before: f32 = x.iter().map(|v| v * v).sum();

    table.apply(&mut x, 1, 8, 7);

    let after: f32 = x.iter().map(|v| v * v).sum();
    assert!((before - after).abs() < 1e-4);
}

#[test]
fn dot_f32_matches_the_naive_sum_for_every_tail_length() {
    let a: Vec<f32> = (0..40).map(|index| (index as f32 * 0.37).sin()).collect();
    let b: Vec<f32> = (0..40).map(|index| (index as f32 * 0.11).cos()).collect();

    for length in 1..=40 {
        let expected: f64 = a[..length]
            .iter()
            .zip(&b[..length])
            .map(|(x, y)| f64::from(*x) * f64::from(*y))
            .sum();
        let actual = dot_f32(&a[..length], &b[..length]);
        assert!(
            (f64::from(actual) - expected).abs() < 1e-5,
            "length {length}: {actual} vs {expected}"
        );
    }
}

#[test]
fn dot_f32_handles_the_attention_length() {
    let a: Vec<f32> = (0..128)
        .map(|index| ((index * 7) % 13) as f32 - 6.0)
        .collect();
    let b: Vec<f32> = (0..128)
        .map(|index| ((index * 5) % 11) as f32 - 5.0)
        .collect();
    let expected: f64 = a
        .iter()
        .zip(&b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();

    let actual = f64::from(dot_f32(&a, &b));

    assert!((actual - expected).abs() < 1e-4, "{actual} vs {expected}");
}
