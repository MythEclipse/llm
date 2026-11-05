//! The tiled prefill attention against a plain host reference.
//!
//! The golden logits tests say the whole model still lands where llama.cpp lands,
//! but they cannot say *where* a kernel broke: a wrong score shows up as one
//! shifted argmax three hundred lines later. This recomputes attention the obvious
//! way — one query at a time, no tiles, no online softmax — and compares every
//! element, which is what caught an accumulator index that ran past the end of the
//! warp's eight slots.
use llm_engine::{AttentionSpec, Compute, Config};

/// The shape of one attention call, so the reference reads like the formula.
struct Shape {
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    kv_dim: usize,
    scale: f32,
}

fn host(q: &[f32], k: &[f32], v: &[f32], tokens: usize, start: usize, shape: &Shape) -> Vec<f32> {
    let Shape {
        n_head,
        n_head_kv,
        head_dim,
        kv_dim,
        scale,
    } = *shape;
    let mut out = vec![0.0f32; tokens * n_head * head_dim];
    for pos in 0..tokens {
        let visible = start + pos + 1;
        for head in 0..n_head {
            let kv_head = head / (n_head / n_head_kv);
            let mut scores = vec![0.0f32; visible];
            let mut peak = f32::NEG_INFINITY;
            for (offset, score) in scores.iter_mut().enumerate() {
                let mut dot = 0.0f32;
                for dim in 0..head_dim {
                    dot += q[(pos * n_head + head) * head_dim + dim]
                        * k[offset * kv_dim + kv_head * head_dim + dim];
                }
                *score = dot * scale;
                peak = peak.max(*score);
            }
            let mut total = 0.0f32;
            for score in scores.iter_mut() {
                *score = (*score - peak).exp();
                total += *score;
            }
            for dim in 0..head_dim {
                let mut sum = 0.0f32;
                for (offset, score) in scores.iter().enumerate() {
                    sum += score / total * v[offset * kv_dim + kv_head * head_dim + dim];
                }
                out[(pos * n_head + head) * head_dim + dim] = sum;
            }
        }
    }
    out
}

#[test]
#[ignore]
fn tiled_matches_host() {
    let (tokens, start, capacity) = (40usize, 32usize, 96usize);
    let (n_head, n_head_kv, head_dim) = (16usize, 2usize, 128usize);
    let kv_dim = n_head_kv * head_dim;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let query: Vec<f32> = (0..tokens * n_head * head_dim)
        .map(|i| ((i as f32) * 0.013).sin())
        .collect();
    let keys: Vec<f32> = (0..capacity * kv_dim)
        .map(|i| ((i as f32) * 0.007).cos())
        .collect();
    let values: Vec<f32> = (0..capacity * kv_dim)
        .map(|i| ((i as f32) * 0.011).sin())
        .collect();
    let shape = Shape {
        n_head,
        n_head_kv,
        head_dim,
        kv_dim,
        scale,
    };
    let expected = host(&query, &keys, &values, tokens, start, &shape);

    // The shape below is the one the tiled tensor-core kernel is written for,
    // so the plan has to resolve to it.
    let config = Config {
        n_layers: 1,
        n_embd: n_head * head_dim,
        n_ff: 1,
        n_head,
        n_head_kv,
        head_dim,
        n_rot: head_dim,
        rope_base: 5e6,
        eps: 1e-6,
        n_vocab: 1,
        expert_count: 0,
        expert_used: 0,
        expert_ff: 6144,
        full_attention_interval: 0,
        ssm: None,
        qk_norm: false,
        q_gate: false,
    };
    let compute = Compute::auto(&config, 0);
    assert_eq!(
        compute.plan().prefill.tactic,
        "attention.prefill_mma_f16",
        "the fixture has to exercise the kernel it was written for"
    );
    let spec = AttentionSpec {
        n_head,
        n_head_kv,
        head_dim,
        kv_dim,
        scores_stride: capacity,
        scale,
        capacity,
    };
    let mut q = compute.alloc(tokens * n_head * head_dim).unwrap();
    let mut k = compute.alloc(capacity * kv_dim).unwrap();
    let mut v = compute.alloc(capacity * kv_dim).unwrap();
    let mut out = compute.alloc(tokens * n_head * head_dim).unwrap();
    let mut scratch = compute.alloc(tokens * capacity).unwrap();
    compute.upload(&mut q, &query).unwrap();
    compute.upload(&mut k, &keys).unwrap();
    compute.upload(&mut v, &values).unwrap();
    compute
        .attend(&q, &k, &v, &mut out, &mut scratch, &spec, start, tokens)
        .unwrap();
    let mut actual = vec![0.0f32; tokens * n_head * head_dim];
    compute.download(&out, &mut actual).unwrap();

    let mut worst = 0.0f32;
    let mut worst_at = 0usize;
    for (index, (want, got)) in expected.iter().zip(&actual).enumerate() {
        if (want - got).abs() > worst {
            worst = (want - got).abs();
            worst_at = index;
        }
    }
    println!(
        "worst |Δ| = {worst:e} at element {worst_at} (token {} head {})",
        worst_at / (n_head * head_dim),
        (worst_at / head_dim) % n_head
    );
    assert!(worst < 1e-4, "drifted: {worst:e}");
}
