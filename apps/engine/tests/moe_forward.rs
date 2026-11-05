//! A routed feed-forward has to compute the same function a dense one does.
//!
//! Four tiny GGUF fixtures from `scripts/make_tiny_moe.py` share one set of
//! seeded weights: a dense model, four identical experts routed two at a time,
//! and the same experts routed one at a time. Between them they pin the softmax
//! over the chosen experts and the weighted accumulation — on the CPU path, which
//! is the reference the GPU kernels are compared against. Which experts get
//! chosen is pinned separately, by a unit test on the router itself, because a
//! fixture cannot decide it: the choice depends on the sign of the hidden state.
//!
//! One caveat is built into the engine: a batched matmul quantizes its
//! activations to Q8_0 (as ggml does), while a single-row matmul dots in f32. A
//! routed block runs one row at a time, so comparing it against a batched dense
//! block shows that difference and nothing else. The tests therefore compare like
//! with like, and the one batched comparison carries a loose bound.

use std::path::PathBuf;

use llm_engine::chat::{Role, Turn, render};
use llm_engine::{GgufFile, Model, Session};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name)
}

fn logits(name: &str, tokens: &[u32]) -> (Vec<f32>, usize) {
    let gguf = GgufFile::open(&fixture(name)).expect("fixture opens");
    let model = Model::load_with_device(&gguf, false).expect("model loads");
    let mut session = Session::new(&model, 64).expect("session");
    let logits = session.prefill(tokens).expect("forward pass");
    (logits, model.config.expert_count)
}

fn worst_delta(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

const ONE: [u32; 1] = [7];
const THREE: [u32; 3] = [1, 7, 42];

#[test]
fn the_dense_fixture_reports_no_experts() {
    let (_, experts) = logits("tiny_dense.gguf", &ONE);
    assert_eq!(experts, 0);
}

#[test]
fn identical_experts_reproduce_the_dense_block_for_one_token() {
    let (dense, _) = logits("tiny_dense.gguf", &ONE);
    let (routed, experts) = logits("tiny_moe.gguf", &ONE);
    assert_eq!(experts, 4, "the routed fixture declares four experts");
    let delta = worst_delta(&dense, &routed);
    assert!(delta < 1e-5, "routed feed-forward drifted by {delta:e}");
}

/// Two identical experts at 0.5 each must equal one expert at 1.0: this is the
/// test that fails if the softmax over the chosen experts does not sum to one.
#[test]
fn routing_two_identical_experts_equals_routing_one() {
    let (one, _) = logits("tiny_moe_one.gguf", &THREE);
    let (two, _) = logits("tiny_moe.gguf", &THREE);
    let delta = worst_delta(&one, &two);
    assert!(delta < 1e-5, "routing weights do not sum to one: {delta:e}");
}

/// The batched comparison: same function, different activation precision.
#[test]
fn a_routed_block_matches_the_dense_block_across_a_prompt() {
    let (dense, _) = logits("tiny_dense.gguf", &THREE);
    let (routed, _) = logits("tiny_moe.gguf", &THREE);
    let delta = worst_delta(&dense, &routed);
    assert!(
        delta < 0.05,
        "routed and dense blocks disagree by {delta:e}, more than the quantized-activation path explains"
    );
}

/// A routed model still renders a prompt and answers; the block type does not
/// leak into the session.
#[test]
fn a_routed_model_runs_a_prompt_end_to_end() {
    let gguf = GgufFile::open(&fixture("tiny_moe.gguf")).expect("fixture opens");
    let model = Model::load_with_device(&gguf, false).expect("model loads");
    let mut session = Session::new(&model, 256).expect("session");
    let prompt = render(
        &[
            Turn {
                role: Role::System,
                content: "Kamu asisten yang membantu.",
            },
            Turn {
                role: Role::User,
                content: "halo",
            },
        ],
        false,
    );
    // The fixture ships no tokenizer, so map the prompt onto its vocabulary.
    let ids: Vec<u32> = prompt.bytes().map(|byte| u32::from(byte) % 64).collect();
    let logits = session.prefill(&ids).expect("forward pass");
    assert_eq!(logits.len(), model.config.n_vocab);
    assert!(logits.iter().all(|value| value.is_finite()));
}
