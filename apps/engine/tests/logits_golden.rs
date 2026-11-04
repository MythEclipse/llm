//! Forward-pass golden tests against llama.cpp's full logit vectors.
//!
//! `llama-debug --save-logits` writes raw f32 logits, so every one of the
//! 130 560 values is compared rather than a truncated dump.
//!
//! Two fixtures on purpose: a single-token prompt (llama.cpp's gemv path) and a
//! five-token prompt (its batched path).
//!
//! Why the tolerance is 0.2 and not 1e-4: ggml dequantizes Q8_0 weights to f16
//! inside its matmul, while this engine keeps the exact `d·Σqᵢxᵢ` dot product.
//! The two agree to about 0.1 on logits whose magnitude is ~9 — that is the
//! accuracy of llama.cpp's path, not of the model. What has to be identical is
//! the decision: argmax, the top-5, and (see `greedy_golden.rs`) the whole
//! 32-token greedy continuation.

use std::path::PathBuf;

use llm_engine::{GgufFile, Model, Session, Tokenizer};
use serde::Deserialize;

#[derive(Deserialize)]
struct Golden {
    prompt: String,
    tokens: Vec<u32>,
    logits_file: String,
}

/// Fixture stem plus the logit budget it can honour.
///
/// The batched prompt gets a wider budget because ggml dequantizes the weights to
/// f16 for that path, so llama.cpp is the less exact side there.
const FIXTURES: [(&str, f32); 3] = [
    ("logits_golden_single", 0.2),
    ("logits_golden", 0.7),
    ("logits_golden_long", 0.7),
];

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn model_path() -> PathBuf {
    std::env::var_os("LLM_MODEL").map_or_else(
        || root().join("models/MiniCPM5-2B-Q8_0.gguf"),
        PathBuf::from,
    )
}

fn golden(stem: &str) -> Golden {
    let path = root().join(format!("tests/fixtures/{stem}.json"));
    serde_json::from_str(&std::fs::read_to_string(path).expect("logits fixture exists"))
        .expect("fixture parses")
}

fn reference_logits(golden: &Golden) -> Vec<f32> {
    let bytes = std::fs::read(root().join("tests/fixtures").join(&golden.logits_file))
        .expect("logits binary exists");
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

fn ranked_ids(values: &[f32]) -> Vec<usize> {
    let mut ranked: Vec<usize> = (0..values.len()).collect();
    ranked.sort_unstable_by(|a, b| values[*b].total_cmp(&values[*a]));
    ranked
}

#[test]
#[ignore = "requires models/MiniCPM5-2B-Q8_0.gguf and recorded fixtures"]
fn prompt_tokens_match_llama_cpp() {
    let gguf = GgufFile::open(&model_path()).expect("model opens");
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("tokenizer builds");
    for (stem, _) in FIXTURES {
        let golden = golden(stem);
        assert_eq!(
            tokenizer.encode(&golden.prompt),
            golden.tokens,
            "{stem}: prompt ids differ"
        );
    }
}

#[test]
#[ignore = "requires models/MiniCPM5-2B-Q8_0.gguf and recorded fixtures"]
fn logits_match_llama_cpp() {
    let gguf = GgufFile::open(&model_path()).expect("model opens");
    let model = Model::load(&gguf).expect("model loads");

    for (stem, budget) in FIXTURES {
        let golden = golden(stem);
        let mut session = Session::new(&model, 512).expect("session builds");
        let logits = session.prefill(&golden.tokens).expect("forward pass runs");
        let reference = reference_logits(&golden);
        assert_eq!(logits.len(), reference.len(), "vocabulary size mismatch");

        let ours = ranked_ids(&logits);
        let theirs = ranked_ids(&reference);
        assert_eq!(ours[0], theirs[0], "{stem}: argmax differs");
        assert_eq!(&ours[..3], &theirs[..3], "{stem}: top-3 ordering differs");

        let overlap = ours[..10]
            .iter()
            .filter(|id| theirs[..10].contains(id))
            .count();
        assert!(
            overlap >= 7,
            "{stem}: top-10 overlap is only {overlap}/10 (near-ties may swap, real disagreements may not)"
        );

        let max_delta = logits
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("{stem}: max |Δ| = {max_delta}");
        assert!(
            max_delta < budget,
            "{stem}: max |Δ| = {max_delta}, budget {budget}"
        );
    }
}
