//! Tokenizer golden test: every corpus line must produce llama.cpp's exact ids.
//!
//! A tokenizer that is 99% right is wrong — one wrong boundary changes every
//! downstream id — so this asserts equality on the whole corpus, not a sample.

use std::path::PathBuf;

use llm_engine::{GgufFile, Tokenizer};
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    text: String,
    ids: Vec<u32>,
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn model_path() -> PathBuf {
    std::env::var_os("LLM_MODEL").map_or_else(
        || root().join("models/MiniCPM5-2B-Q8_0.gguf"),
        PathBuf::from,
    )
}

#[test]
#[ignore = "requires models/MiniCPM5-2B-Q8_0.gguf and recorded fixtures"]
fn encode_matches_llama_cpp_for_the_whole_corpus() {
    let gguf = GgufFile::open(&model_path()).expect("model opens");
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("tokenizer builds");

    let golden_path = root().join("tests/fixtures/tokenizer_golden.json");
    let golden: Vec<Case> =
        serde_json::from_str(&std::fs::read_to_string(&golden_path).expect("fixture exists"))
            .expect("fixture parses");
    assert!(golden.len() >= 30, "corpus should stay broad");

    let mut failures = Vec::new();
    for case in &golden {
        let actual = tokenizer.encode(&case.text);
        if actual != case.ids {
            failures.push(format!(
                "text {:?}\n  expected {:?}\n  actual   {:?}",
                case.text, case.ids, actual
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} corpus cases diverged from llama.cpp:\n{}",
        failures.len(),
        golden.len(),
        failures.join("\n")
    );
}

#[test]
#[ignore = "requires models/MiniCPM5-2B-Q8_0.gguf and recorded fixtures"]
fn decode_round_trips_every_corpus_line() {
    let gguf = GgufFile::open(&model_path()).expect("model opens");
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("tokenizer builds");

    let golden_path = root().join("tests/fixtures/tokenizer_golden.json");
    let golden: Vec<Case> =
        serde_json::from_str(&std::fs::read_to_string(&golden_path).expect("fixture exists"))
            .expect("fixture parses");

    for case in &golden {
        assert_eq!(tokenizer.decode(&case.ids), case.text, "round trip failed");
    }
}

#[test]
#[ignore = "requires models/MiniCPM5-2B-Q8_0.gguf"]
fn special_tokens_are_matched_before_bpe() {
    let gguf = GgufFile::open(&model_path()).expect("model opens");
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("tokenizer builds");

    let start = tokenizer.id_of("<|im_start|>").expect("chat marker exists");
    let end = tokenizer.id_of("<|im_end|>").expect("chat marker exists");
    assert_eq!(
        tokenizer.encode("<|im_start|>"),
        vec![start],
        "a chat marker must stay one token"
    );
    assert_eq!(tokenizer.encode("<|im_end|>"), vec![end]);
    assert_eq!(tokenizer.decode(&[start, end]), "<|im_start|><|im_end|>");
}
