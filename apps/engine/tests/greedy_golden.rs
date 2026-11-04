//! End-to-end greedy check: our sampler must reproduce llama.cpp's continuation.

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

use llm_engine::{
    Event, GenerationRequest, GgufFile, Model, SamplingParams, Session, Tokenizer, generate,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Golden {
    prompt: String,
    max_tokens: usize,
    text: String,
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
fn greedy_continuation_matches_llama_cpp() {
    let gguf = GgufFile::open(&model_path()).expect("model opens");
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("tokenizer builds");
    let model = Model::load(&gguf).expect("model loads");
    let mut session = Session::new(&model, 512).expect("session builds");

    let golden: Golden = serde_json::from_str(
        &std::fs::read_to_string(root().join("tests/fixtures/greedy_golden.json"))
            .expect("fixture exists"),
    )
    .expect("fixture parses");

    let request = GenerationRequest {
        prompt: tokenizer.encode(&golden.prompt),
        max_tokens: golden.max_tokens,
        params: SamplingParams {
            temperature: 0.0,
            repeat_penalty: 1.0,
            ..SamplingParams::default()
        },
        stop_tokens: vec![1, 130_073],
    };

    let mut text = String::new();
    let stats = generate(
        &mut session,
        &tokenizer,
        &request,
        &AtomicBool::new(false),
        |event| {
            if let Event::Token { text: chunk, .. } = event {
                text.push_str(&chunk);
            }
        },
    )
    .expect("generation runs");

    println!(
        "{} tokens, {:.2} tok/s, ttft {} ms",
        stats.generated_tokens, stats.tokens_per_second, stats.ttft_ms
    );
    assert_eq!(text.trim_end(), golden.text.trim_end());
}
