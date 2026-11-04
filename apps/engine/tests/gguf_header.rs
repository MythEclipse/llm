//! Model geometry read straight from the GGUF header.
//!
//! Needs the real model file, so it is `#[ignore]`d like every model-dependent
//! test: run with `cargo test -p llm-engine -- --ignored`.

use std::path::PathBuf;

use llm_engine::{DType, GgufFile};

fn model_path() -> PathBuf {
    let default =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/MiniCPM5-2B-Q8_0.gguf");
    std::env::var_os("LLM_MODEL").map_or(default, PathBuf::from)
}

#[test]
#[ignore = "requires models/MiniCPM5-2B-Q8_0.gguf"]
fn header_reports_minicpm5_geometry() {
    let gguf = GgufFile::open(&model_path()).expect("model opens");

    assert_eq!(gguf.meta_str("general.architecture").unwrap(), "llama");
    assert_eq!(gguf.meta_u64("llama.block_count").unwrap(), 42);
    assert_eq!(gguf.meta_u64("llama.embedding_length").unwrap(), 2048);
    assert_eq!(gguf.meta_u64("llama.feed_forward_length").unwrap(), 6144);
    assert_eq!(gguf.meta_u64("llama.attention.head_count").unwrap(), 16);
    assert_eq!(gguf.meta_u64("llama.attention.head_count_kv").unwrap(), 2);
    assert_eq!(gguf.meta_u64("llama.attention.key_length").unwrap(), 128);
    assert_eq!(gguf.meta_u64("llama.rope.dimension_count").unwrap(), 128);
    assert_eq!(gguf.meta_u64("llama.vocab_size").unwrap(), 130560);
    assert_eq!(gguf.meta_f32("llama.rope.freq_base").unwrap(), 5_000_000.0);
    assert_eq!(gguf.meta_str("tokenizer.ggml.model").unwrap(), "gpt2");
    assert_eq!(gguf.meta_str("tokenizer.ggml.pre").unwrap(), "minicpm5");
    assert!(!gguf.meta_bool("tokenizer.ggml.add_bos_token").unwrap());
}

#[test]
#[ignore = "requires models/MiniCPM5-2B-Q8_0.gguf"]
fn tensor_table_matches_expectations() {
    let gguf = GgufFile::open(&model_path()).expect("model opens");

    assert_eq!(gguf.tensors().len(), 381);
    let q8 = gguf
        .tensors()
        .iter()
        .filter(|t| t.dtype == DType::Q8_0)
        .count();
    let f32_count = gguf
        .tensors()
        .iter()
        .filter(|t| t.dtype == DType::F32)
        .count();
    assert_eq!((q8, f32_count), (296, 85));

    let output = gguf.tensor("output.weight").unwrap();
    assert_eq!(output.dims, vec![2048, 130560]);
    assert_eq!(output.dtype, DType::Q8_0);
    assert_eq!(
        gguf.data("output.weight").unwrap().len(),
        2048 * 130560 / 32 * 34
    );

    let norm = gguf.tensor("output_norm.weight").unwrap();
    assert_eq!(norm.dims, vec![2048]);
    assert_eq!(norm.dtype, DType::F32);

    let k = gguf.tensor("blk.41.attn_k.weight").unwrap();
    assert_eq!(k.dims, vec![2048, 256]);
}

#[test]
#[ignore = "requires models/MiniCPM5-2B-Q8_0.gguf"]
fn metadata_string_arrays_decode_lazily() {
    let gguf = GgufFile::open(&model_path()).expect("model opens");

    let tokens = gguf.meta_str_array("tokenizer.ggml.tokens").unwrap();
    assert_eq!(tokens.len(), 130560);
    assert_eq!(tokens[0], "<s>");
    assert_eq!(tokens[1], "</s>");

    let merges = gguf.meta_str_array("tokenizer.ggml.merges").unwrap();
    assert_eq!(merges.len(), 129794);
}
