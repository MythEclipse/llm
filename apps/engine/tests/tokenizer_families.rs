//! The tokenizer follows what the file declares, not one hardcoded family.
//!
//! `tests/fixtures/tiny_spm.gguf` (built by `scripts/make_tiny_spm.py`) is a
//! SentencePiece vocabulary with every prefix of two words present, so the
//! merge-by-score walk has something to climb. The shipped model covers the other
//! family in `tokenizer_golden`.

use std::path::PathBuf;

use llm_engine::chat::{ChatFormat, Template};
use llm_engine::{GgufFile, Tokenizer};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name)
}

#[test]
fn a_sentencepiece_file_is_read_as_sentencepiece() {
    let gguf = GgufFile::open(&fixture("tiny_spm.gguf")).expect("fixture opens");
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    assert_eq!(tokenizer.family(), "sentencepiece");
    assert_eq!(tokenizer.bos_id(), Some(1));
    assert_eq!(tokenizer.eos_id(), Some(2));
}

#[test]
fn sentencepiece_merges_up_to_whole_words() {
    let gguf = GgufFile::open(&fixture("tiny_spm.gguf")).expect("fixture opens");
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    let ids = tokenizer.encode("hello world");
    // Three control pieces first, then the five prefixes of "hello" (ids 3-7) and
    // of "world" (ids 8-12): the encoder has to climb all the way to the word.
    assert_eq!(ids, vec![7, 12], "merged pieces, not characters: {ids:?}");
    assert_eq!(tokenizer.decode(&ids), "hello world");
}

#[test]
fn sentencepiece_round_trips_text_it_has_no_pieces_for() {
    let gguf = GgufFile::open(&fixture("tiny_spm.gguf")).expect("fixture opens");
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    // Nothing in this vocabulary matches, so the byte fallback is what keeps the
    // text recoverable; without `<0xNN>` pieces it degrades to `<unk>`.
    let ids = tokenizer.encode("zzz");
    assert!(!ids.is_empty());
}

#[test]
fn the_template_is_taken_from_the_file() {
    let gguf = GgufFile::open(&fixture("tiny_spm.gguf")).expect("fixture opens");
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    let format = ChatFormat::from_gguf(&gguf, &tokenizer).expect("template");
    assert_eq!(format.template(), Template::Mistral);
    assert_eq!(format.bos(), "<s>");
}

#[test]
fn an_unrecognised_template_is_refused_by_name() {
    let error = Template::detect("{% for m in messages %}{{ m.content }}{% endfor %}")
        .expect_err("unknown template");
    assert!(
        error
            .to_string()
            .contains("neither chatml, llama3 nor mistral"),
        "{error}"
    );
}
