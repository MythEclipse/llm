//! `llm-cli inspect <file.gguf>` — what a file is, and whether it can run here.
//!
//! This is the loader answering for any model, including the ones whose blocks
//! this engine does not implement: it reads the header, the tokenizer and the
//! chat template, then says plainly whether the file can run and why not. The
//! verdict comes from actually attempting the load, so it cannot drift from what
//! the engine really does.

use std::collections::BTreeMap;
use std::error::Error;
use std::path::Path;

use llm_engine::chat::{ChatFormat, Template};
use llm_engine::{GgufFile, Model, Tokenizer};

pub fn run(path: &Path) -> Result<(), Box<dyn Error>> {
    let gguf = GgufFile::open(path)?;
    let bytes = std::fs::metadata(path)?.len();
    let architecture = gguf
        .meta_str("general.architecture")
        .unwrap_or("(undeclared)");

    println!("file          {}", path.display());
    println!("size          {} ({} bytes)", size(bytes), bytes);
    println!("architecture  {architecture}");
    if let Ok(name) = gguf.meta_str("general.name") {
        println!("name          {name}");
    }
    if let Ok(label) = gguf.meta_str("general.size_label") {
        println!("params        {label}");
    }

    println!("\ngeometry");
    for (label, key) in [
        ("layers", "block_count"),
        ("embedding", "embedding_length"),
        ("feed-forward", "feed_forward_length"),
        ("heads", "attention.head_count"),
        ("kv heads", "attention.head_count_kv"),
        ("head dim", "attention.key_length"),
        ("vocab", "vocab_size"),
        ("experts", "expert_count"),
        ("experts/token", "expert_used_count"),
        ("context", "context_length"),
        ("rope base", "rope.freq_base"),
    ] {
        let value = gguf
            .meta_u64_opt(&format!("{architecture}.{key}"))
            .or_else(|| gguf.meta_u64_opt(&format!("llama.{key}")))
            .or_else(|| {
                gguf.meta_f32_opt(&format!("{architecture}.{key}"))
                    .map(|f| f as u64)
            });
        if let Some(value) = value {
            println!("  {label:<14} {value}");
        }
    }

    println!("\ntensors");
    let mut by_type: BTreeMap<String, (usize, u64)> = BTreeMap::new();
    for tensor in gguf.tensors() {
        let entry = by_type
            .entry(format!("{:?}", tensor.dtype))
            .or_insert((0, 0));
        entry.0 += 1;
        entry.1 += tensor.dtype.byte_len(tensor.elements());
    }
    for (dtype, (count, bytes)) in &by_type {
        println!("  {dtype:<10} {count:>4} tensors, {}", size(*bytes));
    }

    println!("\ntokenizer");
    let tokenizer = match Tokenizer::from_gguf(&gguf) {
        Ok(tokenizer) => {
            println!("  family        {}", tokenizer.family());
            println!("  vocab         {}", tokenizer.vocab_size());
            println!("  bos id        {:?}", tokenizer.bos_id());
            println!("  eos id        {:?}", tokenizer.eos_id());
            Some(tokenizer)
        }
        Err(error) => {
            println!("  unreadable    {error}");
            if let Ok(tokens) = gguf.meta_str_array("tokenizer.ggml.tokens") {
                println!("  vocab         {} (from the token array)", tokens.len());
            }
            if let Some(id) = gguf.meta_u64_opt("tokenizer.ggml.bos_token_id") {
                println!("  bos id        {id}");
            }
            if let Some(id) = gguf.meta_u64_opt("tokenizer.ggml.eos_token_id") {
                println!("  eos id        {id}");
            }
            None
        }
    };
    if let Ok(pre) = gguf.meta_str("tokenizer.ggml.pre") {
        println!("  pre-tokenizer {pre}");
    }

    println!("\nchat template");
    match &tokenizer {
        Some(tokenizer) => match ChatFormat::from_gguf(&gguf, tokenizer) {
            Ok(format) => {
                println!("  shape         {}", format.template().name());
                println!("  bos marker    {:?}", format.bos());
            }
            Err(error) => println!("  unsupported   {error}"),
        },
        None => match gguf.meta_str("tokenizer.chat_template") {
            Ok(declared) => match Template::detect(declared) {
                Ok(template) => println!("  shape         {}", template.name()),
                Err(error) => println!("  unsupported   {error}"),
            },
            Err(_) => println!("  absent        the file declares no chat template"),
        },
    }

    println!("\nverdict");
    match Model::load_with_device(&gguf, false) {
        Ok(model) => {
            let description = model.describe();
            println!("  runnable      yes, on the cpu path");
            println!("  tensors used  {}", description.tensors);
            println!("  weight bytes  {}", size(description.weight_bytes));
        }
        Err(error) => {
            println!("  runnable      no");
            println!("  reason        {error}");
        }
    }
    Ok(())
}

fn size(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}
