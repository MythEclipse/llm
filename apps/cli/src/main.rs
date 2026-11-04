//! Terminal front end for the engine: one-shot completion or a chat REPL.
//!
//! ```text
//! llm-cli "The capital of France is"      # raw completion, prints run stats
//! llm-cli                                 # chat REPL with the ChatML template
//! ```

use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

use llm_engine::chat::{Role, Turn, render};
use llm_engine::{
    Event, GenerationRequest, GgufFile, Model, SamplingParams, Session, Tokenizer, generate,
};

const DEFAULT_MODEL: &str = "models/MiniCPM5-2B-Q8_0.gguf";
const DEFAULT_CONTEXT: usize = 4096;
const DEFAULT_MAX_TOKENS: usize = 256;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model_path = PathBuf::from(DEFAULT_MODEL);
    let mut context = DEFAULT_CONTEXT;
    let mut max_tokens = DEFAULT_MAX_TOKENS;
    let mut temperature = 0.7f32;
    let mut prompt_parts = Vec::new();

    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--model" => model_path = PathBuf::from(args.next().ok_or("--model needs a path")?),
            "--context" => context = args.next().ok_or("--context needs a number")?.parse()?,
            "--max-tokens" => {
                max_tokens = args.next().ok_or("--max-tokens needs a number")?.parse()?
            }
            "--temperature" => {
                temperature = args.next().ok_or("--temperature needs a number")?.parse()?
            }
            other => prompt_parts.push(other.to_string()),
        }
    }

    let gguf = GgufFile::open(&model_path)?;
    let model = Model::load(&gguf)?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let params = SamplingParams {
        temperature,
        ..SamplingParams::default()
    };
    let stop_tokens: Vec<u32> = ["<|im_end|>", "</s>"]
        .iter()
        .filter_map(|token| tokenizer.id_of(token))
        .collect();

    if prompt_parts.is_empty() {
        chat_loop(
            &model,
            &tokenizer,
            context,
            max_tokens,
            params,
            &stop_tokens,
        )
    } else {
        let prompt = tokenizer.encode(&prompt_parts.join(" "));
        complete(
            &model,
            &tokenizer,
            prompt,
            context,
            max_tokens,
            params,
            &stop_tokens,
        )
    }
}

fn complete(
    model: &Model<'_>,
    tokenizer: &Tokenizer,
    prompt: Vec<u32>,
    context: usize,
    max_tokens: usize,
    params: SamplingParams,
    stop_tokens: &[u32],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut session = Session::new(model, context)?;
    run(
        &mut session,
        tokenizer,
        prompt,
        max_tokens,
        params,
        stop_tokens,
    )?;
    Ok(())
}

fn chat_loop(
    model: &Model<'_>,
    tokenizer: &Tokenizer,
    context: usize,
    max_tokens: usize,
    params: SamplingParams,
    stop_tokens: &[u32],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut history: Vec<(Role, String)> = Vec::new();
    println!("model siap. kosongkan baris untuk keluar.");

    loop {
        print!("\nkamu> ");
        std::io::stdout().flush()?;
        let mut line = String::new();
        if std::io::stdin().read_line(&mut line)? == 0 || line.trim().is_empty() {
            break;
        }
        history.push((Role::User, line.trim().to_string()));

        let turns: Vec<Turn<'_>> = history
            .iter()
            .map(|(role, content)| Turn {
                role: *role,
                content,
            })
            .collect();
        let prompt = tokenizer.encode(&render(&turns, false));

        let mut session = Session::new(model, context)?;
        let answer = run(
            &mut session,
            tokenizer,
            prompt,
            max_tokens,
            params,
            stop_tokens,
        )?;
        history.push((Role::Assistant, answer));
    }
    Ok(())
}

fn run(
    session: &mut Session<'_>,
    tokenizer: &Tokenizer,
    prompt: Vec<u32>,
    max_tokens: usize,
    params: SamplingParams,
    stop_tokens: &[u32],
) -> Result<String, Box<dyn std::error::Error>> {
    let request = GenerationRequest {
        prompt,
        max_tokens,
        params,
        stop_tokens: stop_tokens.to_vec(),
    };
    let mut text = String::new();
    let stats = generate(
        session,
        tokenizer,
        &request,
        &AtomicBool::new(false),
        |event| {
            if let Event::Token { text: chunk, .. } = event {
                print!("{chunk}");
                std::io::stdout().flush().ok();
                text.push_str(&chunk);
            }
        },
    )?;
    println!(
        "\n[{} token, {:.2} tok/s, prefill {} ms, ttft {} ms, stop {}]",
        stats.generated_tokens,
        stats.tokens_per_second,
        stats.prefill_ms,
        stats.ttft_ms,
        stats.stop_reason.as_str()
    );
    Ok(text)
}
