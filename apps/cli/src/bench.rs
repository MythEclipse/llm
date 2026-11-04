//! Benchmark harness that mirrors `llama-bench`, so the two can be compared line
//! by line: the same `pp`/`tg` test names, the same repeat-and-report-the-spread
//! shape, and the same `-p`/`-n`/`-r` flags.
//!
//! ```text
//! llm-cli bench -p 128,512,1024 -n 128 -r 5
//! llm-cli bench -p 512 -o json
//! ```
//!
//! `pp` measures the prefill of N tokens and nothing else. `tg` prefills a short
//! prompt and then measures only the decode phase, so a slow prefill cannot hide
//! inside the generation number.

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use llm_engine::{
    GenerationRequest, GgufFile, Model, SamplingParams, Session, Tokenizer, generate,
};

/// Prompt used to reach the requested `pp` length; content is irrelevant to cost.
const FILLER: &str = "the quick brown fox jumps over the lazy dog ";

/// Tokens prefilled before a `tg` run, so the decode loop starts from a real cache.
const TG_PROMPT: usize = 16;

pub struct BenchOptions {
    pub model_path: PathBuf,
    pub context: usize,
    pub prompts: Vec<usize>,
    pub generations: Vec<usize>,
    pub repeats: usize,
    pub json: bool,
}

pub struct Measurement {
    pub test: String,
    pub tokens_per_second: f64,
    pub spread: f64,
}

/// Parses `bench` arguments, mirroring llama-bench's flag names.
pub fn parse(arguments: &mut impl Iterator<Item = String>) -> Result<BenchOptions, String> {
    let mut options = BenchOptions {
        model_path: PathBuf::from("models/MiniCPM5-2B-Q8_0.gguf"),
        context: 4096,
        prompts: vec![512],
        generations: vec![128],
        repeats: 5,
        json: false,
    };
    while let Some(argument) = arguments.next() {
        let mut value = || {
            arguments
                .next()
                .ok_or_else(|| format!("{argument} needs a value"))
        };
        match argument.as_str() {
            "-m" | "--model" => options.model_path = PathBuf::from(value()?),
            "-c" | "--context" => options.context = value()?.parse().map_err(|_| "bad -c")?,
            "-p" | "--prompt" => options.prompts = parse_list(&value()?)?,
            "-n" | "--gen" => options.generations = parse_list(&value()?)?,
            "-r" | "--reps" => options.repeats = value()?.parse().map_err(|_| "bad -r")?,
            "-o" | "--output" => options.json = value()?.eq_ignore_ascii_case("json"),
            other => return Err(format!("unknown bench flag {other}")),
        }
    }
    Ok(options)
}

fn parse_list(text: &str) -> Result<Vec<usize>, String> {
    text.split(',')
        .map(|part| {
            part.trim()
                .parse::<usize>()
                .map_err(|_| format!("bad length {part}"))
        })
        .collect()
}

pub fn run(options: &BenchOptions) -> Result<Vec<Measurement>, Box<dyn std::error::Error>> {
    let gguf = GgufFile::open(&options.model_path)?;
    let model = Model::load(&gguf)?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let filler: Vec<u32> = tokenizer.encode(FILLER);
    let device = model.device_name().unwrap_or("cpu").to_string();

    let mut measurements = Vec::new();
    for &prompt in &options.prompts {
        let tokens = repeat_to(&filler, prompt);
        let mut samples = Vec::with_capacity(options.repeats);
        for _ in 0..options.repeats {
            let mut session = Session::new(&model, options.context)?;
            let started = Instant::now();
            session.prefill(&tokens)?;
            samples.push(prompt as f64 / started.elapsed().as_secs_f64());
        }
        measurements.push(Measurement {
            test: format!("pp{prompt}"),
            tokens_per_second: mean(&samples),
            spread: deviation(&samples),
        });
    }

    for &generation in options.generations.iter().filter(|&&count| count > 0) {
        let tokens = repeat_to(&filler, TG_PROMPT);
        let mut samples = Vec::with_capacity(options.repeats);
        for _ in 0..options.repeats {
            let mut session = Session::new(&model, options.context)?;
            session.prefill(&tokens)?;
            let request = GenerationRequest {
                prompt: tokens.clone(),
                max_tokens: generation,
                params: SamplingParams {
                    temperature: 0.0,
                    ..SamplingParams::default()
                },
                stop_tokens: Vec::new(),
            };
            let cancel = AtomicBool::new(false);
            let stats = generate(&mut session, &tokenizer, &request, &cancel, |_| {})?;
            samples.push(generation as f64 / (stats.decode_ms as f64 / 1000.0));
        }
        measurements.push(Measurement {
            test: format!("tg{generation}"),
            tokens_per_second: mean(&samples),
            spread: deviation(&samples),
        });
    }

    report(
        &measurements,
        &device,
        &options.model_path,
        model.compute().device_footprint(),
    );
    Ok(measurements)
}

/// Repeats the filler until it holds exactly `length` tokens.
fn repeat_to(filler: &[u32], length: usize) -> Vec<u32> {
    let mut tokens = Vec::with_capacity(length);
    while tokens.len() < length {
        tokens.extend_from_slice(filler);
    }
    tokens.truncate(length);
    tokens
}

fn mean(samples: &[f64]) -> f64 {
    samples.iter().sum::<f64>() / samples.len().max(1) as f64
}

/// Sample standard deviation, the same spread llama-bench prints after the ±.
fn deviation(samples: &[f64]) -> f64 {
    if samples.len() < 2 {
        return 0.0;
    }
    let average = mean(samples);
    let variance = samples
        .iter()
        .map(|sample| (sample - average).powi(2))
        .sum::<f64>()
        / (samples.len() - 1) as f64;
    variance.sqrt()
}

fn report(
    measurements: &[Measurement],
    device: &str,
    model_path: &std::path::Path,
    device_bytes: Option<(usize, usize)>,
) {
    let name = model_path.file_stem().map_or_else(
        || "model".to_string(),
        |stem| stem.to_string_lossy().to_string(),
    );
    println!("| model | backend | test | t/s |");
    println!("|---|---|---|---|");
    for measurement in measurements {
        println!(
            "| {name} | {device} | {} | {:.2} ± {:.2} |",
            measurement.test, measurement.tokens_per_second, measurement.spread
        );
    }
    if let Some((live, peak)) = device_bytes {
        println!("| | | vram | {} live, {} peak |", gib(live), gib(peak));
    }
}

/// Bytes as GiB with two decimals, the unit llama.cpp prints its buffers in.
fn gib(bytes: usize) -> String {
    format!("{:.2} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
}

/// JSON form, matching llama-bench's `-o json` shape closely enough to be read by
/// the same scripts: one object per test with the mean and the spread.
pub fn to_json(measurements: &[Measurement], device: &str) -> String {
    let rows: Vec<String> = measurements
        .iter()
        .map(|measurement| {
            format!(
                "  {{\"test\": \"{}\", \"backend\": \"{}\", \"avg_ts\": {:.4}, \"stddev_ts\": {:.4}}}",
                measurement.test, device, measurement.tokens_per_second, measurement.spread
            )
        })
        .collect();
    format!("[\n{}\n]", rows.join(",\n"))
}
