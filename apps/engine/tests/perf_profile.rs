//! Position-dependent cost profile: how much of a decode step is attention.
//!
//! `cargo test --release -p llm-engine --test perf_profile -- --ignored --nocapture`

use std::path::PathBuf;
use std::time::Instant;

use llm_engine::{GgufFile, Model, Session};

fn model_path() -> PathBuf {
    std::env::var_os("LLM_MODEL").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/MiniCPM5-2B-Q8_0.gguf"),
        PathBuf::from,
    )
}

fn ms(duration: std::time::Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

#[test]
#[ignore = "requires models/MiniCPM5-2B-Q8_0.gguf"]
fn decode_step_cost_grows_with_position() {
    let gguf = GgufFile::open(&model_path()).expect("model opens");
    let model = Model::load(&gguf).expect("model loads");
    let filler: Vec<u32> = (0..4096).map(|index| 100 + index % 900).collect();

    for position in [0usize, 128, 512, 1024, 2048, 4090] {
        let mut session = Session::new(&model, (position + 16).next_power_of_two().max(512))
            .expect("session builds");
        let prefill_started = Instant::now();
        let mut logits = session
            .prefill(&filler[..position.max(1)])
            .expect("prefill");
        let prefill_ms = ms(prefill_started.elapsed());

        // The machine is shared, so report the median step instead of the mean:
        // one descheduled step otherwise dominates the average.
        let steps = 9;
        let mut timings = Vec::with_capacity(steps);
        for _ in 0..steps {
            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(index, _)| index as u32)
                .expect("logits");
            let started = Instant::now();
            logits = session.step(next).expect("decode");
            timings.push(ms(started.elapsed()));
        }
        timings.sort_by(f64::total_cmp);
        let per_step = timings[steps / 2];
        println!(
            "position {position:4}: prefill {prefill_ms:7.1} ms, decode {per_step:7.1} ms/token, {:.1} tok/s",
            1000.0 / per_step
        );
    }
}
