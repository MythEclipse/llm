//! Token sampling: repeat penalty → temperature → top-k → top-p → min-p.
//!
//! The order is llama.cpp's default sampler chain, so a `temperature = 0` run
//! reproduces its greedy output exactly and a warm run is comparable.

use crate::error::EngineError;

#[derive(Debug, Clone, Copy)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub seed: u64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            // Narrower than llama.cpp's 0.95 / 0.05: with the wider pool this
            // model leaves Indonesian entirely. Measured on the shipped model:
            // 0.95 / 0.05 answered in Javanese and denied facts it was given,
            // 0.8 / 0.15 was correct but kept a Javanese tail, 0.75 / 0.25 came
            // back clean on every question.
            top_p: 0.75,
            min_p: 0.25,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            seed: 0x5EED_1234_ABCD_0001,
        }
    }
}

pub struct Sampler {
    state: u64,
}

impl Sampler {
    pub fn new(seed: u64) -> Self {
        Self {
            // xorshift needs a non-zero state.
            state: seed | 1,
        }
    }

    pub fn reseed(&mut self, seed: u64) {
        self.state = seed | 1;
    }

    pub fn sample(
        &mut self,
        logits: &mut [f32],
        recent: &[u32],
        params: &SamplingParams,
    ) -> Result<u32, EngineError> {
        if logits.is_empty() {
            return Err(EngineError::Format("empty logits".into()));
        }
        apply_repeat_penalty(logits, recent, params);
        if params.temperature <= 0.0 {
            return Ok(argmax(logits));
        }
        let inverse_temperature = 1.0 / params.temperature;
        for value in logits.iter_mut() {
            *value *= inverse_temperature;
        }

        let mut ranked: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        if params.top_k > 0 && params.top_k < ranked.len() {
            // Only the top-k can survive, so partition instead of sorting all
            // 130 560 candidates.
            ranked.select_nth_unstable_by(params.top_k - 1, |a, b| b.1.total_cmp(&a.1));
            ranked.truncate(params.top_k);
        }
        ranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        softmax_ranked(&mut ranked);
        apply_top_p(&mut ranked, params.top_p);
        apply_min_p(&mut ranked, params.min_p);

        let total: f32 = ranked.iter().map(|(_, probability)| probability).sum();
        if total <= 0.0 {
            return Ok(argmax(logits));
        }
        let mut target = self.next_f32() * total;
        for (id, probability) in &ranked {
            target -= probability;
            if target <= 0.0 {
                return Ok(*id as u32);
            }
        }
        Ok(ranked[0].0 as u32)
    }

    fn next_f32(&mut self) -> f32 {
        // xorshift64*
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        let value = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((value >> 40) as f32) / ((1u32 << 24) as f32)
    }
}

fn apply_repeat_penalty(logits: &mut [f32], recent: &[u32], params: &SamplingParams) {
    if params.repeat_penalty == 1.0 || params.repeat_last_n == 0 {
        return;
    }
    let window = recent.len().saturating_sub(params.repeat_last_n);
    for id in &recent[window..] {
        let index = *id as usize;
        if index >= logits.len() {
            continue;
        }
        let value = logits[index];
        logits[index] = if value > 0.0 {
            value / params.repeat_penalty
        } else {
            value * params.repeat_penalty
        };
    }
}

fn softmax_ranked(ranked: &mut [(usize, f32)]) {
    let max = ranked[0].1;
    let mut sum = 0.0f32;
    for (_, value) in ranked.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    if sum > 0.0 {
        for (_, value) in ranked.iter_mut() {
            *value /= sum;
        }
    }
}

fn apply_top_p(ranked: &mut Vec<(usize, f32)>, top_p: f32) {
    if !(0.0..1.0).contains(&top_p) {
        return;
    }
    let mut cumulative = 0.0f32;
    let mut keep = ranked.len();
    for (index, (_, probability)) in ranked.iter().enumerate() {
        cumulative += probability;
        if cumulative > top_p {
            keep = index + 1;
            break;
        }
    }
    ranked.truncate(keep);
}

fn apply_min_p(ranked: &mut Vec<(usize, f32)>, min_p: f32) {
    if min_p <= 0.0 {
        return;
    }
    let threshold = ranked[0].1 * min_p;
    ranked.retain(|(_, probability)| *probability >= threshold);
}

fn argmax(values: &[f32]) -> u32 {
    let mut best = 0usize;
    for (index, value) in values.iter().enumerate() {
        if *value > values[best] {
            best = index;
        }
    }
    best as u32
}
