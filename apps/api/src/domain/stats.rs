//! Per-run telemetry: the numbers the statistics panel shows.

use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::domain::error::RepositoryError;

#[derive(Debug, Clone)]
pub struct RunStats {
    pub id: Uuid,
    pub conversation_id: Uuid,
    pub model: String,
    pub prompt_tokens: u32,
    pub generated_tokens: u32,
    pub prefill_ms: u32,
    pub ttft_ms: u32,
    pub decode_ms: u32,
    pub total_ms: u32,
    pub tokens_per_second: f64,
    pub ctx_used: u32,
    pub stop_reason: String,
    pub sampler: String,
    pub token_latencies_ms: Vec<f32>,
    pub peak_rss_bytes: u64,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct NewRun {
    pub id: Uuid,
    pub conversation_id: Uuid,
    pub model: String,
    pub prompt_tokens: u32,
    pub generated_tokens: u32,
    pub prefill_ms: u32,
    pub ttft_ms: u32,
    pub decode_ms: u32,
    pub total_ms: u32,
    pub tokens_per_second: f64,
    pub ctx_used: u32,
    pub stop_reason: String,
    pub sampler: String,
    pub token_latencies_ms: Vec<f32>,
    pub peak_rss_bytes: u64,
}

#[derive(Debug, Clone, Default)]
pub struct Overview {
    pub total_runs: u64,
    pub total_tokens: u64,
    pub average_tokens_per_second: f64,
    pub p50_tokens_per_second: f64,
    pub p95_tokens_per_second: f64,
    pub average_ttft_ms: f64,
    pub runs_today: u64,
    pub tokens_today: u64,
}

#[derive(Debug, Clone)]
pub struct DayBucket {
    pub date: String,
    pub runs: u64,
    pub tokens: u64,
    pub average_tokens_per_second: f64,
}

pub trait RunRepository: Send + Sync {
    fn insert(&self, run: NewRun)
    -> impl Future<Output = Result<RunStats, RepositoryError>> + Send;

    fn overview(&self) -> impl Future<Output = Result<Overview, RepositoryError>> + Send;

    fn recent(
        &self,
        limit: u64,
    ) -> impl Future<Output = Result<Vec<RunStats>, RepositoryError>> + Send;

    fn daily(
        &self,
        days: u32,
    ) -> impl Future<Output = Result<Vec<DayBucket>, RepositoryError>> + Send;
}
