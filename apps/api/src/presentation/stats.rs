//! Statistics endpoints: overview, recent runs, daily timeseries.

use axum::Json;
use axum::extract::{Query, State};
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::application::ports::ChatEngine;
use crate::application::{ListRunsUseCase, OverviewUseCase, TimeseriesUseCase};
use crate::domain::stats::{DayBucket, Overview};
use crate::presentation::chat::RunResponse;
use crate::presentation::error::AppError;
use crate::presentation::state::AppState;

#[derive(Debug, Deserialize)]
pub struct RunsQuery {
    #[serde(default = "default_limit")]
    pub limit: u64,
}

fn default_limit() -> u64 {
    20
}

#[derive(Debug, Deserialize)]
pub struct TimeseriesQuery {
    #[serde(default = "default_days")]
    pub days: u32,
}

fn default_days() -> u32 {
    30
}

#[derive(Debug, Serialize)]
pub struct OverviewResponse {
    pub total_runs: u64,
    pub total_tokens: u64,
    pub average_tokens_per_second: f64,
    pub p50_tokens_per_second: f64,
    pub p95_tokens_per_second: f64,
    pub average_ttft_ms: f64,
    pub runs_today: u64,
    pub tokens_today: u64,
}

impl From<Overview> for OverviewResponse {
    fn from(overview: Overview) -> Self {
        Self {
            total_runs: overview.total_runs,
            total_tokens: overview.total_tokens,
            average_tokens_per_second: overview.average_tokens_per_second,
            p50_tokens_per_second: overview.p50_tokens_per_second,
            p95_tokens_per_second: overview.p95_tokens_per_second,
            average_ttft_ms: overview.average_ttft_ms,
            runs_today: overview.runs_today,
            tokens_today: overview.tokens_today,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct DayBucketResponse {
    pub date: String,
    pub runs: u64,
    pub tokens: u64,
    pub average_tokens_per_second: f64,
}

impl From<DayBucket> for DayBucketResponse {
    fn from(bucket: DayBucket) -> Self {
        Self {
            date: bucket.date,
            runs: bucket.runs,
            tokens: bucket.tokens,
            average_tokens_per_second: bucket.average_tokens_per_second,
        }
    }
}

#[instrument(skip_all)]
pub async fn overview<E: ChatEngine + 'static>(
    State(state): State<AppState<E>>,
) -> Result<Json<OverviewResponse>, AppError> {
    let overview = OverviewUseCase::new(state.runs.clone()).execute().await?;
    Ok(Json(OverviewResponse::from(overview)))
}

#[instrument(skip_all, fields(limit = query.limit))]
pub async fn runs<E: ChatEngine + 'static>(
    State(state): State<AppState<E>>,
    Query(query): Query<RunsQuery>,
) -> Result<Json<Vec<RunResponse>>, AppError> {
    let runs = ListRunsUseCase::new(state.runs.clone())
        .execute(query.limit)
        .await?;
    Ok(Json(runs.into_iter().map(RunResponse::from).collect()))
}

#[instrument(skip_all, fields(days = query.days))]
pub async fn timeseries<E: ChatEngine + 'static>(
    State(state): State<AppState<E>>,
    Query(query): Query<TimeseriesQuery>,
) -> Result<Json<Vec<DayBucketResponse>>, AppError> {
    let buckets = TimeseriesUseCase::new(state.runs.clone())
        .execute(query.days)
        .await?;
    Ok(Json(
        buckets.into_iter().map(DayBucketResponse::from).collect(),
    ))
}
