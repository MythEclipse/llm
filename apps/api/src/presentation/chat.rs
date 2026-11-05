//! Streaming chat: `POST /messages` answers with server-sent events, and
//! `POST /stop` flips the cancel flag of the run in flight.

use std::convert::Infallible;
use std::time::Duration;

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use serde::{Deserialize, Serialize};
use tokio_stream::Stream;
use tokio_stream::wrappers::ReceiverStream;
use tracing::instrument;
use uuid::Uuid;
use zod_rs::prelude::{Schema as _, ZodSchema};

use crate::application::ports::{ChatEngine, GenerationEvent, RunOutcome};
use crate::application::{
    FinishChatCommand, FinishChatUseCase, SendMessageCommand, SendMessageUseCase,
};
use crate::domain::stats::RunStats;
use crate::presentation::error::AppError;
use crate::presentation::state::AppState;
use crate::presentation::validation::validate;

const DEFAULT_MAX_TOKENS: usize = 512;
const MAX_MAX_TOKENS: usize = 4096;

#[derive(Debug, Deserialize, Serialize, ZodSchema)]
pub struct SendMessageRequest {
    #[zod(min_length(1), max_length(32000))]
    pub content: String,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub repeat_penalty: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct RunResponse {
    pub id: Uuid,
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
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl From<RunStats> for RunResponse {
    fn from(run: RunStats) -> Self {
        Self {
            id: run.id,
            prompt_tokens: run.prompt_tokens,
            generated_tokens: run.generated_tokens,
            prefill_ms: run.prefill_ms,
            ttft_ms: run.ttft_ms,
            decode_ms: run.decode_ms,
            total_ms: run.total_ms,
            tokens_per_second: run.tokens_per_second,
            ctx_used: run.ctx_used,
            stop_reason: run.stop_reason,
            sampler: run.sampler,
            token_latencies_ms: run.token_latencies_ms,
            peak_rss_bytes: run.peak_rss_bytes,
            created_at: run.created_at,
        }
    }
}

#[instrument(skip_all, fields(conversation_id = %id))]
pub async fn send_message<E: ChatEngine + 'static>(
    State(state): State<AppState<E>>,
    Path(id): Path<Uuid>,
    Json(body): Json<SendMessageRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, AppError> {
    let body = validate(body)?;
    let command = SendMessageCommand {
        conversation_id: id,
        content: body.content,
        max_tokens: body
            .max_tokens
            .unwrap_or(DEFAULT_MAX_TOKENS)
            .clamp(1, MAX_MAX_TOKENS),
        temperature: body.temperature.unwrap_or(0.7).clamp(0.0, 2.0),
        // Truncation defaults are narrower than llama.cpp's (0.95 / 0.05)
        // because this model leaves Indonesian when the candidate pool is wide:
        // at 0.95 / 0.05 it answers in Javanese and refuses facts it was just
        // given, at 0.75 / 0.25 every answer came back clean Indonesian.
        top_p: body.top_p.unwrap_or(0.75).clamp(0.0, 1.0),
        top_k: body.top_k.unwrap_or(40).min(200),
        min_p: body.min_p.unwrap_or(0.25).clamp(0.0, 1.0),
        repeat_penalty: body.repeat_penalty.unwrap_or(1.1).clamp(0.5, 2.0),
        seed: body.seed.unwrap_or(0x5EED_1234_ABCD_0001),
    };

    let stream = SendMessageUseCase::new(
        state.conversations.clone(),
        state.engine.clone(),
        state.model_name.clone(),
    )
    .execute(command)
    .await?;

    state.register_cancel(id, stream.handle.cancel.clone());

    let (sender, receiver) = tokio::sync::mpsc::channel(64);
    let conversations = state.conversations.clone();
    let runs = state.runs.clone();
    let cancels = state.clone();
    let model = stream.model.clone();
    let sampler = stream.sampler.clone();
    let mut events = stream.handle.events;
    let cancel_flag = stream.handle.cancel.clone();

    tokio::spawn(async move {
        let mut text = String::new();
        let mut outcome: Option<RunOutcome> = None;
        let mut delivered = true;

        while let Some(event) = events.recv().await {
            let sse = match event {
                GenerationEvent::Start { prompt_tokens } => Event::default()
                    .event("start")
                    .json_data(serde_json::json!({ "prompt_tokens": prompt_tokens }))
                    .ok(),
                GenerationEvent::Token { text: chunk } => {
                    text.push_str(&chunk);
                    Event::default()
                        .event("token")
                        .json_data(serde_json::json!({ "text": chunk }))
                        .ok()
                }
                GenerationEvent::Finished(run_outcome) => {
                    outcome = Some(run_outcome);
                    None
                }
                GenerationEvent::Failed { message } => Event::default()
                    .event("error")
                    .json_data(serde_json::json!({ "message": message }))
                    .ok(),
            };
            if let Some(sse) = sse
                && delivered
                && sender.send(Ok(sse)).await.is_err()
            {
                // The browser went away: stop paying for tokens nobody reads, but
                // keep draining so the engine can finish and the run is still
                // recorded with stop_reason = cancelled.
                cancel_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                delivered = false;
            }
        }

        if let Some(outcome) = outcome {
            let finish = FinishChatUseCase::new(runs, conversations)
                .execute(FinishChatCommand {
                    conversation_id: id,
                    outcome,
                    assistant_text: text,
                    model,
                    sampler,
                })
                .await;
            if delivered {
                let sse = match finish {
                    Ok(run) => Event::default()
                        .event("stats")
                        .json_data(RunResponse::from(run))
                        .ok(),
                    Err(error) => Event::default()
                        .event("error")
                        .json_data(serde_json::json!({ "message": error.to_string() }))
                        .ok(),
                };
                if let Some(sse) = sse {
                    let _ = sender.send(Ok(sse)).await;
                }
            }
        }
        cancels.clear_cancel(id);
    });

    Ok(Sse::new(ReceiverStream::new(receiver))
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(15))))
}

#[instrument(skip_all, fields(conversation_id = %id))]
pub async fn stop<E: ChatEngine + 'static>(
    State(state): State<AppState<E>>,
    Path(id): Path<Uuid>,
) -> Result<StatusCode, AppError> {
    if state.cancel(id) {
        tracing::warn!(conversation_id = %id, "cancelling generation on request");
    }
    Ok(StatusCode::ACCEPTED)
}

#[derive(Debug, Serialize)]
pub struct StopResponse {
    pub cancelled: bool,
}

#[instrument(skip_all, fields(conversation_id = %id))]
pub async fn stop_status<E: ChatEngine + 'static>(
    State(state): State<AppState<E>>,
    Path(id): Path<Uuid>,
) -> Json<StopResponse> {
    Json(StopResponse {
        cancelled: state.cancel(id),
    })
}
