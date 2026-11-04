//! `GET /api/model` — what the header of the UI reports.

use axum::Json;
use axum::extract::State;
use serde::Serialize;
use tracing::instrument;

use crate::application::ports::ChatEngine;
use crate::presentation::error::AppError;
use crate::presentation::state::AppState;

#[derive(Debug, Serialize)]
pub struct ModelResponse {
    pub name: String,
    pub architecture: String,
    pub layers: usize,
    pub embedding: usize,
    pub feed_forward: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub context: usize,
    pub vocab: usize,
    pub tensors: usize,
    pub quantized_tensors: usize,
    pub weight_bytes: u64,
    pub kv_bytes: usize,
    pub threads: usize,
    /// `null` on the CPU backend.
    pub device: Option<String>,
}

#[instrument(skip_all)]
pub async fn model_info<E: ChatEngine + 'static>(
    State(state): State<AppState<E>>,
) -> Result<Json<ModelResponse>, AppError> {
    let info = state.engine.info().await?;
    Ok(Json(ModelResponse {
        name: info.name,
        architecture: info.architecture,
        layers: info.layers,
        embedding: info.embedding,
        feed_forward: info.feed_forward,
        heads: info.heads,
        kv_heads: info.kv_heads,
        head_dim: info.head_dim,
        context: info.context,
        vocab: info.vocab,
        tensors: info.tensors,
        quantized_tensors: info.quantized_tensors,
        weight_bytes: info.weight_bytes,
        kv_bytes: info.kv_bytes,
        threads: info.threads,
        device: info.device,
    }))
}
