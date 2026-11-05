//! `GET /api/model` — what the header of the UI reports.

use axum::Json;
use axum::extract::State;
use serde::Serialize;
use tracing::instrument;

use crate::application::ports::{BuildInfo, ChatEngine, PlanInfo};
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
    pub build: BuildInfoResponse,
    pub plan: PlanInfoResponse,
}

/// The artifact these numbers came from: source hash, PTX hash, compiler, arch.
#[derive(Debug, Serialize)]
pub struct BuildInfoResponse {
    pub arch: String,
    pub source_sha256: String,
    pub ptx_sha256: Option<String>,
    pub nvcc: String,
    pub kernels: usize,
    pub identity: String,
}

#[derive(Debug, Serialize)]
pub struct PlanInfoResponse {
    pub decode: String,
    pub prefill: String,
    pub identity: String,
    pub fallbacks: Vec<String>,
}

impl From<BuildInfo> for BuildInfoResponse {
    fn from(info: BuildInfo) -> Self {
        Self {
            arch: info.arch,
            source_sha256: info.source_sha256,
            ptx_sha256: info.ptx_sha256,
            nvcc: info.nvcc,
            kernels: info.kernels,
            identity: info.identity,
        }
    }
}

impl From<PlanInfo> for PlanInfoResponse {
    fn from(info: PlanInfo) -> Self {
        Self {
            decode: info.decode,
            prefill: info.prefill,
            identity: info.identity,
            fallbacks: info.fallbacks,
        }
    }
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
        build: info.build.into(),
        plan: info.plan.into(),
    }))
}
