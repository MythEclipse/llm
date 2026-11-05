//! Ports the application owns: the engine contract and the shapes it exchanges.
//!
//! The engine is reached through a channel handle rather than a callback: a port
//! that took a closure returning `impl Future` cannot be made `Send` for every
//! caller lifetime, so the contract hands back owned data instead.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use tokio::sync::mpsc;

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("engine is busy with another generation")]
    Busy,
    #[error("context window is full")]
    ContextFull,
    #[error("engine failure: {0}")]
    Internal(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnRole {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone)]
pub struct ChatTurn {
    pub role: TurnRole,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub turns: Vec<ChatTurn>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub seed: u64,
}

impl GenerateRequest {
    /// Human-readable sampler settings, stored with the run for later comparison.
    pub fn sampler_summary(&self) -> String {
        format!(
            "temp={} top_p={} top_k={} min_p={} repeat_penalty={} seed={}",
            self.temperature, self.top_p, self.top_k, self.min_p, self.repeat_penalty, self.seed
        )
    }
}

#[derive(Debug, Clone, Default)]
pub struct RunOutcome {
    /// Identity of the kernels that produced these numbers, so a stored run can
    /// be traced back to one build.
    pub build_hash: String,
    pub prompt_tokens: u32,
    pub generated_tokens: u32,
    pub prefill_ms: u32,
    pub ttft_ms: u32,
    pub decode_ms: u32,
    pub total_ms: u32,
    pub tokens_per_second: f64,
    pub ctx_used: u32,
    pub stop_reason: String,
    pub token_latencies_ms: Vec<f32>,
    pub peak_rss_bytes: u64,
}

#[derive(Debug, Clone)]
pub enum GenerationEvent {
    Start {
        prompt_tokens: u32,
        /// History turns left out so the prompt and the reply fit the window.
        dropped_turns: u32,
    },
    Token {
        text: String,
    },
    Finished(RunOutcome),
    Failed {
        message: String,
    },
}

pub struct GenerationHandle {
    pub events: mpsc::Receiver<GenerationEvent>,
    pub cancel: Arc<AtomicBool>,
}

/// What the binary was compiled from.
#[derive(Debug, Clone)]
pub struct BuildInfo {
    pub arch: String,
    pub source_sha256: String,
    pub ptx_sha256: Option<String>,
    pub nvcc: String,
    pub kernels: usize,
    /// Hash over the fields above: the artifact identity.
    pub identity: String,
}

/// Which kernels this model resolved to at load, and whether any of them is the
/// general path.
#[derive(Debug, Clone)]
pub struct PlanInfo {
    pub decode: String,
    pub prefill: String,
    pub identity: String,
    /// `regime: reason` for every regime on the general path.
    pub fallbacks: Vec<String>,
    /// Scratch the largest prefill chunk plans, and how much of it is shared
    /// between buffers whose lifetimes do not overlap.
    pub scratch_bytes: usize,
    pub scratch_reused_bytes: usize,
    pub scratch_identity: String,
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
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
    /// Device running the matmuls; `None` means the CPU.
    pub device: Option<String>,
    pub build: BuildInfo,
    pub plan: PlanInfo,
}

/// `Arc<T>` is a chat engine whenever `T` is, so handlers can hold one shared
/// instance in `AppState` without a wrapper type.
impl<T: ChatEngine + ?Sized> ChatEngine for Arc<T> {
    fn info(&self) -> impl Future<Output = Result<ModelInfo, EngineError>> + Send {
        (**self).info()
    }

    fn start(
        &self,
        request: GenerateRequest,
    ) -> impl Future<Output = Result<GenerationHandle, EngineError>> + Send {
        (**self).start(request)
    }
}

pub trait ChatEngine: Send + Sync {
    fn info(&self) -> impl Future<Output = Result<ModelInfo, EngineError>> + Send;

    fn start(
        &self,
        request: GenerateRequest,
    ) -> impl Future<Output = Result<GenerationHandle, EngineError>> + Send;
}
