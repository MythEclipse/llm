//! Engine adapter: the only place that knows both the port and llm-engine.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use llm_engine::chat::{Role as EngineRole, Turn, render};
use llm_engine::{
    Event, GenerationRequest as CoreRequest, GgufFile, Model, SamplingParams, Session, Tokenizer,
    generate,
};
use tokio::sync::{Semaphore, mpsc};

use crate::application::ports::{
    ChatEngine, ChatTurn, EngineError, GenerateRequest, GenerationEvent, GenerationHandle,
    ModelInfo, RunOutcome, TurnRole,
};

const STOP_TOKENS: [&str; 2] = ["<|im_end|>", "</s>"];

/// Owns the loaded model and runs one generation at a time.
pub struct LocalChatEngine {
    model: &'static Model<'static>,
    tokenizer: Arc<Tokenizer>,
    stop_tokens: Vec<u32>,
    info: ModelInfo,
    slots: Arc<Semaphore>,
    context: usize,
}

impl LocalChatEngine {
    pub fn load(path: &Path, context: usize, threads: usize) -> Result<Self, EngineError> {
        configure_thread_pool(threads);
        let gguf = GgufFile::open(path).map_err(internal)?;
        let gguf: &'static GgufFile = Box::leak(Box::new(gguf));
        let model = Model::load(gguf).map_err(internal)?;
        let model: &'static Model<'static> = Box::leak(Box::new(model));
        let tokenizer = Tokenizer::from_gguf(gguf).map_err(internal)?;
        let stop_tokens = STOP_TOKENS
            .iter()
            .filter_map(|token| tokenizer.id_of(token))
            .collect();

        let description = model.describe();
        let config = description.config;
        let kv_bytes = config.n_layers * config.kv_dim() * 2 * 2 * context;
        let info = ModelInfo {
            name: path
                .file_stem()
                .map(|stem| stem.to_string_lossy().into_owned())
                .unwrap_or_else(|| "model".to_string()),
            architecture: "llama".to_string(),
            layers: config.n_layers,
            embedding: config.n_embd,
            feed_forward: config.n_ff,
            heads: config.n_head,
            kv_heads: config.n_head_kv,
            head_dim: config.head_dim,
            context,
            vocab: config.n_vocab,
            tensors: description.tensors,
            quantized_tensors: description.quantized_tensors,
            weight_bytes: description.weight_bytes,
            kv_bytes,
            threads,
            device: model.device_name().map(str::to_owned),
        };
        if let Some(note) = model.device_note() {
            tracing::warn!(%note, "no GPU backend, running on the cpu");
        }
        tracing::info!(
            model = %info.name,
            layers = info.layers,
            context = info.context,
            weight_bytes = info.weight_bytes,
            device = info.device.as_deref().unwrap_or("cpu"),
            "engine ready"
        );
        Ok(Self {
            model,
            tokenizer: Arc::new(tokenizer),
            stop_tokens,
            info,
            slots: Arc::new(Semaphore::new(1)),
            context,
        })
    }

    fn prompt(&self, turns: &[ChatTurn]) -> String {
        let turns: Vec<Turn<'_>> = turns
            .iter()
            .map(|turn| Turn {
                role: match turn.role {
                    TurnRole::System => EngineRole::System,
                    TurnRole::User => EngineRole::User,
                    TurnRole::Assistant => EngineRole::Assistant,
                },
                content: &turn.content,
            })
            .collect();
        render(&turns, false)
    }
}

impl ChatEngine for LocalChatEngine {
    fn info(&self) -> impl Future<Output = Result<ModelInfo, EngineError>> + Send {
        let info = self.info.clone();
        async move { Ok(info) }
    }

    fn start(
        &self,
        request: GenerateRequest,
    ) -> impl Future<Output = Result<GenerationHandle, EngineError>> + Send {
        let prompt = self.prompt(&request.turns);
        let tokens = self.tokenizer.encode(&prompt);
        let context = self.context;
        let model = self.model;
        let tokenizer = Arc::clone(&self.tokenizer);
        let stop_tokens = self.stop_tokens.clone();
        let slots = Arc::clone(&self.slots);
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = Arc::clone(&cancel);

        async move {
            if tokens.len() + request.max_tokens > context {
                return Err(EngineError::ContextFull);
            }
            let permit = slots.try_acquire_owned().map_err(|_| EngineError::Busy)?;
            let core_request = CoreRequest {
                prompt: tokens,
                max_tokens: request.max_tokens,
                params: SamplingParams {
                    temperature: request.temperature,
                    top_k: request.top_k,
                    top_p: request.top_p,
                    min_p: request.min_p,
                    repeat_penalty: request.repeat_penalty,
                    seed: request.seed,
                    ..SamplingParams::default()
                },
                stop_tokens,
            };

            let (sender, events) = mpsc::channel(32);
            tokio::task::spawn_blocking(move || {
                let mut session = match Session::new(model, context) {
                    Ok(session) => session,
                    Err(error) => {
                        let _ = sender.blocking_send(GenerationEvent::Failed {
                            message: error.to_string(),
                        });
                        return;
                    }
                };
                let mut sink = |event: Event| match event {
                    Event::Start { prompt_tokens } => {
                        let _ = sender.blocking_send(GenerationEvent::Start {
                            prompt_tokens: prompt_tokens as u32,
                        });
                    }
                    Event::Token { text, .. } => {
                        let _ = sender.blocking_send(GenerationEvent::Token { text });
                    }
                    Event::Finished(_) => {}
                };
                match generate(
                    &mut session,
                    &tokenizer,
                    &core_request,
                    &worker_cancel,
                    &mut sink,
                ) {
                    Ok(stats) => {
                        let outcome = RunOutcome {
                            prompt_tokens: stats.prompt_tokens as u32,
                            generated_tokens: stats.generated_tokens as u32,
                            prefill_ms: stats.prefill_ms as u32,
                            ttft_ms: stats.ttft_ms as u32,
                            decode_ms: stats.decode_ms as u32,
                            total_ms: stats.total_ms as u32,
                            tokens_per_second: stats.tokens_per_second,
                            ctx_used: session.position() as u32,
                            stop_reason: stats.stop_reason.as_str().to_string(),
                            token_latencies_ms: stats.token_latencies_ms,
                            peak_rss_bytes: resident_bytes(),
                        };
                        let _ = sender.blocking_send(GenerationEvent::Finished(outcome));
                    }
                    Err(error) => {
                        let _ = sender.blocking_send(GenerationEvent::Failed {
                            message: error.to_string(),
                        });
                    }
                }
                drop(permit);
            });

            Ok(GenerationHandle { events, cancel })
        }
    }
}

fn configure_thread_pool(threads: usize) {
    if threads == 0 {
        return;
    }
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global();
}

fn internal(error: impl std::fmt::Display) -> EngineError {
    EngineError::Internal(error.to_string())
}

/// Resident set size of this process, read from procfs (no extra dependency).
fn resident_bytes() -> u64 {
    let Ok(statm) = std::fs::read_to_string("/proc/self/statm") else {
        return 0;
    };
    let pages = statm
        .split_whitespace()
        .nth(1)
        .and_then(|field| field.parse::<u64>().ok())
        .unwrap_or(0);
    pages * page_size()
}

fn page_size() -> u64 {
    // 4 KiB on every platform this service runs on.
    4096
}
