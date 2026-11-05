//! Engine adapter: the only place that knows both the port and llm-engine.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use llm_engine::chat::{ChatFormat, Role as EngineRole, Turn};
use llm_engine::manifest::BuildManifest;
use llm_engine::{
    Event, GenerationRequest as CoreRequest, GgufFile, Model, SamplingParams, Session, Tokenizer,
    generate, session::PREFILL_CHUNK,
};
use tokio::sync::{Semaphore, mpsc};
use tracing::{debug, info, instrument};

use crate::application::ports::{
    BuildInfo, ChatEngine, ChatTurn, EngineError, GenerateRequest, GenerationEvent,
    GenerationHandle, ModelInfo, PlanInfo, RunOutcome, TurnRole,
};

const STOP_TOKENS: [&str; 2] = ["<|im_end|>", "</s>"];

/// Owns the loaded model and runs one generation at a time.
pub struct LocalChatEngine {
    model: &'static Model<'static>,
    tokenizer: Arc<Tokenizer>,
    chat: ChatFormat,
    stop_tokens: Vec<u32>,
    info: ModelInfo,
    build_hash: String,
    slots: Arc<Semaphore>,
    context: usize,
}

impl LocalChatEngine {
    #[instrument(skip_all, fields(model = %path.display(), context, threads))]
    pub fn load(path: &Path, context: usize, threads: usize) -> Result<Self, EngineError> {
        configure_thread_pool(threads);
        let gguf = GgufFile::open(path).map_err(internal)?;
        let gguf: &'static GgufFile = Box::leak(Box::new(gguf));
        let model = Model::load(gguf).map_err(internal)?;
        let model: &'static Model<'static> = Box::leak(Box::new(model));
        let tokenizer = Tokenizer::from_gguf(gguf).map_err(internal)?;
        // The prompt shape comes from the file, not from a constant: a model
        // trained on another template answers badly when handed ChatML.
        let chat = ChatFormat::from_gguf(gguf, &tokenizer).map_err(internal)?;
        tracing::info!(
            template = chat.template().name(),
            bos = chat.bos(),
            vocabulary = tokenizer.family(),
            "chat format read from the file"
        );
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
            architecture: description.architecture.clone(),
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
            build: build_info(),
            plan: plan_info(model, context),
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
            chat,
            stop_tokens,
            build_hash: info.build.identity.clone(),
            info,
            slots: Arc::new(Semaphore::new(1)),
            context,
        })
    }

    #[instrument(skip_all, fields(turns = turns.len()))]
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
        self.chat.render(&turns, false)
    }

    /// Render the turns that still fit, dropping the oldest history first.
    ///
    /// `None` means even the newest turn alone does not fit the window.
    fn fit_turns(&self, turns: &[ChatTurn], max_tokens: usize) -> Option<(String, usize)> {
        let budget = self.context.checked_sub(max_tokens)?;
        let (kept, dropped) = keep_that_fits(turns, budget, |candidate| {
            self.tokenizer.encode(&self.prompt(candidate)).len()
        })?;
        Some((self.prompt(&kept), dropped))
    }
}

/// Keep the longest tail of `turns` whose prompt costs at most `budget` tokens.
///
/// A conversation outgrows any window, and refusing to answer is a worse answer
/// than forgetting how the chat started. The system turn always stays, and how
/// many trailing turns survive is monotone in prompt length, so the search
/// bisects instead of re-rendering once per dropped turn. Returns the kept turns
/// and how many were dropped.
fn keep_that_fits(
    turns: &[ChatTurn],
    budget: usize,
    mut measure: impl FnMut(&[ChatTurn]) -> usize,
) -> Option<(Vec<ChatTurn>, usize)> {
    let system = usize::from(matches!(
        turns.first().map(|turn| turn.role),
        Some(TurnRole::System)
    ));
    let body = &turns[system..];
    let keep = |tail: usize| {
        let mut start = body.len().saturating_sub(tail);
        // A history that begins with an assistant turn reads as a reply to
        // nothing. Dropping it only shortens the prompt, so the search stays
        // monotone.
        while start + 1 < body.len() && body[start].role == TurnRole::Assistant {
            start += 1;
        }
        let mut kept: Vec<ChatTurn> = turns[..system].to_vec();
        kept.extend_from_slice(&body[start..]);
        (kept, start)
    };

    let whole = keep(body.len());
    if measure(&whole.0) <= budget {
        return Some(whole);
    }
    let mut low = 1;
    let mut high = body.len();
    let mut best = None;
    while low <= high {
        let tail = low + (high - low) / 2;
        let candidate = keep(tail);
        if measure(&candidate.0) <= budget {
            best = Some(candidate);
            low = tail + 1;
        } else if tail == 1 {
            return best;
        } else {
            high = tail - 1;
        }
    }
    best
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
        let prepared = self
            .fit_turns(&request.turns, request.max_tokens)
            .map(|(prompt, dropped)| (self.tokenizer.encode(&prompt), dropped));
        if let Some((tokens, dropped_turns)) = &prepared {
            debug!(
                prompt_tokens = tokens.len(),
                dropped_turns,
                max_tokens = request.max_tokens,
                "generation request prepared"
            );
        }
        let context = self.context;
        let build_hash = self.build_hash.clone();
        let model = self.model;
        let tokenizer = Arc::clone(&self.tokenizer);
        let stop_tokens = self.stop_tokens.clone();
        let slots = Arc::clone(&self.slots);
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = Arc::clone(&cancel);

        async move {
            let Some((tokens, dropped_turns)) = prepared else {
                return Err(EngineError::ContextFull);
            };
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
                            dropped_turns: dropped_turns as u32,
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
                        info!(
                            prompt_tokens = stats.prompt_tokens,
                            generated_tokens = stats.generated_tokens,
                            total_ms = stats.total_ms,
                            tokens_per_second = stats.tokens_per_second,
                            "generation worker finished"
                        );
                        let outcome = RunOutcome {
                            build_hash: build_hash.clone(),
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

/// The artifact identity, straight from the build script's record.
fn build_info() -> BuildInfo {
    let manifest = BuildManifest::current();
    BuildInfo {
        arch: manifest.arch.to_string(),
        source_sha256: manifest.source_sha256.to_string(),
        ptx_sha256: manifest.ptx_sha256.map(str::to_owned),
        nvcc: manifest.nvcc.to_string(),
        kernels: manifest.kernels.len(),
        identity: manifest.identity(),
    }
}

/// Which kernels this model resolved to, and why any of them is the general one.
fn plan_info(model: &Model<'_>, context: usize) -> PlanInfo {
    let plan = model.compute().plan();
    let scratch = model.scratch_plan(PREFILL_CHUNK.min(context), context);
    PlanInfo {
        decode: plan.decode.tactic.to_string(),
        prefill: plan.prefill.tactic.to_string(),
        identity: plan.identity(),
        fallbacks: plan
            .fallbacks()
            .into_iter()
            .map(|(regime, reason)| format!("{regime}: {reason}"))
            .collect(),
        scratch_bytes: scratch.bytes(),
        scratch_reused_bytes: scratch.reused_floats() * std::mem::size_of::<f32>(),
        scratch_identity: scratch.identity(),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(role: TurnRole, content: &str) -> ChatTurn {
        ChatTurn {
            role,
            content: content.to_string(),
        }
    }

    /// Ten tokens per character, so a test budget reads like a prompt size.
    fn ten_per_char(turns: &[ChatTurn]) -> usize {
        turns.iter().map(|turn| turn.content.len()).sum::<usize>() * 10
    }

    #[test]
    fn a_conversation_that_fits_keeps_every_turn() {
        let turns = [
            turn(TurnRole::System, "sys"),
            turn(TurnRole::User, "hi"),
            turn(TurnRole::Assistant, "hello"),
            turn(TurnRole::User, "again"),
        ];
        let (kept, dropped) = keep_that_fits(&turns, 1000, ten_per_char).expect("fits");
        assert_eq!(kept.len(), 4);
        assert_eq!(dropped, 0);
    }

    #[test]
    fn the_oldest_turns_go_first_and_the_system_turn_stays() {
        let turns = [
            turn(TurnRole::System, "sys"),
            turn(TurnRole::User, "a".repeat(10).as_str()),
            turn(TurnRole::Assistant, "b".repeat(10).as_str()),
            turn(TurnRole::User, "c".repeat(10).as_str()),
            turn(TurnRole::Assistant, "d".repeat(10).as_str()),
            turn(TurnRole::User, "ee"),
        ];
        // Whole prompt is 450; 250 holds the system turn plus the last three.
        let (kept, dropped) = keep_that_fits(&turns, 250, ten_per_char).expect("some fit");
        assert_eq!(dropped, 2);
        assert_eq!(kept.len(), 4);
        assert_eq!(kept[0].role, TurnRole::System);
        assert_eq!(kept.last().expect("kept").content, "ee");
    }

    #[test]
    fn a_history_that_would_start_with_a_reply_drops_that_reply() {
        let turns = [
            turn(TurnRole::System, "sys"),
            turn(TurnRole::User, "a".repeat(10).as_str()),
            turn(TurnRole::Assistant, "b".repeat(10).as_str()),
            turn(TurnRole::User, "cc"),
        ];
        // 240 cannot hold the whole prompt (250), and the tail that would fit is
        // [assistant, user]: the reply to nothing goes too.
        let (kept, dropped) = keep_that_fits(&turns, 240, ten_per_char).expect("some fit");
        assert_eq!(dropped, 2);
        assert_eq!(kept.len(), 2);
        assert_eq!(kept[1].role, TurnRole::User);
        assert_eq!(kept[1].content, "cc");
    }

    #[test]
    fn a_single_turn_wider_than_the_window_reports_no_fit() {
        let turns = [
            turn(TurnRole::System, "sys"),
            turn(TurnRole::User, "x".repeat(1000).as_str()),
        ];
        assert!(keep_that_fits(&turns, 100, ten_per_char).is_none());
    }
}
