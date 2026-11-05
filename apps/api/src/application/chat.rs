//! Sending a message: build the turn list, start the engine, then record what
//! the run actually cost.

use uuid::Uuid;

use crate::application::error::ChatError;
use crate::application::ports::{
    ChatEngine, ChatTurn, GenerateRequest, GenerationHandle, RunOutcome, TurnRole,
};
use crate::domain::conversation::{ConversationRepository, NewMessage, Role, UNTITLED, title_from};
use crate::domain::stats::{NewRun, RunRepository, RunStats};

const DEFAULT_SYSTEM_PROMPT: &str = "You are a helpful assistant.";

pub struct SendMessageCommand {
    pub conversation_id: Uuid,
    pub content: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub seed: u64,
}

/// What the presentation layer needs to stream one answer.
pub struct ChatStream {
    pub handle: GenerationHandle,
    pub conversation_id: Uuid,
    pub model: String,
    pub sampler: String,
    pub prompt_tokens: u32,
}

pub struct SendMessageUseCase<C, E> {
    conversations: C,
    engine: E,
    model_name: String,
}

impl<C: ConversationRepository, E: ChatEngine> SendMessageUseCase<C, E> {
    pub fn new(conversations: C, engine: E, model_name: String) -> Self {
        Self {
            conversations,
            engine,
            model_name,
        }
    }

    pub async fn execute(&self, command: SendMessageCommand) -> Result<ChatStream, ChatError> {
        let content = command.content.trim();
        if content.is_empty() {
            return Err(ChatError::EmptyMessage);
        }
        let conversation = self
            .conversations
            .find(command.conversation_id)
            .await?
            .ok_or(ChatError::NotFound)?;

        let history = self.conversations.messages(conversation.id).await?;
        let mut turns = Vec::with_capacity(history.len() + 2);
        turns.push(ChatTurn {
            role: TurnRole::System,
            content: DEFAULT_SYSTEM_PROMPT.to_string(),
        });
        for message in &history {
            if let Some(role) = to_turn_role(message.role) {
                turns.push(ChatTurn {
                    role,
                    content: message.content.clone(),
                });
            }
        }
        turns.push(ChatTurn {
            role: TurnRole::User,
            content: content.to_string(),
        });

        let request = GenerateRequest {
            turns,
            max_tokens: command.max_tokens,
            temperature: command.temperature,
            top_p: command.top_p,
            top_k: command.top_k,
            min_p: command.min_p,
            repeat_penalty: command.repeat_penalty,
            seed: command.seed,
        };
        let sampler = request.sampler_summary();
        let handle = self.engine.start(request).await?;

        self.conversations
            .append_message(NewMessage {
                id: Uuid::new_v4(),
                conversation_id: conversation.id,
                role: Role::User,
                content: content.to_string(),
                token_count: 0,
            })
            .await?;
        if conversation.title == UNTITLED {
            self.conversations
                .rename(conversation.id, title_from(content))
                .await?;
        }

        Ok(ChatStream {
            handle,
            conversation_id: conversation.id,
            model: self.model_name.clone(),
            sampler,
            prompt_tokens: 0,
        })
    }
}

fn to_turn_role(role: Role) -> Option<TurnRole> {
    match role {
        Role::System => None,
        Role::User => Some(TurnRole::User),
        Role::Assistant => Some(TurnRole::Assistant),
    }
}

pub struct FinishChatCommand {
    pub conversation_id: Uuid,
    pub outcome: RunOutcome,
    pub assistant_text: String,
    pub model: String,
    pub sampler: String,
}

pub struct FinishChatUseCase<R, C> {
    runs: R,
    conversations: C,
}

impl<R: RunRepository, C: ConversationRepository> FinishChatUseCase<R, C> {
    pub fn new(runs: R, conversations: C) -> Self {
        Self {
            runs,
            conversations,
        }
    }

    /// Persist the assistant message and the run telemetry in one place so a
    /// cancelled or failed run still shows up in the statistics.
    pub async fn execute(&self, command: FinishChatCommand) -> Result<RunStats, ChatError> {
        let trimmed = command.assistant_text.trim_end();
        if !trimmed.is_empty() {
            self.conversations
                .append_message(NewMessage {
                    id: Uuid::new_v4(),
                    conversation_id: command.conversation_id,
                    role: Role::Assistant,
                    content: trimmed.to_string(),
                    token_count: command.outcome.generated_tokens,
                })
                .await?;
        }
        self.conversations.touch(command.conversation_id).await?;

        let run = self
            .runs
            .insert(NewRun {
                id: Uuid::new_v4(),
                conversation_id: command.conversation_id,
                model: command.model,
                prompt_tokens: command.outcome.prompt_tokens,
                generated_tokens: command.outcome.generated_tokens,
                prefill_ms: command.outcome.prefill_ms,
                ttft_ms: command.outcome.ttft_ms,
                decode_ms: command.outcome.decode_ms,
                total_ms: command.outcome.total_ms,
                tokens_per_second: command.outcome.tokens_per_second,
                ctx_used: command.outcome.ctx_used,
                stop_reason: command.outcome.stop_reason,
                sampler: command.sampler,
                token_latencies_ms: command.outcome.token_latencies_ms,
                peak_rss_bytes: command.outcome.peak_rss_bytes,
                build_hash: Some(command.outcome.build_hash),
            })
            .await?;
        tracing::info!(
            conversation_id = %command.conversation_id,
            generated_tokens = run.generated_tokens,
            tokens_per_second = run.tokens_per_second,
            stop_reason = %run.stop_reason,
            "run recorded"
        );
        Ok(run)
    }
}
