//! Conversation aggregate: a thread of messages and the port that stores it.

use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::domain::error::RepositoryError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "system" => Some(Self::System),
            "user" => Some(Self::User),
            "assistant" => Some(Self::Assistant),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Conversation {
    pub id: Uuid,
    pub title: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct Message {
    pub id: Uuid,
    pub conversation_id: Uuid,
    pub role: Role,
    pub content: String,
    pub token_count: u32,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct NewConversation {
    pub id: Uuid,
    pub title: String,
}

#[derive(Debug, Clone)]
pub struct NewMessage {
    pub id: Uuid,
    pub conversation_id: Uuid,
    pub role: Role,
    pub content: String,
    pub token_count: u32,
}

/// Title shown for a conversation that has no user message yet.
pub const UNTITLED: &str = "Percakapan baru";

/// Derive a title from the first user message: first line, bounded length.
pub fn title_from(message: &str) -> String {
    let first_line = message.lines().next().unwrap_or("").trim();
    let mut title: String = first_line.chars().take(60).collect();
    if title.is_empty() {
        return UNTITLED.to_string();
    }
    if first_line.chars().count() > 60 {
        title.push('…');
    }
    title
}

pub trait ConversationRepository: Send + Sync {
    fn create(
        &self,
        new: NewConversation,
    ) -> impl Future<Output = Result<Conversation, RepositoryError>> + Send;

    fn list(&self) -> impl Future<Output = Result<Vec<Conversation>, RepositoryError>> + Send;

    fn find(
        &self,
        id: Uuid,
    ) -> impl Future<Output = Result<Option<Conversation>, RepositoryError>> + Send;

    fn messages(
        &self,
        conversation_id: Uuid,
    ) -> impl Future<Output = Result<Vec<Message>, RepositoryError>> + Send;

    fn append_message(
        &self,
        new: NewMessage,
    ) -> impl Future<Output = Result<Message, RepositoryError>> + Send;

    fn rename(
        &self,
        id: Uuid,
        title: String,
    ) -> impl Future<Output = Result<(), RepositoryError>> + Send;

    fn touch(&self, id: Uuid) -> impl Future<Output = Result<(), RepositoryError>> + Send;

    fn delete(&self, id: Uuid) -> impl Future<Output = Result<(), RepositoryError>> + Send;
}
