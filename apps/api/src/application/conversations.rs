//! Conversation lifecycle: create, list, read, delete.

use uuid::Uuid;

use crate::application::error::ChatError;
use crate::domain::conversation::{
    Conversation, ConversationRepository, Message, NewConversation, UNTITLED,
};

pub struct CreateConversationCommand {
    pub title: Option<String>,
}

pub struct CreateConversationUseCase<R> {
    repository: R,
}

impl<R: ConversationRepository> CreateConversationUseCase<R> {
    pub fn new(repository: R) -> Self {
        Self { repository }
    }

    pub async fn execute(
        &self,
        command: CreateConversationCommand,
    ) -> Result<Conversation, ChatError> {
        let title = command
            .title
            .map(|title| title.trim().to_string())
            .filter(|title| !title.is_empty())
            .unwrap_or_else(|| UNTITLED.to_string());
        let conversation = self
            .repository
            .create(NewConversation {
                id: Uuid::new_v4(),
                title,
            })
            .await?;
        tracing::info!(conversation_id = %conversation.id, "conversation created");
        Ok(conversation)
    }
}

pub struct ListConversationsUseCase<R> {
    repository: R,
}

impl<R: ConversationRepository> ListConversationsUseCase<R> {
    pub fn new(repository: R) -> Self {
        Self { repository }
    }

    pub async fn execute(&self) -> Result<Vec<Conversation>, ChatError> {
        Ok(self.repository.list().await?)
    }
}

pub struct ConversationDetail {
    pub conversation: Conversation,
    pub messages: Vec<Message>,
}

pub struct ConversationDetailUseCase<R> {
    repository: R,
}

impl<R: ConversationRepository> ConversationDetailUseCase<R> {
    pub fn new(repository: R) -> Self {
        Self { repository }
    }

    pub async fn execute(&self, id: Uuid) -> Result<ConversationDetail, ChatError> {
        let conversation = self.repository.find(id).await?.ok_or(ChatError::NotFound)?;
        let messages = self.repository.messages(id).await?;
        Ok(ConversationDetail {
            conversation,
            messages,
        })
    }
}

pub struct DeleteConversationUseCase<R> {
    repository: R,
}

impl<R: ConversationRepository> DeleteConversationUseCase<R> {
    pub fn new(repository: R) -> Self {
        Self { repository }
    }

    pub async fn execute(&self, id: Uuid) -> Result<(), ChatError> {
        self.repository.delete(id).await?;
        tracing::info!(conversation_id = %id, "conversation deleted");
        Ok(())
    }
}
