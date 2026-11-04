//! Conversation endpoints and their wire shapes.

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tracing::instrument;
use uuid::Uuid;
use zod_rs::prelude::{Schema as _, ZodSchema};

use crate::application::ports::ChatEngine;
use crate::application::{
    ConversationDetailUseCase, CreateConversationCommand, CreateConversationUseCase,
    DeleteConversationUseCase, ListConversationsUseCase,
};
use crate::domain::conversation::{Conversation, Message};
use crate::presentation::error::AppError;
use crate::presentation::state::AppState;
use crate::presentation::validation::validate;

#[derive(Debug, Deserialize, Serialize, ZodSchema)]
pub struct CreateConversationRequest {
    #[serde(default)]
    #[zod(max_length(80))]
    pub title: String,
}

#[derive(Debug, Serialize)]
pub struct ConversationResponse {
    pub id: Uuid,
    pub title: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl From<Conversation> for ConversationResponse {
    fn from(conversation: Conversation) -> Self {
        Self {
            id: conversation.id,
            title: conversation.title,
            created_at: conversation.created_at,
            updated_at: conversation.updated_at,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct MessageResponse {
    pub id: Uuid,
    pub role: String,
    pub content: String,
    pub token_count: u32,
    pub created_at: DateTime<Utc>,
}

impl From<Message> for MessageResponse {
    fn from(message: Message) -> Self {
        Self {
            id: message.id,
            role: message.role.as_str().to_string(),
            content: message.content,
            token_count: message.token_count,
            created_at: message.created_at,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ConversationDetailResponse {
    pub conversation: ConversationResponse,
    pub messages: Vec<MessageResponse>,
}

#[instrument(skip_all, fields(conversation_id = %id))]
pub async fn detail<E: ChatEngine + 'static>(
    State(state): State<AppState<E>>,
    Path(id): Path<Uuid>,
) -> Result<Json<ConversationDetailResponse>, AppError> {
    let detail = ConversationDetailUseCase::new(state.conversations.clone())
        .execute(id)
        .await?;
    Ok(Json(ConversationDetailResponse {
        conversation: detail.conversation.into(),
        messages: detail
            .messages
            .into_iter()
            .map(MessageResponse::from)
            .collect(),
    }))
}

#[instrument(skip_all)]
pub async fn list<E: ChatEngine + 'static>(
    State(state): State<AppState<E>>,
) -> Result<Json<Vec<ConversationResponse>>, AppError> {
    let conversations = ListConversationsUseCase::new(state.conversations.clone())
        .execute()
        .await?;
    Ok(Json(
        conversations
            .into_iter()
            .map(ConversationResponse::from)
            .collect(),
    ))
}

#[instrument(skip_all, fields(title = %body.title))]
pub async fn create<E: ChatEngine + 'static>(
    State(state): State<AppState<E>>,
    Json(body): Json<CreateConversationRequest>,
) -> Result<(StatusCode, Json<ConversationResponse>), AppError> {
    let body = validate(body)?;
    let title = (!body.title.trim().is_empty()).then_some(body.title);
    let conversation = CreateConversationUseCase::new(state.conversations.clone())
        .execute(CreateConversationCommand { title })
        .await?;
    Ok((
        StatusCode::CREATED,
        Json(ConversationResponse::from(conversation)),
    ))
}

#[instrument(skip_all, fields(conversation_id = %id))]
pub async fn delete<E: ChatEngine + 'static>(
    State(state): State<AppState<E>>,
    Path(id): Path<Uuid>,
) -> Result<StatusCode, AppError> {
    DeleteConversationUseCase::new(state.conversations.clone())
        .execute(id)
        .await?;
    Ok(StatusCode::NO_CONTENT)
}
