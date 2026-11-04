//! HTTP error mapping. Domain and engine failures become status codes here and
//! nowhere else.

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

use crate::application::ChatError;
use crate::application::ports::EngineError;
use crate::domain::RepositoryError;

#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("{0}")]
    BadRequest(String),
    #[error("{0}")]
    NotFound(String),
    #[error("{0}")]
    Conflict(String),
    #[error("{0}")]
    Busy(String),
    #[error("{0}")]
    Internal(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let status = match &self {
            Self::BadRequest(_) => StatusCode::BAD_REQUEST,
            Self::NotFound(_) => StatusCode::NOT_FOUND,
            Self::Conflict(_) => StatusCode::CONFLICT,
            Self::Busy(_) => StatusCode::TOO_MANY_REQUESTS,
            Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        if status == StatusCode::INTERNAL_SERVER_ERROR {
            tracing::error!(error = %self, "request failed");
        }
        (status, Json(json!({ "error": self.to_string() }))).into_response()
    }
}

impl From<ChatError> for AppError {
    fn from(error: ChatError) -> Self {
        match error {
            ChatError::NotFound => Self::NotFound("conversation not found".to_string()),
            ChatError::EmptyMessage => Self::BadRequest("message is empty".to_string()),
            ChatError::InvalidRequest(message) => Self::BadRequest(message),
            ChatError::Engine(engine) => Self::from(engine),
            ChatError::Repository(repository) => Self::from(repository),
        }
    }
}

impl From<EngineError> for AppError {
    fn from(error: EngineError) -> Self {
        match error {
            EngineError::Busy => Self::Busy("another generation is running".to_string()),
            EngineError::ContextFull => {
                Self::BadRequest("prompt plus reply does not fit in the context window".to_string())
            }
            EngineError::Internal(message) => Self::Internal(message),
        }
    }
}

impl From<RepositoryError> for AppError {
    fn from(error: RepositoryError) -> Self {
        match error {
            RepositoryError::NotFound => Self::NotFound("record not found".to_string()),
            RepositoryError::Conflict(message) => Self::Conflict(message),
            RepositoryError::Database(message) => Self::Internal(message),
        }
    }
}
