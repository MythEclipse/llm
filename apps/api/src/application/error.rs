use crate::application::ports::EngineError;
use crate::domain::RepositoryError;

#[derive(Debug, thiserror::Error)]
pub enum ChatError {
    #[error("conversation not found")]
    NotFound,
    #[error("message is empty")]
    EmptyMessage,
    #[error("{0}")]
    InvalidRequest(String),
    #[error(transparent)]
    Engine(#[from] EngineError),
    #[error(transparent)]
    Repository(#[from] RepositoryError),
}
