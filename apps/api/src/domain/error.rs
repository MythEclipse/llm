/// Failure shared by every repository.
#[derive(Debug, thiserror::Error)]
pub enum RepositoryError {
    #[error("record not found")]
    NotFound,
    #[error("conflict: {0}")]
    Conflict(String),
    #[error("database error: {0}")]
    Database(String),
}
