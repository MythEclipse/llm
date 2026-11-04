use std::fmt;

/// Everything that can go wrong inside the engine.
///
/// Kept as one enum because every layer of the engine (parser, kernel, tokenizer,
/// model, session) fails for the same handful of reasons and callers only need to
/// distinguish "bad file", "bad request" and "ran out of room".
#[derive(Debug)]
pub enum EngineError {
    Io(String),
    Format(String),
    MissingMetadata(String),
    MissingTensor(String),
    Unsupported(String),
    ContextFull { used: usize, capacity: usize },
    Cancelled,
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(message) => write!(f, "io error: {message}"),
            Self::Format(message) => write!(f, "malformed gguf: {message}"),
            Self::MissingMetadata(key) => write!(f, "missing metadata key: {key}"),
            Self::MissingTensor(name) => write!(f, "missing tensor: {name}"),
            Self::Unsupported(message) => write!(f, "unsupported: {message}"),
            Self::ContextFull { used, capacity } => {
                write!(f, "context full: {used} of {capacity} tokens used")
            }
            Self::Cancelled => write!(f, "generation cancelled"),
        }
    }
}

impl std::error::Error for EngineError {}

impl From<std::io::Error> for EngineError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error.to_string())
    }
}
