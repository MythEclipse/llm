//! GGUF inference engine: parsing, quantized kernels, tokenizer, forward pass.

pub mod byte_level;
pub mod chat;
pub mod compute;
mod cuda;
pub mod error;
pub mod generate;
pub mod gguf;
pub mod model;
pub mod ops;
pub mod quant;
pub mod sampler;
pub mod session;
pub mod tensor;
pub mod tokenizer;

pub use compute::{Activation, AttentionSpec, Compute};
pub use error::EngineError;
pub use generate::{Event, GenerationRequest, RunStats, StopReason, generate};
pub use gguf::{DType, GgufFile, TensorInfo};
pub use model::{Config, KvCache, Model};
pub use sampler::SamplingParams;
pub use session::Session;
pub use tokenizer::{StreamDecoder, Tokenizer};
