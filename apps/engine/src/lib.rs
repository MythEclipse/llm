//! GGUF inference engine: parsing, quantized kernels, tokenizer, forward pass.

pub mod byte_level;
pub mod chat;
pub mod compute;
mod cuda;
pub mod deltanet;
pub mod error;
pub mod generate;
pub mod gguf;
pub mod manifest;
pub mod model;
pub mod ops;
pub mod quant;
pub mod sampler;
pub mod session;
pub mod tactics;
pub mod tensor;
pub mod tokenizer;
pub mod workspace;

pub use compute::{Activation, AttentionSpec, Compute};
pub use cuda::Graph;
pub use error::EngineError;
pub use generate::{Event, GenerationRequest, RunStats, StopReason, generate};
pub use gguf::{DType, GgufFile, TensorInfo};
pub use model::{Config, KvCache, Model, Scratch};
pub use sampler::SamplingParams;
pub use session::Session;
pub use tokenizer::{StreamDecoder, Tokenizer};
