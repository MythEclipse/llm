//! Request validation at the presentation boundary.
//!
//! Each request DTO carries its own zod schema; handlers call [`validate`] once
//! so the value checks serde cannot express (lengths, ranges) fail with a 400
//! before any use case runs.

use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::presentation::error::AppError;

pub trait ValidatedRequest: Sized + Serialize + DeserializeOwned {
    fn validate_against_schema(value: &Value) -> Result<Self, zod_rs::__private::ValidationResult>;
}

pub fn validate<T: ValidatedRequest>(request: T) -> Result<T, AppError> {
    let value = serde_json::to_value(&request)
        .map_err(|error| AppError::Internal(format!("request re-encode failed: {error}")))?;
    T::validate_against_schema(&value).map_err(|issues| {
        tracing::warn!(%issues, "request validation failed");
        AppError::BadRequest(format!("Invalid request:{issues}"))
    })
}

macro_rules! impl_validated_request {
    ($($request:ty),* $(,)?) => {
        $(
            impl ValidatedRequest for $request {
                fn validate_against_schema(
                    value: &Value,
                ) -> Result<Self, zod_rs::__private::ValidationResult> {
                    Self::validate_and_parse(value)
                }
            }
        )*
    };
}

impl_validated_request!(
    crate::presentation::conversation::CreateConversationRequest,
    crate::presentation::chat::SendMessageRequest,
);
