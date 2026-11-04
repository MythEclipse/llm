//! Chat service in the Kana four-layer shape.
//!
//! `domain` knows nothing but `uuid`/`chrono`; `application` speaks only to
//! ports; `infrastructure` implements those ports with SeaORM and the engine;
//! `presentation` turns them into HTTP.

pub mod application;
pub mod domain;
pub mod infrastructure;
pub mod presentation;
