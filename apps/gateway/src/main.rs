//! Assembles the service: load the engine, open the database, serve the SPA.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use axum::Router;
use sea_orm::Database;
use sea_orm_migration::MigratorTrait;
use tower_http::services::{ServeDir, ServeFile};
use tracing_subscriber::EnvFilter;

use llm_api::infrastructure::{LocalChatEngine, SeaOrmConversationRepository, SeaOrmRunRepository};
use llm_api::presentation::{AppState, build_router};

const DEFAULT_MODEL: &str = "models/MiniCPM5-2B-Q8_0.gguf";
const DEFAULT_DATABASE: &str = "sqlite://llm.db?mode=rwc";
const DEFAULT_ADDRESS: &str = "127.0.0.1:8099";
const DEFAULT_CONTEXT: usize = 8192;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_env("LLM_LOG").unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let model_path = PathBuf::from(env_or("LLM_MODEL", DEFAULT_MODEL));
    let context = env_parse("LLM_CONTEXT", DEFAULT_CONTEXT);
    let threads = env_parse("LLM_THREADS", available_threads());
    let address: SocketAddr = env_or("LLM_ADDRESS", DEFAULT_ADDRESS).parse()?;

    let engine = Arc::new(LocalChatEngine::load(&model_path, context, threads)?);
    let database_url = env_or("LLM_DATABASE", DEFAULT_DATABASE);
    let db = Database::connect(&database_url).await?;
    migration::Migrator::up(&db, None).await?;

    let model_name = model_path
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model".to_string());
    let state = AppState::new(
        SeaOrmConversationRepository::new(db.clone()),
        SeaOrmRunRepository::new(db),
        Arc::clone(&engine),
        model_name,
    );

    let app = Router::new()
        .merge(build_router(state))
        .fallback_service(ServeDir::new("web").fallback(ServeFile::new("web/index.html")));

    tracing::info!(%address, "listening");
    let listener = tokio::net::TcpListener::bind(address).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn env_parse<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn available_threads() -> usize {
    std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(4)
}
