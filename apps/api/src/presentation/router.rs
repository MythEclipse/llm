//! Route table. Everything under `/api`; the SPA is served by the gateway.

use axum::Router;
use axum::routing::{get, post};
use tower_http::trace::TraceLayer;

use crate::application::ports::ChatEngine;
use crate::presentation::chat;
use crate::presentation::conversation;
use crate::presentation::model;
use crate::presentation::state::AppState;
use crate::presentation::stats;

pub fn build_router<E: ChatEngine + 'static>(state: AppState<E>) -> Router {
    Router::new()
        .route("/api/health", get(health))
        .route("/api/model", get(model::model_info))
        .route(
            "/api/conversations",
            get(conversation::list).post(conversation::create),
        )
        .route(
            "/api/conversations/{id}",
            get(conversation::detail).delete(conversation::delete),
        )
        .route("/api/conversations/{id}/messages", post(chat::send_message))
        .route("/api/conversations/{id}/stop", post(chat::stop))
        .route("/api/stats/overview", get(stats::overview))
        .route("/api/stats/runs", get(stats::runs))
        .route("/api/stats/timeseries", get(stats::timeseries))
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(|request: &axum::http::Request<_>| {
                    tracing::info_span!(
                        "http_request",
                        method = %request.method(),
                        uri = %request.uri(),
                        version = ?request.version()
                    )
                })
                .on_response(
                    |response: &axum::http::Response<_>,
                     latency: std::time::Duration,
                     _span: &tracing::Span| {
                        tracing::info!(
                            status = response.status().as_u16(),
                            latency_ms = latency.as_secs_f64() * 1000.0,
                            "http response"
                        );
                    },
                ),
        )
        .with_state(state)
}

async fn health() -> &'static str {
    "ok"
}
