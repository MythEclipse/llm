//! HTTP surface tests: a scripted engine plus an in-memory SQLite database, so
//! the router, the use cases, the persistence and the SSE framing are all
//! exercised without loading a 2.7 GB model.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use sea_orm::{ConnectOptions, Database};
use sea_orm_migration::MigratorTrait;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tower::ServiceExt;

use llm_api::application::ports::{
    BuildInfo, ChatEngine, EngineError, GenerateRequest, GenerationEvent, GenerationHandle,
    ModelInfo, PlanInfo, RunOutcome,
};
use llm_api::infrastructure::{SeaOrmConversationRepository, SeaOrmRunRepository};
use llm_api::presentation::{AppState, build_router};

#[derive(Clone)]
struct FakeEngine {
    info: ModelInfo,
}

impl Default for FakeEngine {
    fn default() -> Self {
        Self {
            info: ModelInfo {
                name: "fake".to_string(),
                architecture: "llama".to_string(),
                layers: 2,
                embedding: 8,
                feed_forward: 16,
                heads: 2,
                kv_heads: 1,
                head_dim: 4,
                context: 512,
                vocab: 100,
                tensors: 3,
                quantized_tensors: 2,
                weight_bytes: 1024,
                kv_bytes: 2048,
                threads: 1,
                device: None,
                build: BuildInfo {
                    arch: "sm_89".to_string(),
                    source_sha256: "0".repeat(64),
                    ptx_sha256: Some("1".repeat(64)),
                    nvcc: "release 13.4".to_string(),
                    kernels: 21,
                    identity: "2".repeat(64),
                },
                plan: PlanInfo {
                    decode: "attention.decode_split".to_string(),
                    prefill: "attention.prefill_mma_f16".to_string(),
                    identity: "3".repeat(64),
                    fallbacks: Vec::new(),
                    scratch_bytes: 4096,
                    scratch_reused_bytes: 1024,
                    scratch_identity: "4".repeat(64),
                },
            },
        }
    }
}

impl ChatEngine for FakeEngine {
    fn info(&self) -> impl Future<Output = Result<ModelInfo, EngineError>> + Send {
        let info = self.info.clone();
        async move { Ok(info) }
    }

    fn start(
        &self,
        request: GenerateRequest,
    ) -> impl Future<Output = Result<GenerationHandle, EngineError>> + Send {
        let prompt_tokens = request.turns.len() as u32 + 5;
        async move {
            let (sender, events) = mpsc::channel(8);
            tokio::spawn(async move {
                let _ = sender
                    .send(GenerationEvent::Start {
                        prompt_tokens,
                        dropped_turns: 0,
                    })
                    .await;
                for chunk in ["Halo", " dunia"] {
                    let _ = sender
                        .send(GenerationEvent::Token {
                            text: chunk.to_string(),
                        })
                        .await;
                }
                let _ = sender
                    .send(GenerationEvent::Finished(RunOutcome {
                        build_hash: "2".repeat(64),
                        prompt_tokens,
                        generated_tokens: 2,
                        prefill_ms: 100,
                        ttft_ms: 120,
                        decode_ms: 40,
                        total_ms: 140,
                        tokens_per_second: 25.0,
                        ctx_used: prompt_tokens + 2,
                        stop_reason: "eos".to_string(),
                        token_latencies_ms: vec![20.0, 20.0],
                        peak_rss_bytes: 1024,
                    }))
                    .await;
            });
            Ok(GenerationHandle {
                events,
                cancel: Arc::new(AtomicBool::new(false)),
            })
        }
    }
}

async fn app() -> Router {
    let mut options = ConnectOptions::new("sqlite::memory:".to_string());
    options.max_connections(1).sqlx_logging(false);
    let db = Database::connect(options)
        .await
        .expect("in-memory database");
    migration::Migrator::up(&db, None)
        .await
        .expect("migrations run");

    let state = AppState::new(
        SeaOrmConversationRepository::new(db.clone()),
        SeaOrmRunRepository::new(db),
        Arc::new(FakeEngine::default()),
        "fake".to_string(),
    );
    build_router(state)
}

async fn body_json(response: axum::response::Response) -> Value {
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("body collects")
        .to_bytes();
    serde_json::from_slice(&bytes).expect("body is json")
}

async fn body_text(response: axum::response::Response) -> String {
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("body collects")
        .to_bytes();
    String::from_utf8_lossy(&bytes).into_owned()
}

fn post(uri: &str, body: Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .expect("request builds")
}

fn get(uri: &str) -> Request<Body> {
    Request::builder()
        .uri(uri)
        .body(Body::empty())
        .expect("request builds")
}

#[tokio::test]
async fn model_info_reports_the_engine_geometry() {
    let response = app().await.oneshot(get("/api/model")).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["name"], "fake");
    assert_eq!(body["layers"], 2);
    assert_eq!(body["context"], 512);
}

#[tokio::test]
async fn creating_a_conversation_returns_201_and_appears_in_the_list() {
    let router = app().await;

    let created = router
        .clone()
        .oneshot(post("/api/conversations", json!({ "title": "Uji" })))
        .await
        .unwrap();
    assert_eq!(created.status(), StatusCode::CREATED);
    let created = body_json(created).await;
    assert_eq!(created["title"], "Uji");

    let list = router.oneshot(get("/api/conversations")).await.unwrap();
    let list = body_json(list).await;
    assert_eq!(list.as_array().unwrap().len(), 1);
    assert_eq!(list[0]["id"], created["id"]);
}

#[tokio::test]
async fn empty_title_falls_back_to_the_untitled_label() {
    let router = app().await;
    let created = router
        .oneshot(post("/api/conversations", json!({ "title": "   " })))
        .await
        .unwrap();
    let created = body_json(created).await;
    assert_eq!(created["title"], "Percakapan baru");
}

#[tokio::test]
async fn sending_a_message_streams_events_and_records_the_run() {
    let router = app().await;
    let created = body_json(
        router
            .clone()
            .oneshot(post("/api/conversations", json!({ "title": "" })))
            .await
            .unwrap(),
    )
    .await;
    let id = created["id"].as_str().unwrap().to_string();

    let response = router
        .clone()
        .oneshot(post(
            &format!("/api/conversations/{id}/messages"),
            json!({ "content": "Halo model" }),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers()["content-type"].to_str().unwrap(),
        "text/event-stream"
    );
    let stream = body_text(response).await;
    assert!(stream.contains("event: start"), "missing start: {stream}");
    assert!(stream.contains("event: token"), "missing token: {stream}");
    assert!(stream.contains("event: stats"), "missing stats: {stream}");
    assert!(stream.contains("\"Halo\""), "missing first token: {stream}");
    assert!(
        stream.contains("\" dunia\""),
        "missing second token: {stream}"
    );

    let detail = body_json(
        router
            .clone()
            .oneshot(get(&format!("/api/conversations/{id}")))
            .await
            .unwrap(),
    )
    .await;
    let messages = detail["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0]["role"], "user");
    assert_eq!(messages[0]["content"], "Halo model");
    assert_eq!(messages[1]["role"], "assistant");
    assert_eq!(messages[1]["content"], "Halo dunia");
    assert_eq!(detail["conversation"]["title"], "Halo model");

    let overview = body_json(router.oneshot(get("/api/stats/overview")).await.unwrap()).await;
    assert_eq!(overview["total_runs"], 1);
    assert_eq!(overview["total_tokens"], 2);
    assert!((overview["average_tokens_per_second"].as_f64().unwrap() - 25.0).abs() < 1e-6);
}

#[tokio::test]
async fn empty_message_is_rejected_with_400() {
    let router = app().await;
    let created = body_json(
        router
            .clone()
            .oneshot(post("/api/conversations", json!({ "title": "" })))
            .await
            .unwrap(),
    )
    .await;
    let id = created["id"].as_str().unwrap().to_string();

    let response = router
        .oneshot(post(
            &format!("/api/conversations/{id}/messages"),
            json!({ "content": "" }),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn unknown_conversation_is_404() {
    let router = app().await;
    let response = router
        .oneshot(get(
            "/api/conversations/00000000-0000-0000-0000-000000000000",
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn deleting_a_conversation_returns_204_and_removes_it() {
    let router = app().await;
    let created = body_json(
        router
            .clone()
            .oneshot(post("/api/conversations", json!({ "title": "Hapus" })))
            .await
            .unwrap(),
    )
    .await;
    let id = created["id"].as_str().unwrap().to_string();

    let deleted = router
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(format!("/api/conversations/{id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(deleted.status(), StatusCode::NO_CONTENT);

    let list = body_json(router.oneshot(get("/api/conversations")).await.unwrap()).await;
    assert!(list.as_array().unwrap().is_empty());
}

#[tokio::test]
async fn stop_on_idle_conversation_is_accepted() {
    let router = app().await;
    let created = body_json(
        router
            .clone()
            .oneshot(post("/api/conversations", json!({ "title": "" })))
            .await
            .unwrap(),
    )
    .await;
    let id = created["id"].as_str().unwrap().to_string();

    let response = router
        .oneshot(post(&format!("/api/conversations/{id}/stop"), json!({})))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::ACCEPTED);
}

#[tokio::test]
async fn timeseries_returns_buckets_for_recorded_runs() {
    let router = app().await;
    let created = body_json(
        router
            .clone()
            .oneshot(post("/api/conversations", json!({ "title": "" })))
            .await
            .unwrap(),
    )
    .await;
    let id = created["id"].as_str().unwrap().to_string();
    let _ = body_text(
        router
            .clone()
            .oneshot(post(
                &format!("/api/conversations/{id}/messages"),
                json!({ "content": "satu" }),
            ))
            .await
            .unwrap(),
    )
    .await;

    let buckets = body_json(
        router
            .oneshot(get("/api/stats/timeseries?days=7"))
            .await
            .unwrap(),
    )
    .await;
    let buckets = buckets.as_array().unwrap();
    assert_eq!(buckets.len(), 1);
    assert_eq!(buckets[0]["runs"], 1);
    assert_eq!(buckets[0]["tokens"], 2);
}
