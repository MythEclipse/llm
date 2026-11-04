//! Run telemetry persistence and the aggregates the dashboard reads.

use chrono::Utc;
use sea_orm::{
    ActiveModelTrait, ConnectionTrait, DatabaseConnection, DbBackend, DbErr, EntityTrait,
    QueryOrder, QuerySelect, Set, Statement,
};
use uuid::Uuid;

use crate::domain::error::RepositoryError;
use crate::domain::stats::{DayBucket, NewRun, Overview, RunRepository, RunStats};
use crate::infrastructure::repository::entities::run;

#[derive(Clone)]
pub struct SeaOrmRunRepository {
    db: DatabaseConnection,
}

impl SeaOrmRunRepository {
    pub fn new(db: DatabaseConnection) -> Self {
        Self { db }
    }
}

impl From<run::Model> for RunStats {
    fn from(model: run::Model) -> Self {
        Self {
            id: Uuid::parse_str(&model.id).unwrap_or_default(),
            conversation_id: Uuid::parse_str(&model.conversation_id).unwrap_or_default(),
            model: model.model,
            prompt_tokens: model.prompt_tokens.max(0) as u32,
            generated_tokens: model.generated_tokens.max(0) as u32,
            prefill_ms: model.prefill_ms.max(0) as u32,
            ttft_ms: model.ttft_ms.max(0) as u32,
            decode_ms: model.decode_ms.max(0) as u32,
            total_ms: model.total_ms.max(0) as u32,
            tokens_per_second: model.tokens_per_second,
            ctx_used: model.ctx_used.max(0) as u32,
            stop_reason: model.stop_reason,
            sampler: model.sampler,
            token_latencies_ms: serde_json::from_str(&model.token_latencies_ms).unwrap_or_default(),
            peak_rss_bytes: model.peak_rss_bytes.max(0) as u64,
            created_at: model.created_at.with_timezone(&Utc),
        }
    }
}

fn map_db_err(error: DbErr) -> RepositoryError {
    tracing::error!(error = %error, "run repository failure");
    RepositoryError::Database(error.to_string())
}

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = ((sorted.len() - 1) as f64 * fraction).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}

impl RunRepository for SeaOrmRunRepository {
    async fn insert(&self, run: NewRun) -> Result<RunStats, RepositoryError> {
        let model = run::ActiveModel {
            id: Set(run.id.to_string()),
            conversation_id: Set(run.conversation_id.to_string()),
            model: Set(run.model),
            prompt_tokens: Set(run.prompt_tokens as i64),
            generated_tokens: Set(run.generated_tokens as i64),
            prefill_ms: Set(run.prefill_ms as i64),
            ttft_ms: Set(run.ttft_ms as i64),
            decode_ms: Set(run.decode_ms as i64),
            total_ms: Set(run.total_ms as i64),
            tokens_per_second: Set(run.tokens_per_second),
            ctx_used: Set(run.ctx_used as i64),
            stop_reason: Set(run.stop_reason),
            sampler: Set(run.sampler),
            token_latencies_ms: Set(
                serde_json::to_string(&run.token_latencies_ms).unwrap_or_else(|_| "[]".to_string())
            ),
            peak_rss_bytes: Set(run.peak_rss_bytes as i64),
            created_at: Set(Utc::now().into()),
        };
        let inserted = model.insert(&self.db).await.map_err(map_db_err)?;
        Ok(RunStats::from(inserted))
    }

    async fn overview(&self) -> Result<Overview, RepositoryError> {
        let totals = self
            .db
            .query_one(Statement::from_string(
                DbBackend::Sqlite,
                "SELECT COUNT(*) AS runs, \
                        COALESCE(SUM(generated_tokens), 0) AS tokens, \
                        COALESCE(AVG(tokens_per_second), 0) AS avg_tps, \
                        COALESCE(AVG(ttft_ms), 0) AS avg_ttft, \
                        COALESCE(SUM(CASE WHEN date(created_at) = date('now') THEN 1 ELSE 0 END), 0) AS runs_today, \
                        COALESCE(SUM(CASE WHEN date(created_at) = date('now') THEN generated_tokens ELSE 0 END), 0) AS tokens_today \
                 FROM runs"
                    .to_string(),
            ))
            .await
            .map_err(map_db_err)?;

        let mut overview = Overview::default();
        if let Some(row) = totals {
            overview.total_runs = row.try_get::<i64>("", "runs").unwrap_or(0).max(0) as u64;
            overview.total_tokens = row.try_get::<i64>("", "tokens").unwrap_or(0).max(0) as u64;
            overview.average_tokens_per_second = row.try_get::<f64>("", "avg_tps").unwrap_or(0.0);
            overview.average_ttft_ms = row.try_get::<f64>("", "avg_ttft").unwrap_or(0.0);
            overview.runs_today = row.try_get::<i64>("", "runs_today").unwrap_or(0).max(0) as u64;
            overview.tokens_today =
                row.try_get::<i64>("", "tokens_today").unwrap_or(0).max(0) as u64;
        }

        let speeds = self
            .db
            .query_all(Statement::from_string(
                DbBackend::Sqlite,
                "SELECT tokens_per_second FROM runs ORDER BY tokens_per_second".to_string(),
            ))
            .await
            .map_err(map_db_err)?;
        let mut sorted: Vec<f64> = speeds
            .iter()
            .filter_map(|row| row.try_get::<f64>("", "tokens_per_second").ok())
            .collect();
        sorted.sort_by(f64::total_cmp);
        overview.p50_tokens_per_second = percentile(&sorted, 0.5);
        overview.p95_tokens_per_second = percentile(&sorted, 0.95);
        Ok(overview)
    }

    async fn recent(&self, limit: u64) -> Result<Vec<RunStats>, RepositoryError> {
        let rows = run::Entity::find()
            .order_by_desc(run::Column::CreatedAt)
            .limit(limit)
            .all(&self.db)
            .await
            .map_err(map_db_err)?;
        Ok(rows.into_iter().map(RunStats::from).collect())
    }

    async fn daily(&self, days: u32) -> Result<Vec<DayBucket>, RepositoryError> {
        let rows = self
            .db
            .query_all(Statement::from_sql_and_values(
                DbBackend::Sqlite,
                "SELECT date(created_at) AS day, \
                        COUNT(*) AS runs, \
                        COALESCE(SUM(generated_tokens), 0) AS tokens, \
                        COALESCE(AVG(tokens_per_second), 0) AS avg_tps \
                 FROM runs \
                 WHERE date(created_at) >= date('now', ?) \
                 GROUP BY day ORDER BY day",
                [format!("-{} days", days.saturating_sub(1)).into()],
            ))
            .await
            .map_err(map_db_err)?;
        Ok(rows
            .into_iter()
            .map(|row| DayBucket {
                date: row.try_get::<String>("", "day").unwrap_or_default(),
                runs: row.try_get::<i64>("", "runs").unwrap_or(0).max(0) as u64,
                tokens: row.try_get::<i64>("", "tokens").unwrap_or(0).max(0) as u64,
                average_tokens_per_second: row.try_get::<f64>("", "avg_tps").unwrap_or(0.0),
            })
            .collect())
    }
}
