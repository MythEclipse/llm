//! Read-side use cases for the statistics panel.

use crate::application::error::ChatError;
use crate::domain::stats::{DayBucket, Overview, RunRepository, RunStats};

pub struct OverviewUseCase<R> {
    repository: R,
}

impl<R: RunRepository> OverviewUseCase<R> {
    pub fn new(repository: R) -> Self {
        Self { repository }
    }

    pub async fn execute(&self) -> Result<Overview, ChatError> {
        Ok(self.repository.overview().await?)
    }
}

pub struct ListRunsUseCase<R> {
    repository: R,
}

impl<R: RunRepository> ListRunsUseCase<R> {
    pub fn new(repository: R) -> Self {
        Self { repository }
    }

    pub async fn execute(&self, limit: u64) -> Result<Vec<RunStats>, ChatError> {
        let limit = limit.clamp(1, 200);
        Ok(self.repository.recent(limit).await?)
    }
}

pub struct TimeseriesUseCase<R> {
    repository: R,
}

impl<R: RunRepository> TimeseriesUseCase<R> {
    pub fn new(repository: R) -> Self {
        Self { repository }
    }

    pub async fn execute(&self, days: u32) -> Result<Vec<DayBucket>, ChatError> {
        let days = days.clamp(1, 365);
        Ok(self.repository.daily(days).await?)
    }
}
