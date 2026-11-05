//! Records which build produced each run.
//!
//! The hash is the artifact identity from `BuildManifest`: source, PTX, arch and
//! kernel names. A row with a different hash was produced by different machine
//! code, so two numbers in the dashboard are only comparable when the hashes
//! match.

use sea_orm_migration::prelude::*;

pub struct Migration;

impl MigrationName for Migration {
    fn name(&self) -> &str {
        "m20261105_000002_run_build_identity"
    }
}

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .alter_table(
                Table::alter()
                    .table(Runs::Table)
                    .add_column(ColumnDef::new(Runs::BuildHash).text().null())
                    .to_owned(),
            )
            .await
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .alter_table(
                Table::alter()
                    .table(Runs::Table)
                    .drop_column(Runs::BuildHash)
                    .to_owned(),
            )
            .await
    }
}

#[derive(DeriveIden)]
enum Runs {
    Table,
    BuildHash,
}
