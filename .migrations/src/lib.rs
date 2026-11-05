//! Schema migrations for the local chat database.

pub use sea_orm_migration::prelude::*;

mod m20261104_000001_init;
mod m20261105_000002_run_build_identity;

pub struct Migrator;

#[async_trait::async_trait]
impl MigratorTrait for Migrator {
    fn migrations() -> Vec<Box<dyn MigrationTrait>> {
        vec![
            Box::new(m20261104_000001_init::Migration),
            Box::new(m20261105_000002_run_build_identity::Migration),
        ]
    }
}
