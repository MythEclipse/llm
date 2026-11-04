use sea_orm_migration::prelude::*;

pub struct Migration;

impl MigrationName for Migration {
    fn name(&self) -> &str {
        "m20261104_000001_init"
    }
}

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .create_table(
                Table::create()
                    .table(Conversations::Table)
                    .if_not_exists()
                    .col(
                        ColumnDef::new(Conversations::Id)
                            .text()
                            .not_null()
                            .primary_key(),
                    )
                    .col(ColumnDef::new(Conversations::Title).text().not_null())
                    .col(
                        ColumnDef::new(Conversations::CreatedAt)
                            .timestamp_with_time_zone()
                            .not_null(),
                    )
                    .col(
                        ColumnDef::new(Conversations::UpdatedAt)
                            .timestamp_with_time_zone()
                            .not_null(),
                    )
                    .to_owned(),
            )
            .await?;

        manager
            .create_table(
                Table::create()
                    .table(Messages::Table)
                    .if_not_exists()
                    .col(ColumnDef::new(Messages::Id).text().not_null().primary_key())
                    .col(ColumnDef::new(Messages::ConversationId).text().not_null())
                    .col(ColumnDef::new(Messages::Role).text().not_null())
                    .col(ColumnDef::new(Messages::Content).text().not_null())
                    .col(
                        ColumnDef::new(Messages::TokenCount)
                            .big_integer()
                            .not_null()
                            .default(0),
                    )
                    .col(
                        ColumnDef::new(Messages::CreatedAt)
                            .timestamp_with_time_zone()
                            .not_null(),
                    )
                    .foreign_key(
                        ForeignKey::create()
                            .name("fk_messages_conversation")
                            .from(Messages::Table, Messages::ConversationId)
                            .to(Conversations::Table, Conversations::Id)
                            .on_delete(ForeignKeyAction::Cascade),
                    )
                    .to_owned(),
            )
            .await?;

        manager
            .create_index(
                Index::create()
                    .name("idx_messages_conversation")
                    .table(Messages::Table)
                    .col(Messages::ConversationId)
                    .col(Messages::CreatedAt)
                    .to_owned(),
            )
            .await?;

        manager
            .create_table(
                Table::create()
                    .table(Runs::Table)
                    .if_not_exists()
                    .col(ColumnDef::new(Runs::Id).text().not_null().primary_key())
                    .col(ColumnDef::new(Runs::ConversationId).text().not_null())
                    .col(ColumnDef::new(Runs::Model).text().not_null())
                    .col(
                        ColumnDef::new(Runs::PromptTokens)
                            .big_integer()
                            .not_null()
                            .default(0),
                    )
                    .col(
                        ColumnDef::new(Runs::GeneratedTokens)
                            .big_integer()
                            .not_null()
                            .default(0),
                    )
                    .col(
                        ColumnDef::new(Runs::PrefillMs)
                            .big_integer()
                            .not_null()
                            .default(0),
                    )
                    .col(
                        ColumnDef::new(Runs::TtftMs)
                            .big_integer()
                            .not_null()
                            .default(0),
                    )
                    .col(
                        ColumnDef::new(Runs::DecodeMs)
                            .big_integer()
                            .not_null()
                            .default(0),
                    )
                    .col(
                        ColumnDef::new(Runs::TotalMs)
                            .big_integer()
                            .not_null()
                            .default(0),
                    )
                    .col(
                        ColumnDef::new(Runs::TokensPerSecond)
                            .double()
                            .not_null()
                            .default(0.0),
                    )
                    .col(
                        ColumnDef::new(Runs::CtxUsed)
                            .big_integer()
                            .not_null()
                            .default(0),
                    )
                    .col(ColumnDef::new(Runs::StopReason).text().not_null())
                    .col(ColumnDef::new(Runs::Sampler).text().not_null())
                    .col(ColumnDef::new(Runs::TokenLatenciesMs).text().not_null())
                    .col(
                        ColumnDef::new(Runs::PeakRssBytes)
                            .big_integer()
                            .not_null()
                            .default(0),
                    )
                    .col(
                        ColumnDef::new(Runs::CreatedAt)
                            .timestamp_with_time_zone()
                            .not_null(),
                    )
                    .foreign_key(
                        ForeignKey::create()
                            .name("fk_runs_conversation")
                            .from(Runs::Table, Runs::ConversationId)
                            .to(Conversations::Table, Conversations::Id)
                            .on_delete(ForeignKeyAction::Cascade),
                    )
                    .to_owned(),
            )
            .await?;

        manager
            .create_index(
                Index::create()
                    .name("idx_runs_created_at")
                    .table(Runs::Table)
                    .col(Runs::CreatedAt)
                    .to_owned(),
            )
            .await
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .drop_table(Table::drop().table(Runs::Table).to_owned())
            .await?;
        manager
            .drop_table(Table::drop().table(Messages::Table).to_owned())
            .await?;
        manager
            .drop_table(Table::drop().table(Conversations::Table).to_owned())
            .await
    }
}

#[derive(DeriveIden)]
enum Conversations {
    Table,
    Id,
    Title,
    CreatedAt,
    UpdatedAt,
}

#[derive(DeriveIden)]
enum Messages {
    Table,
    Id,
    ConversationId,
    Role,
    Content,
    TokenCount,
    CreatedAt,
}

#[derive(DeriveIden)]
enum Runs {
    Table,
    Id,
    ConversationId,
    Model,
    PromptTokens,
    GeneratedTokens,
    PrefillMs,
    TtftMs,
    DecodeMs,
    TotalMs,
    TokensPerSecond,
    CtxUsed,
    StopReason,
    Sampler,
    TokenLatenciesMs,
    PeakRssBytes,
    CreatedAt,
}
