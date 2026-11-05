//! SeaORM models. Identifiers are stored as text because SQLite has no uuid
//! type and the column is only ever compared, never ordered.

pub mod conversation {
    use sea_orm::entity::prelude::*;

    #[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
    #[sea_orm(table_name = "conversations")]
    pub struct Model {
        #[sea_orm(primary_key, auto_increment = false)]
        pub id: String,
        pub title: String,
        pub created_at: DateTimeWithTimeZone,
        pub updated_at: DateTimeWithTimeZone,
    }

    #[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
    pub enum Relation {
        #[sea_orm(has_many = "super::message::Entity")]
        Messages,
    }

    impl Related<super::message::Entity> for Entity {
        fn to() -> RelationDef {
            Relation::Messages.def()
        }
    }

    impl ActiveModelBehavior for ActiveModel {}
}

pub mod message {
    use sea_orm::entity::prelude::*;

    #[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
    #[sea_orm(table_name = "messages")]
    pub struct Model {
        #[sea_orm(primary_key, auto_increment = false)]
        pub id: String,
        pub conversation_id: String,
        pub role: String,
        pub content: String,
        pub token_count: i64,
        pub created_at: DateTimeWithTimeZone,
    }

    #[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
    pub enum Relation {
        #[sea_orm(
            belongs_to = "super::conversation::Entity",
            from = "Column::ConversationId",
            to = "super::conversation::Column::Id"
        )]
        Conversation,
    }

    impl Related<super::conversation::Entity> for Entity {
        fn to() -> RelationDef {
            Relation::Conversation.def()
        }
    }

    impl ActiveModelBehavior for ActiveModel {}
}

pub mod run {
    use sea_orm::entity::prelude::*;

    #[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
    #[sea_orm(table_name = "runs")]
    pub struct Model {
        #[sea_orm(primary_key, auto_increment = false)]
        pub id: String,
        pub conversation_id: String,
        pub model: String,
        pub prompt_tokens: i64,
        pub generated_tokens: i64,
        pub prefill_ms: i64,
        pub ttft_ms: i64,
        pub decode_ms: i64,
        pub total_ms: i64,
        pub tokens_per_second: f64,
        pub ctx_used: i64,
        pub stop_reason: String,
        pub sampler: String,
        pub token_latencies_ms: String,
        pub peak_rss_bytes: i64,
        pub build_hash: Option<String>,
        pub created_at: DateTimeWithTimeZone,
    }

    #[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
    pub enum Relation {
        #[sea_orm(
            belongs_to = "super::conversation::Entity",
            from = "Column::ConversationId",
            to = "super::conversation::Column::Id"
        )]
        Conversation,
    }

    impl ActiveModelBehavior for ActiveModel {}
}
