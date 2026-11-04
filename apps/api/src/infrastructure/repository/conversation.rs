//! Conversation persistence.

use chrono::Utc;
use sea_orm::{
    ActiveModelTrait, ColumnTrait, DatabaseConnection, DbErr, EntityTrait, QueryFilter, QueryOrder,
    Set,
};
use uuid::Uuid;

use crate::domain::conversation::{
    Conversation, ConversationRepository, Message, NewConversation, NewMessage, Role,
};
use crate::domain::error::RepositoryError;
use crate::infrastructure::repository::entities::{conversation, message};

#[derive(Clone)]
pub struct SeaOrmConversationRepository {
    db: DatabaseConnection,
}

impl SeaOrmConversationRepository {
    pub fn new(db: DatabaseConnection) -> Self {
        Self { db }
    }
}

impl From<conversation::Model> for Conversation {
    fn from(model: conversation::Model) -> Self {
        Self {
            id: parse_id(&model.id),
            title: model.title,
            created_at: model.created_at.with_timezone(&Utc),
            updated_at: model.updated_at.with_timezone(&Utc),
        }
    }
}

impl From<message::Model> for Message {
    fn from(model: message::Model) -> Self {
        Self {
            id: parse_id(&model.id),
            conversation_id: parse_id(&model.conversation_id),
            role: Role::parse(&model.role).unwrap_or(Role::User),
            content: model.content,
            token_count: model.token_count.max(0) as u32,
            created_at: model.created_at.with_timezone(&Utc),
        }
    }
}

fn parse_id(raw: &str) -> Uuid {
    Uuid::parse_str(raw).unwrap_or_default()
}

fn map_db_err(error: DbErr) -> RepositoryError {
    match error {
        DbErr::RecordNotFound(_) => RepositoryError::NotFound,
        other => {
            let text = other.to_string();
            if text.contains("UNIQUE") || text.contains("unique") {
                RepositoryError::Conflict(text)
            } else {
                tracing::error!(error = %other, "database operation failed");
                RepositoryError::Database(text)
            }
        }
    }
}

impl ConversationRepository for SeaOrmConversationRepository {
    async fn create(&self, new: NewConversation) -> Result<Conversation, RepositoryError> {
        let now = Utc::now().into();
        let model = conversation::ActiveModel {
            id: Set(new.id.to_string()),
            title: Set(new.title),
            created_at: Set(now),
            updated_at: Set(now),
        };
        let inserted = model.insert(&self.db).await.map_err(map_db_err)?;
        Ok(Conversation::from(inserted))
    }

    async fn list(&self) -> Result<Vec<Conversation>, RepositoryError> {
        let rows = conversation::Entity::find()
            .order_by_desc(conversation::Column::UpdatedAt)
            .all(&self.db)
            .await
            .map_err(map_db_err)?;
        Ok(rows.into_iter().map(Conversation::from).collect())
    }

    async fn find(&self, id: Uuid) -> Result<Option<Conversation>, RepositoryError> {
        let row = conversation::Entity::find_by_id(id.to_string())
            .one(&self.db)
            .await
            .map_err(map_db_err)?;
        Ok(row.map(Conversation::from))
    }

    async fn messages(&self, conversation_id: Uuid) -> Result<Vec<Message>, RepositoryError> {
        let rows = message::Entity::find()
            .filter(message::Column::ConversationId.eq(conversation_id.to_string()))
            .order_by_asc(message::Column::CreatedAt)
            .all(&self.db)
            .await
            .map_err(map_db_err)?;
        Ok(rows.into_iter().map(Message::from).collect())
    }

    async fn append_message(&self, new: NewMessage) -> Result<Message, RepositoryError> {
        let model = message::ActiveModel {
            id: Set(new.id.to_string()),
            conversation_id: Set(new.conversation_id.to_string()),
            role: Set(new.role.as_str().to_string()),
            content: Set(new.content),
            token_count: Set(new.token_count as i64),
            created_at: Set(Utc::now().into()),
        };
        let inserted = model.insert(&self.db).await.map_err(map_db_err)?;
        Ok(Message::from(inserted))
    }

    async fn rename(&self, id: Uuid, title: String) -> Result<(), RepositoryError> {
        let update = conversation::ActiveModel {
            id: Set(id.to_string()),
            title: Set(title),
            updated_at: Set(Utc::now().into()),
            ..Default::default()
        };
        update.update(&self.db).await.map_err(map_db_err)?;
        Ok(())
    }

    async fn touch(&self, id: Uuid) -> Result<(), RepositoryError> {
        let update = conversation::ActiveModel {
            id: Set(id.to_string()),
            updated_at: Set(Utc::now().into()),
            ..Default::default()
        };
        update.update(&self.db).await.map_err(map_db_err)?;
        Ok(())
    }

    async fn delete(&self, id: Uuid) -> Result<(), RepositoryError> {
        let result = conversation::Entity::delete_by_id(id.to_string())
            .exec(&self.db)
            .await
            .map_err(map_db_err)?;
        if result.rows_affected == 0 {
            return Err(RepositoryError::NotFound);
        }
        Ok(())
    }
}
