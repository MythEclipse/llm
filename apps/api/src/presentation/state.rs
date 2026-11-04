//! Everything a handler needs, in concrete types only — no trait objects.

use std::collections::HashMap;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use uuid::Uuid;

use crate::infrastructure::{LocalChatEngine, SeaOrmConversationRepository, SeaOrmRunRepository};

/// Handlers share one state; the engine parameter is generic so tests can run
/// the same router against a scripted engine instead of the real model.
pub struct AppState<E = LocalChatEngine> {
    pub conversations: SeaOrmConversationRepository,
    pub runs: SeaOrmRunRepository,
    pub engine: Arc<E>,
    pub model_name: String,
    cancels: Arc<Mutex<HashMap<Uuid, Arc<AtomicBool>>>>,
}

impl<E> Clone for AppState<E> {
    fn clone(&self) -> Self {
        Self {
            conversations: self.conversations.clone(),
            runs: self.runs.clone(),
            engine: Arc::clone(&self.engine),
            model_name: self.model_name.clone(),
            cancels: Arc::clone(&self.cancels),
        }
    }
}

impl<E> AppState<E> {
    pub fn new(
        conversations: SeaOrmConversationRepository,
        runs: SeaOrmRunRepository,
        engine: Arc<E>,
        model_name: String,
    ) -> Self {
        Self {
            conversations,
            runs,
            engine,
            model_name,
            cancels: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Remember how to stop the run currently streaming for this conversation.
    pub fn register_cancel(&self, conversation_id: Uuid, flag: Arc<AtomicBool>) {
        if let Ok(mut cancels) = self.cancels.lock() {
            cancels.insert(conversation_id, flag);
        }
    }

    pub fn clear_cancel(&self, conversation_id: Uuid) {
        if let Ok(mut cancels) = self.cancels.lock() {
            cancels.remove(&conversation_id);
        }
    }

    /// Signal the running generation to stop; returns false when nothing runs.
    pub fn cancel(&self, conversation_id: Uuid) -> bool {
        let Ok(cancels) = self.cancels.lock() else {
            return false;
        };
        match cancels.get(&conversation_id) {
            Some(flag) => {
                flag.store(true, std::sync::atomic::Ordering::Relaxed);
                true
            }
            None => false,
        }
    }
}
