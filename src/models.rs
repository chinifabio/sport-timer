use std::time::SystemTime;

use diesel::prelude::*;
use pgvector::Vector;
use serde::{Deserialize, Serialize};

#[derive(Clone, Insertable, Queryable, Selectable, Serialize, Deserialize)]
#[diesel(table_name = crate::schema::posper)]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub struct PersonPosition {
    pub embeddings: Vector,
    pub position: String,
    pub timestamp: SystemTime,
}

impl std::fmt::Debug for PersonPosition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PersonPosition")
            .field("embeddings", &"<...>")
            .field("position", &self.position)
            .field("timestamp", &self.timestamp)
            .finish()
    }
}
