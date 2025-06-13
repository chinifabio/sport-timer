use std::time::SystemTime;

use diesel::prelude::*;
use pgvector::Vector;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Insertable, Queryable, Selectable, Serialize, Deserialize)]
#[diesel(table_name = crate::schema::posper)]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub struct PersonPosition {
    pub embeddings: Vector,
    pub position: String,
    pub timestamp: SystemTime,
}
