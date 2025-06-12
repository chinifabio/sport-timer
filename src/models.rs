use diesel::prelude::*;
use pgvector::Vector;

#[derive(Debug, Clone, Insertable, Queryable, Selectable)]
#[diesel(table_name = crate::schema::posper)]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub struct PersonPosition {
    pub embeddings: Vector,
    pub position: String,
}
