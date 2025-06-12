// @generated automatically by Diesel CLI.

diesel::table! {
    use diesel::sql_types::*;
    use pgvector::sql_types::*;

    posper (id) {
        id -> Int4,
        embeddings -> Vector,
        position -> Varchar,
    }
}
