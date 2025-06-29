use crate::models::PersonPosition;

#[derive(Debug, Clone, Default)]
pub struct Tracker {
    people: Vec<PersonPosition>,
}

impl Tracker {
    pub fn update(&mut self, target_pp: PersonPosition) -> (usize, PersonPosition) {
        let res = self.people.iter().enumerate().find(|(_, pp)| {
            let cos_sim = cosine_similarity(&target_pp.embeddings, &pp.embeddings)
                .expect("Failed to compute similarity");
            cos_sim > 0.8
        });
        match res {
            Some((idx, pp)) => {
                let new_embeddings = average_embeddings(&pp.embeddings, &target_pp.embeddings);
                let new_pp = PersonPosition {
                    embeddings: new_embeddings.expect("Failed to compute average of embeddings"),
                    position: pp.position.clone(),
                    timestamp: pp.timestamp,
                };
                self.people[idx] = new_pp.clone();
                (idx, new_pp)
            }
            None => {
                let id = self.people.len();
                self.people.push(target_pp.clone());
                (id, target_pp)
            }
        }
    }
}

pub fn cosine_similarity(vec1: &pgvector::Vector, vec2: &pgvector::Vector) -> Option<f32> {
    let v1_slice = vec1.to_vec();
    let v2_slice = vec2.to_vec();

    if v1_slice.len() != v2_slice.len() {
        return None; // Vectors must have the same dimension
    }

    if v1_slice.is_empty() {
        // If both are empty, they are considered perfectly similar.
        // If one is empty and the other is not, the length check above would have caught it.
        return Some(1.0);
    }

    let dot_product = v1_slice
        .iter()
        .zip(v2_slice.iter())
        .map(|(a, b)| a * b)
        .sum::<f32>();

    let norm_sq_1 = v1_slice.iter().map(|a| a * a).sum::<f32>();
    let norm_sq_2 = v2_slice.iter().map(|a| a * a).sum::<f32>();

    if norm_sq_1 == 0.0 && norm_sq_2 == 0.0 {
        // Both are zero vectors; considered perfectly similar.
        return Some(1.0);
    }

    if norm_sq_1 == 0.0 || norm_sq_2 == 0.0 {
        // One vector is a zero vector, the other is not.
        // Cosine similarity is 0.
        return Some(0.0);
    }

    // Standard cosine similarity formula
    Some(dot_product / (norm_sq_1.sqrt() * norm_sq_2.sqrt()))
}

pub fn average_embeddings(
    emb1: &pgvector::Vector,
    emb2: &pgvector::Vector,
) -> Option<pgvector::Vector> {
    let v1_slice = emb1.to_vec();
    let v2_slice = emb2.to_vec();

    if v1_slice.len() != v2_slice.len() {
        return None; // Vectors must have the same dimension for averaging
    }

    if v1_slice.is_empty() {
        // Average of two empty vectors is an empty vector
        return Some(pgvector::Vector::from(Vec::new()));
    }

    let averaged_values: Vec<f32> = v1_slice
        .iter()
        .zip(v2_slice.iter())
        .map(|(a, b)| (a + b) / 2.0)
        .collect();

    Some(pgvector::Vector::from(averaged_values))
}
