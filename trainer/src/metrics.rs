use std::collections::HashSet;

pub fn median_cosine_similarity(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) * 0.5
    } else {
        sorted[mid]
    }
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

pub fn distinct_n(tokens: &[String], max_n: usize) -> f32 {
    let mut total = 0usize;
    let mut distinct = 0usize;

    for n in 1..=max_n {
        if tokens.len() < n {
            continue;
        }

        let mut seen = HashSet::new();
        for window in tokens.windows(n) {
            seen.insert(window.to_vec());
        }
        distinct += seen.len();
        total += tokens.len() - n + 1;
    }

    if total == 0 {
        0.0
    } else {
        distinct as f32 / total as f32
    }
}
