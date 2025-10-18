use super::*;

fn base_config() -> RetrieverConfig {
    RetrieverConfig {
        dimension: 2,
        value_dimension: 2,
        max_elements: 16,
        max_layers: 8,
        max_connections: 4,
        ef_construction: 16,
        ef_search: 16,
        top_k: 3,
    }
}

fn record(key: u64, embedding: [f32; 2]) -> MemoryRecord {
    MemoryRecord::new(key, embedding.to_vec(), embedding.to_vec())
}

#[test]
fn ann_search_returns_ranked_matches() {
    let mut retriever = Retriever::new(base_config()).expect("valid config");
    retriever
        .ingest([
            record(1, [1.0, 0.0]),
            record(2, [0.0, 1.0]),
            record(3, [0.7, 0.7]),
        ])
        .expect("ingest succeeds");

    let hits = retriever
        .search(&[1.0, 0.0], Some(2))
        .expect("query executes");
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].key, 1);
    assert!(hits[0].similarity > hits[1].similarity);
    assert!(hits[0].similarity > 0.9);
}

#[test]
fn snapshot_round_trip_preserves_results() {
    let mut retriever = Retriever::new(base_config()).expect("valid config");
    retriever
        .ingest([record(7, [1.0, 0.0]), record(8, [0.0, 1.0])])
        .expect("ingest succeeds");
    let snapshot = retriever.snapshot().expect("snapshot");

    let restored = Retriever::from_snapshot(base_config(), snapshot).expect("restored");
    let hits = restored.search(&[0.0, 1.0], None).expect("query executes");
    assert_eq!(hits.first().map(|hit| hit.key), Some(8));
    let value = restored.value(8).expect("payload available");
    assert!((value[1] - 1.0).abs() < 1e-6);
}
