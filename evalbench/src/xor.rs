//! XOR reasoning demo wiring shared between the CLI and integration tests.

use std::collections::HashMap;

use core_graph::{ConnectionParams, GraphBuilder, Network, NodeParams, NodeType};
use model_dec::BinaryDecoder;
use model_enc::TableEncoder;

/// Builds a compact XOR reasoning graph used by integration tests and the CLI.
pub fn build_xor_network() -> (Network, TableEncoder, BinaryDecoder, usize) {
    let mut builder = GraphBuilder::new();
    let input_a = builder.add_input_node(NodeParams::new(
        NodeType::Excitatory,
        0.4,
        0.95,
        0.9,
        0.3,
        0.0,
        0.1,
        0.0,
    ));
    let input_b = builder.add_input_node(NodeParams::new(
        NodeType::Excitatory,
        0.4,
        0.95,
        0.9,
        0.3,
        0.0,
        0.1,
        0.0,
    ));
    let xor_node = builder.add_node(NodeParams::new(
        NodeType::Excitatory,
        0.7,
        0.98,
        0.9,
        0.7,
        0.0,
        0.05,
        0.0,
    ));
    let gating_node = builder.add_node(NodeParams::new(
        NodeType::Modulatory,
        1.1,
        0.98,
        0.9,
        0.6,
        0.0,
        0.05,
        0.0,
    ));

    builder.add_connection(ConnectionParams::new(
        input_a, xor_node, 1.0, 3.0, 0, 1.0, 1.0, 5.0, 5.0,
    ));
    builder.add_connection(ConnectionParams::new(
        input_b, xor_node, 1.0, 3.0, 0, 1.0, 1.0, 5.0, 5.0,
    ));
    builder.add_connection(ConnectionParams::new(
        input_a,
        gating_node,
        0.7,
        2.0,
        0,
        1.0,
        1.0,
        5.0,
        5.0,
    ));
    builder.add_connection(ConnectionParams::new(
        input_b,
        gating_node,
        0.7,
        2.0,
        0,
        1.0,
        1.0,
        5.0,
        5.0,
    ));
    builder.add_connection(ConnectionParams::new(
        gating_node,
        xor_node,
        2.5,
        3.0,
        0,
        1.0,
        1.0,
        5.0,
        5.0,
    ));

    let network = builder.build().expect("xor network assembly");

    let mut table = HashMap::new();
    table.insert("0 0".to_string(), vec![0.0, 0.0]);
    table.insert("0 1".to_string(), vec![0.0, 1.0]);
    table.insert("1 0".to_string(), vec![1.0, 0.0]);
    table.insert("1 1".to_string(), vec![1.0, 1.0]);
    let encoder = TableEncoder::new(table);
    let decoder = BinaryDecoder::new(0.5);
    (network, encoder, decoder, xor_node)
}
