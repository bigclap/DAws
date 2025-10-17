//! XOR reasoning demo wiring shared between the CLI and integration tests.

use std::collections::HashMap;

use core_graph::{Connection, Network, Node, NodeType};
use model_dec::BinaryDecoder;
use model_enc::TableEncoder;

/// Builds a compact XOR reasoning graph used by integration tests and the CLI.
pub fn build_xor_network() -> (Network, TableEncoder, BinaryDecoder, usize) {
    let mut nodes = Vec::new();
    nodes.push(Node::new(
        0,
        NodeType::Excitatory,
        0.4,
        0.95,
        0.9,
        0.3,
        0.0,
        0.1,
        0.0,
    ));
    nodes.push(Node::new(
        1,
        NodeType::Excitatory,
        0.4,
        0.95,
        0.9,
        0.3,
        0.0,
        0.1,
        0.0,
    ));
    nodes.push(Node::new(
        2,
        NodeType::Excitatory,
        0.7,
        0.98,
        0.9,
        0.7,
        0.0,
        0.05,
        0.0,
    ));
    nodes.push(Node::new(
        3,
        NodeType::Modulatory,
        1.1,
        0.98,
        0.9,
        0.6,
        0.0,
        0.05,
        0.0,
    ));

    let mut connections = Vec::new();
    connections.push(Connection::new(0, 2, 1.0, 3.0, 0, 1.0, 1.0, 5.0, 5.0));
    connections.push(Connection::new(1, 2, 1.0, 3.0, 0, 1.0, 1.0, 5.0, 5.0));
    connections.push(Connection::new(0, 3, 0.7, 2.0, 0, 1.0, 1.0, 5.0, 5.0));
    connections.push(Connection::new(1, 3, 0.7, 2.0, 0, 1.0, 1.0, 5.0, 5.0));
    connections.push(Connection::new(3, 2, 2.5, 3.0, 0, 1.0, 1.0, 5.0, 5.0));

    let input_nodes = vec![0, 1];
    let network = Network::new(nodes, connections, input_nodes);

    let mut table = HashMap::new();
    table.insert("0 0".to_string(), vec![0.0, 0.0]);
    table.insert("0 1".to_string(), vec![0.0, 1.0]);
    table.insert("1 0".to_string(), vec![1.0, 0.0]);
    table.insert("1 1".to_string(), vec![1.0, 1.0]);
    let encoder = TableEncoder::new(table);
    let decoder = BinaryDecoder::new(0.5);
    (network, encoder, decoder, 2)
}
