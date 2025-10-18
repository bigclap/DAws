//! Graph snapshot persistence helpers for `.gbin` archives.

use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use bincode::{deserialize, serialize};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{AssemblyError, Connection, InhibitoryPoolConfig, Network, Node};

/// Errors that may occur while reading or writing graph snapshots.
#[derive(Debug, Error)]
pub enum GraphSnapshotError {
    /// Raised when filesystem access fails.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// Raised when encoding or decoding the snapshot payload fails.
    #[error("bincode error: {0}")]
    Bincode(#[from] Box<bincode::ErrorKind>),
    /// Raised when the captured configuration cannot be applied to a network.
    #[error("failed to apply snapshot: {0}")]
    Assembly(#[from] AssemblyError),
}

/// Serializable representation of a [`Network`].
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GraphSnapshot {
    nodes: Vec<Node>,
    connections: Vec<Connection>,
    input_nodes: Vec<usize>,
    inhibitory_pools: Vec<InhibitoryPoolConfig>,
}

impl GraphSnapshot {
    /// Captures the full state of a [`Network`].
    pub fn capture(network: &Network) -> Self {
        Self {
            nodes: network.nodes.clone(),
            connections: network.connections.clone(),
            input_nodes: network.input_nodes().to_vec(),
            inhibitory_pools: network.inhibitory_pool_configs(),
        }
    }

    /// Reconstructs a [`Network`] from the snapshot contents.
    pub fn to_network(&self) -> Result<Network, AssemblyError> {
        let mut network = Network::new(
            self.nodes.clone(),
            self.connections.clone(),
            self.input_nodes.clone(),
        );
        network.configure_inhibitory_pools(self.inhibitory_pools.clone())?;
        Ok(network)
    }

    /// Writes the snapshot to disk using the `.gbin` convention.
    pub fn write<P: AsRef<Path>>(&self, path: P) -> Result<(), GraphSnapshotError> {
        let path = path.as_ref();
        let payload = serialize(self)?;
        let mut file = File::create(path)?;
        file.write_all(&payload)?;
        Ok(())
    }

    /// Reads a snapshot from the supplied path.
    pub fn read<P: AsRef<Path>>(path: P) -> Result<Self, GraphSnapshotError> {
        let path = path.as_ref();
        let mut bytes = Vec::new();
        File::open(path)?.read_to_end(&mut bytes)?;
        let snapshot = deserialize(&bytes)?;
        Ok(snapshot)
    }

    /// Captures and writes the supplied [`Network`] to disk.
    pub fn write_network<P: AsRef<Path>>(
        path: P,
        network: &Network,
    ) -> Result<(), GraphSnapshotError> {
        Self::capture(network).write(path)
    }

    /// Reads a snapshot and instantiates a [`Network`].
    pub fn read_network<P: AsRef<Path>>(path: P) -> Result<Network, GraphSnapshotError> {
        Ok(Self::read(path)?.to_network()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembly::{ConnectionParams, GraphBuilder, NodeParams};
    use crate::{EpisodeResetPolicy, NodeType};
    use tempfile::NamedTempFile;

    fn build_sample_network() -> Network {
        let mut builder = GraphBuilder::new();
        let input = builder.add_input_node(NodeParams::default());
        let hidden = builder.add_node(NodeParams {
            node_type: NodeType::Modulatory,
            base_threshold: 0.3,
            lambda_v: 0.9,
            lambda_h: 0.95,
            activation_decay: 0.9,
            divisive_beta: 0.1,
            kappa: 0.05,
            modulation_decay: 0.97,
            energy_cap: 5.0,
            episode_period: Some(5),
            reset_policy: EpisodeResetPolicy::Soft,
        });
        builder.add_connection(ConnectionParams::new(
            input, hidden, 0.5, 1.0, 2, 0.01, 0.02, 5.0, 6.0,
        ));
        builder.add_inhibitory_pool(crate::pools::InhibitoryPoolConfig::new(
            vec![input, hidden],
            0.2,
        ));
        let mut network = builder.build().unwrap();
        network.nodes[0].potential = 0.42;
        network.connections[0].eligibility = 0.7;
        network
    }

    #[test]
    fn snapshot_round_trip_via_disk() {
        let network = build_sample_network();
        let snapshot = GraphSnapshot::capture(&network);
        let file = NamedTempFile::new().unwrap();
        snapshot.write(file.path()).unwrap();
        let restored_snapshot = GraphSnapshot::read(file.path()).unwrap();
        assert_eq!(snapshot, restored_snapshot);
        let restored_network = restored_snapshot.to_network().unwrap();
        assert_eq!(restored_network.nodes[0].potential, 0.42);
        assert_eq!(restored_network.connections[0].eligibility, 0.7);
        assert_eq!(restored_network.input_nodes(), network.input_nodes());
    }

    #[test]
    fn snapshot_helper_convenience_methods_work() {
        let network = build_sample_network();
        let file = NamedTempFile::new().unwrap();
        GraphSnapshot::write_network(file.path(), &network).unwrap();
        let restored = GraphSnapshot::read_network(file.path()).unwrap();
        assert_eq!(restored.nodes.len(), network.nodes.len());
        assert_eq!(restored.connections.len(), network.connections.len());
    }
}
