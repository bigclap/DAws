//! Helpers for declarative network assembly used by demos and tests.

use crate::{Connection, EpisodeResetPolicy, Network, Node, NodeType, pools::InhibitoryPoolConfig};

/// Parameters describing the static configuration of a [`Node`].
#[derive(Clone, Debug)]
pub struct NodeParams {
    /// Functional category assigned to the node.
    pub node_type: NodeType,
    /// Baseline firing threshold before modulation/adaptation.
    pub base_threshold: f32,
    /// Leak coefficient applied to the membrane potential each step.
    pub lambda_v: f32,
    /// Adaptation decay applied per step.
    pub lambda_h: f32,
    /// Exponential decay applied to the activation trace.
    pub activation_decay: f32,
    /// Divisive normalisation factor for inhibitory input.
    pub divisive_beta: f32,
    /// Adaptation increment applied on spikes.
    pub kappa: f32,
    /// Modulation decay applied per step.
    pub modulation_decay: f32,
    /// Maximum energy allowed for the node's tracked states.
    pub energy_cap: f32,
    /// Optional number of steps between episodic resets.
    pub episode_period: Option<usize>,
    /// Policy executed when the episode boundary is reached.
    pub reset_policy: EpisodeResetPolicy,
}

impl NodeParams {
    /// Convenience constructor mirroring [`Node::new`] but without the identifier.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        node_type: NodeType,
        base_threshold: f32,
        lambda_v: f32,
        lambda_h: f32,
        activation_decay: f32,
        divisive_beta: f32,
        kappa: f32,
        modulation_decay: f32,
        energy_cap: f32,
        episode_period: Option<usize>,
        reset_policy: EpisodeResetPolicy,
    ) -> Self {
        Self {
            node_type,
            base_threshold,
            lambda_v,
            lambda_h,
            activation_decay,
            divisive_beta,
            kappa,
            modulation_decay,
            energy_cap,
            episode_period,
            reset_policy,
        }
    }
}

impl Default for NodeParams {
    fn default() -> Self {
        Self {
            node_type: NodeType::Excitatory,
            base_threshold: 0.5,
            lambda_v: 0.95,
            lambda_h: 0.98,
            activation_decay: 0.95,
            divisive_beta: 0.0,
            kappa: 0.0,
            modulation_decay: 0.95,
            energy_cap: 10.0,
            episode_period: None,
            reset_policy: EpisodeResetPolicy::None,
        }
    }
}

/// Parameters used when instantiating a [`Connection`].
#[derive(Clone, Debug)]
pub struct ConnectionParams {
    /// Identifier of the source node.
    pub from: usize,
    /// Identifier of the destination node.
    pub to: usize,
    /// Initial synaptic weight.
    pub weight: f32,
    /// Maximum admissible synaptic weight.
    pub max_weight: f32,
    /// Propagation delay in discrete steps.
    pub delay: usize,
    /// STDP potentiation amplitude.
    pub a_plus: f32,
    /// STDP depression amplitude.
    pub a_minus: f32,
    /// STDP potentiation decay constant.
    pub tau_plus: f32,
    /// STDP depression decay constant.
    pub tau_minus: f32,
}

impl ConnectionParams {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        from: usize,
        to: usize,
        weight: f32,
        max_weight: f32,
        delay: usize,
        a_plus: f32,
        a_minus: f32,
        tau_plus: f32,
        tau_minus: f32,
    ) -> Self {
        Self {
            from,
            to,
            weight,
            max_weight,
            delay,
            a_plus,
            a_minus,
            tau_plus,
            tau_minus,
        }
    }
}

/// Errors that can arise while assembling a network.
#[derive(Debug, PartialEq, Eq)]
pub enum AssemblyError {
    /// A connection referenced a node identifier that was never added.
    MissingNode { node_id: usize },
    /// An inhibitory pool referenced a node identifier that was never added.
    MissingPoolNode { node_id: usize },
    /// Regional detectors must have unique labels.
    DuplicateDetectorLabel { label: String },
}

impl core::fmt::Display for AssemblyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            AssemblyError::MissingNode { node_id } => {
                write!(f, "connection references missing node {node_id}")
            }
            AssemblyError::MissingPoolNode { node_id } => {
                write!(f, "pool references missing node {node_id}")
            }
            AssemblyError::DuplicateDetectorLabel { label } => {
                write!(f, "detector label '{label}' was registered more than once")
            }
        }
    }
}

impl std::error::Error for AssemblyError {}

/// Incremental builder that records nodes, connections, and input bindings.
#[derive(Default, Debug)]
pub struct GraphBuilder {
    nodes: Vec<NodeParams>,
    connections: Vec<ConnectionParams>,
    input_nodes: Vec<usize>,
    inhibitory_pools: Vec<InhibitoryPoolConfig>,
}

impl GraphBuilder {
    /// Creates an empty builder ready to accept nodes and connections.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a node and returns its automatically assigned identifier.
    pub fn add_node(&mut self, params: NodeParams) -> usize {
        let id = self.nodes.len();
        self.nodes.push(params);
        id
    }

    /// Adds a node that participates in embedding injection.
    pub fn add_input_node(&mut self, params: NodeParams) -> usize {
        let id = self.add_node(params);
        self.input_nodes.push(id);
        id
    }

    /// Registers an inhibitory pool affecting the supplied members.
    pub fn add_inhibitory_pool(&mut self, pool: InhibitoryPoolConfig) {
        self.inhibitory_pools.push(pool);
    }

    /// Registers a directed connection between previously added nodes.
    pub fn add_connection(&mut self, params: ConnectionParams) {
        self.connections.push(params);
    }

    /// Finalises the builder, producing a [`Network`] instance.
    pub fn build(self) -> Result<Network, AssemblyError> {
        let GraphBuilder {
            nodes,
            connections,
            input_nodes,
            inhibitory_pools,
        } = self;

        let node_count = nodes.len();
        for conn in &connections {
            if conn.from >= node_count {
                return Err(AssemblyError::MissingNode { node_id: conn.from });
            }
            if conn.to >= node_count {
                return Err(AssemblyError::MissingNode { node_id: conn.to });
            }
        }

        for pool in &inhibitory_pools {
            for &member in &pool.members {
                if member >= node_count {
                    return Err(AssemblyError::MissingPoolNode { node_id: member });
                }
            }
        }

        let nodes: Vec<Node> = nodes
            .into_iter()
            .enumerate()
            .map(|(id, params)| {
                Node::new(
                    id,
                    params.node_type,
                    params.base_threshold,
                    params.lambda_v,
                    params.lambda_h,
                    params.activation_decay,
                    params.divisive_beta,
                    params.kappa,
                    params.modulation_decay,
                    params.energy_cap,
                    params.episode_period,
                    params.reset_policy,
                )
            })
            .collect();

        let connections: Vec<Connection> = connections
            .into_iter()
            .map(|params| {
                Connection::new(
                    params.from,
                    params.to,
                    params.weight,
                    params.max_weight,
                    params.delay,
                    params.a_plus,
                    params.a_minus,
                    params.tau_plus,
                    params.tau_minus,
                )
            })
            .collect();

        let mut network = Network::new(nodes, connections, input_nodes);
        network.configure_inhibitory_pools(inhibitory_pools)?;
        Ok(network)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_assigns_identifiers_and_builds_network() {
        let mut builder = GraphBuilder::new();
        let input = builder.add_input_node(NodeParams::default());
        let output = builder.add_node(NodeParams {
            node_type: NodeType::Modulatory,
            base_threshold: 0.3,
            lambda_v: 0.9,
            lambda_h: 0.9,
            activation_decay: 0.8,
            divisive_beta: 0.1,
            kappa: 0.05,
            modulation_decay: 0.9,
            ..NodeParams::default()
        });
        builder.add_connection(ConnectionParams::new(
            input, output, 1.0, 1.5, 0, 1.0, 1.0, 5.0, 5.0,
        ));

        let network = builder.build().expect("valid network");
        assert_eq!(network.input_nodes(), &[0]);
        assert_eq!(network.nodes.len(), 2);
        assert_eq!(network.connections.len(), 1);
        assert_eq!(network.connections[0].from, input);
        assert_eq!(network.connections[0].to, output);
    }

    #[test]
    fn builder_validates_connection_endpoints() {
        let mut builder = GraphBuilder::new();
        let node = builder.add_node(NodeParams::default());
        builder.add_connection(ConnectionParams::new(
            node + 1,
            node,
            0.5,
            1.0,
            0,
            1.0,
            1.0,
            5.0,
            5.0,
        ));

        let error = builder.build().err().expect("missing node error");
        assert_eq!(error, AssemblyError::MissingNode { node_id: node + 1 });
    }
}
