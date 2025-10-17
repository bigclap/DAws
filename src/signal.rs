use std::collections::HashMap;

use crate::io::{BinaryDecoder, TableEncoder};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NodeType {
    Excitatory,
    Inhibitory,
    Modulatory,
}

#[derive(Clone, Debug)]
pub struct Node {
    pub id: usize,
    pub node_type: NodeType,
    pub potential: f32,
    pub activation: f32,
    pub adaptation: f32,
    pub base_threshold: f32,
    pub lambda_v: f32,
    pub lambda_h: f32,
    pub activation_decay: f32,
    pub kappa: f32,
    pub modulation: f32,
    pub modulation_decay: f32,
    pub eligibility: f32,
    pub last_spike_step: Option<usize>,
}

impl Node {
    pub fn new(
        id: usize,
        node_type: NodeType,
        base_threshold: f32,
        lambda_v: f32,
        lambda_h: f32,
        activation_decay: f32,
        kappa: f32,
        modulation_decay: f32,
    ) -> Self {
        Self {
            id,
            node_type,
            potential: 0.0,
            activation: 0.0,
            adaptation: 0.0,
            base_threshold,
            lambda_v,
            lambda_h,
            activation_decay,
            kappa,
            modulation: 0.0,
            modulation_decay,
            eligibility: 0.0,
            last_spike_step: None,
        }
    }

    pub fn effective_threshold(&self) -> f32 {
        let modulation_scale = (1.0 + self.modulation).clamp(0.1, 10.0);
        self.base_threshold * modulation_scale + self.adaptation
    }

    pub fn integrate_decay(&mut self) {
        self.potential *= self.lambda_v;
        self.adaptation *= self.lambda_h;
        self.activation *= self.activation_decay;
        self.modulation *= self.modulation_decay;
    }

    pub fn on_spike(&mut self, step: usize) {
        self.potential = 0.0;
        self.activation = 1.0;
        self.adaptation += self.kappa;
        self.last_spike_step = Some(step);
    }

    pub fn reset_state(&mut self) {
        self.potential = 0.0;
        self.activation = 0.0;
        self.adaptation = 0.0;
        self.modulation = 0.0;
        self.last_spike_step = None;
        self.eligibility = 0.0;
    }
}

#[derive(Clone, Debug)]
pub struct Connection {
    pub from: usize,
    pub to: usize,
    pub weight: f32,
    pub max_weight: f32,
    pub delay: usize,
    pub eligibility: f32,
    pub a_plus: f32,
    pub a_minus: f32,
    pub tau_plus: f32,
    pub tau_minus: f32,
    pub last_pre_spike: Option<usize>,
    pub last_post_spike: Option<usize>,
}

impl Connection {
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
            eligibility: 0.0,
            a_plus,
            a_minus,
            tau_plus,
            tau_minus,
            last_pre_spike: None,
            last_post_spike: None,
        }
    }
}

#[derive(Clone, Debug)]
struct ScheduledEvent {
    target: usize,
    delta: f32,
}

#[derive(Default, Debug, Clone)]
pub struct StepReport {
    pub spikes: Vec<usize>,
    pub modulatory_spikes: Vec<usize>,
}

pub struct Network {
    pub nodes: Vec<Node>,
    pub connections: Vec<Connection>,
    outgoing: Vec<Vec<usize>>,
    incoming: Vec<Vec<usize>>,
    event_queue: Vec<Vec<ScheduledEvent>>,
    max_delay: usize,
    input_nodes: Vec<usize>,
}

impl Network {
    pub fn new(nodes: Vec<Node>, connections: Vec<Connection>, input_nodes: Vec<usize>) -> Self {
        let max_delay = connections.iter().map(|c| c.delay).max().unwrap_or(0);
        let mut outgoing = vec![Vec::new(); nodes.len()];
        let mut incoming = vec![Vec::new(); nodes.len()];
        for (idx, conn) in connections.iter().enumerate() {
            outgoing[conn.from].push(idx);
            incoming[conn.to].push(idx);
        }
        Self {
            nodes,
            connections,
            outgoing,
            incoming,
            event_queue: vec![Vec::new(); max_delay.max(1) + 1],
            max_delay: max_delay.max(1),
            input_nodes,
        }
    }

    pub fn reset_state(&mut self) {
        for node in &mut self.nodes {
            node.reset_state();
        }
        for conn in &mut self.connections {
            conn.eligibility = 0.0;
            conn.last_post_spike = None;
            conn.last_pre_spike = None;
        }
        for queue in &mut self.event_queue {
            queue.clear();
        }
    }

    pub fn inject(&mut self, excitations: &[(usize, f32)]) {
        for &(id, value) in excitations {
            if let Some(node) = self.nodes.get_mut(id) {
                node.potential += value;
            }
        }
    }

    pub fn inject_embedding(&mut self, embedding: &[f32]) {
        for (node_id, value) in self.input_nodes.iter().zip(embedding.iter().copied()) {
            if let Some(node) = self.nodes.get_mut(*node_id) {
                node.potential += value;
            }
        }
    }

    pub fn step(&mut self, step: usize) -> StepReport {
        let mut report = StepReport::default();
        self.apply_decay();
        let due_events = self.pop_events();
        for event in due_events {
            if let Some(node) = self.nodes.get_mut(event.target) {
                node.potential += event.delta;
            }
        }

        let modulatory_spikes: Vec<usize> = self
            .nodes
            .iter()
            .filter(|node| node.node_type == NodeType::Modulatory)
            .filter(|node| node.potential > node.effective_threshold())
            .map(|node| node.id)
            .collect();

        for node_id in modulatory_spikes.iter().copied() {
            let outgoing = self.outgoing[node_id].clone();
            for conn_id in outgoing {
                let conn = &self.connections[conn_id];
                let weight = conn.weight;
                if let Some(target) = self.nodes.get_mut(conn.to) {
                    target.modulation += weight;
                    if target.modulation < -0.9 {
                        target.modulation = -0.9;
                    }
                }
                let conn = &mut self.connections[conn_id];
                conn.last_pre_spike = Some(step);
            }
            if let Some(node) = self.nodes.get_mut(node_id) {
                node.on_spike(step);
            }
        }
        report.modulatory_spikes = modulatory_spikes;

        let mut spikes = Vec::new();
        for node in self.nodes.iter() {
            if node.node_type != NodeType::Modulatory && node.potential > node.effective_threshold()
            {
                spikes.push(node.id);
            }
        }

        for node_id in spikes.iter().copied() {
            let node_type = self.nodes[node_id].node_type;
            let outgoing = self.outgoing[node_id].clone();
            for conn_id in outgoing {
                let conn = &self.connections[conn_id];
                let sign = match node_type {
                    NodeType::Inhibitory => -1.0,
                    _ => 1.0,
                };
                let delta = sign * conn.weight;
                self.schedule_event(conn.delay, conn.to, delta);
                let conn = &mut self.connections[conn_id];
                conn.last_pre_spike = Some(step);
            }
            if let Some(node) = self.nodes.get_mut(node_id) {
                node.on_spike(step);
            }
            for &conn_id in &self.incoming[node_id] {
                let conn = &mut self.connections[conn_id];
                conn.last_post_spike = Some(step);
            }
        }
        report.spikes = spikes;

        report
    }

    fn apply_decay(&mut self) {
        for node in &mut self.nodes {
            node.integrate_decay();
        }
    }

    fn pop_events(&mut self) -> Vec<ScheduledEvent> {
        let due = std::mem::take(&mut self.event_queue[0]);
        self.event_queue.rotate_left(1);
        if let Some(last) = self.event_queue.last_mut() {
            last.clear();
        }
        due
    }

    fn schedule_event(&mut self, delay: usize, target: usize, delta: f32) {
        let actual_delay = delay.min(self.max_delay);
        self.event_queue[actual_delay].push(ScheduledEvent { target, delta });
    }

    pub fn active_ratio(&self, activation_threshold: f32) -> f32 {
        if self.nodes.is_empty() {
            return 0.0;
        }
        let active = self
            .nodes
            .iter()
            .filter(|node| node.activation > activation_threshold)
            .count();
        active as f32 / self.nodes.len() as f32
    }

    pub fn apply_reward(&mut self, reward: f32, learning_rate: f32) {
        for conn in &mut self.connections {
            if let (Some(pre), Some(post)) = (conn.last_pre_spike, conn.last_post_spike) {
                let delta_t = post as isize - pre as isize;
                let delta_t_f = delta_t as f32;
                let potentiation = if delta_t_f >= 0.0 {
                    conn.a_plus * (-(delta_t_f) / conn.tau_plus).exp()
                } else {
                    0.0
                };
                let depression = if delta_t_f <= 0.0 {
                    conn.a_minus * (delta_t_f.abs() / conn.tau_minus).exp()
                } else {
                    0.0
                };
                conn.eligibility = potentiation - depression;
                let delta_w = learning_rate * reward * conn.eligibility;
                conn.weight = (conn.weight + delta_w).clamp(0.0, conn.max_weight);
            }
        }
    }

    pub fn connection_weight(&self, connection_id: usize) -> f32 {
        self.connections
            .get(connection_id)
            .map(|conn| conn.weight)
            .unwrap_or(0.0)
    }

    pub fn node(&self, node_id: usize) -> &Node {
        &self.nodes[node_id]
    }

    pub fn node_mut(&mut self, node_id: usize) -> &mut Node {
        &mut self.nodes[node_id]
    }

    pub fn state_vector(&self) -> Vec<f32> {
        self.nodes.iter().map(|node| node.activation).collect()
    }

    pub fn set_state(&mut self, state: &[f32]) {
        for (node, value) in self.nodes.iter_mut().zip(state.iter().copied()) {
            node.activation = value;
        }
    }

    pub fn consensus_state(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.nodes.len());
        for (idx, node) in self.nodes.iter().enumerate() {
            let incoming = &self.incoming[idx];
            if incoming.is_empty() {
                result.push(node.activation);
                continue;
            }
            let mut sum = 0.0;
            let mut total_weight = 0.0;
            for &conn_id in incoming {
                let conn = &self.connections[conn_id];
                let source = &self.nodes[conn.from];
                let strength = conn.weight.abs() * source.activation.abs();
                if strength == 0.0 {
                    continue;
                }
                let sign = match source.node_type {
                    NodeType::Inhibitory | NodeType::Modulatory => -1.0,
                    NodeType::Excitatory => 1.0,
                };
                sum += sign * conn.weight * source.activation;
                total_weight += strength;
            }
            if total_weight == 0.0 {
                result.push(node.activation);
            } else {
                let average = (node.activation + sum) / (1.0 + total_weight);
                result.push(average);
            }
        }
        result
    }

    pub fn energy(&self) -> f32 {
        self.nodes
            .iter()
            .map(|node| node.potential.abs() + node.activation)
            .sum()
    }

    pub fn input_nodes(&self) -> &[usize] {
        &self.input_nodes
    }
}

pub fn build_xor_network() -> (Network, TableEncoder, BinaryDecoder, usize) {
    let mut nodes = Vec::new();
    nodes.push(Node::new(
        0,
        NodeType::Excitatory,
        0.4,
        0.95,
        0.9,
        0.3,
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
        0.05,
        0.0,
    ));

    let mut connections = Vec::new();
    connections.push(Connection::new(0, 2, 1.0, 3.0, 1, 1.0, 1.0, 5.0, 5.0));
    connections.push(Connection::new(1, 2, 1.0, 3.0, 1, 1.0, 1.0, 5.0, 5.0));
    connections.push(Connection::new(0, 3, 0.7, 2.0, 1, 1.0, 1.0, 5.0, 5.0));
    connections.push(Connection::new(1, 3, 0.7, 2.0, 1, 1.0, 1.0, 5.0, 5.0));
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

pub mod test_helpers {
    use super::{Connection, Network, Node, NodeType};

    pub fn two_node_modulatory() -> Network {
        let nodes = vec![
            Node::new(0, NodeType::Modulatory, 0.5, 0.99, 0.9, 0.9, 0.05, 0.0),
            Node::new(1, NodeType::Excitatory, 0.6, 0.99, 0.9, 0.9, 0.05, 0.0),
        ];
        let connections = vec![Connection::new(0, 1, 1.0, 2.0, 0, 1.0, 1.0, 5.0, 5.0)];
        Network::new(nodes, connections, vec![0])
    }

    pub fn simple_pair() -> Network {
        let nodes = vec![
            Node::new(0, NodeType::Excitatory, 0.3, 0.99, 0.9, 0.9, 0.05, 0.0),
            Node::new(1, NodeType::Excitatory, 0.5, 0.99, 0.9, 0.9, 0.05, 0.0),
        ];
        let connections = vec![Connection::new(0, 1, 0.8, 2.0, 1, 1.0, 1.0, 5.0, 5.0)];
        Network::new(nodes, connections, vec![0])
    }
}
