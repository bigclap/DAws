use std::collections::HashMap;

use crate::io::{BinaryDecoder, TableEncoder};

pub const MAX_DELAY: usize = 16;
pub const ALPHA_KERNEL: [f32; 8] = [0.6, 0.9, 0.8, 0.5, 0.3, 0.15, 0.08, 0.04];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NodeType {
    Excitatory,
    Inhibitory,
    Modulatory,
    Memory,
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
    pub divisive_beta: f32,
    pub kappa: f32,
    pub modulation: f32,
    pub modulation_decay: f32,
    pub eligibility: f32,
    pub last_spike_step: Option<usize>,
    pub inhibition_accumulator: f32,
    pub future_exc: [f32; MAX_DELAY],
    pub accum_exc: f32,
    pub accum_inh: f32,
}

impl Node {
    pub fn new(
        id: usize,
        node_type: NodeType,
        base_threshold: f32,
        lambda_v: f32,
        lambda_h: f32,
        activation_decay: f32,
        divisive_beta: f32,
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
            divisive_beta,
            kappa,
            modulation: 0.0,
            modulation_decay,
            eligibility: 0.0,
            last_spike_step: None,
            inhibition_accumulator: 0.0,
            future_exc: [0.0; MAX_DELAY],
            accum_exc: 0.0,
            accum_inh: 0.0,
        }
    }

    pub fn effective_threshold(&self) -> f32 {
        let modulation_scale = (1.0 + self.modulation).clamp(0.1, 10.0);
        self.base_threshold * modulation_scale + self.adaptation
    }

    pub fn step_integrate(&mut self) {
        let incoming_exc = self.future_exc[0];
        self.future_exc.rotate_left(1);
        self.future_exc[MAX_DELAY - 1] = 0.0;

        self.potential = self.potential * self.lambda_v + incoming_exc + self.accum_exc;
        self.adaptation *= self.lambda_h;
        self.activation *= self.activation_decay;
        self.modulation *= self.modulation_decay;

        let inhibition = self.accum_inh.max(0.0);
        if self.divisive_beta > 0.0 && (inhibition > 0.0 || self.adaptation > 0.0) {
            let denom = 1.0 + self.divisive_beta * (inhibition + self.adaptation.max(0.0));
            if denom > 0.0 {
                self.potential /= denom;
            }
        }
        self.inhibition_accumulator = inhibition;
        self.accum_exc = 0.0;
        self.accum_inh = 0.0;
    }

    pub fn on_spike(&mut self, step: usize) {
        self.potential = 0.0;
        self.activation = 1.0;
        self.adaptation += self.kappa;
        self.last_spike_step = Some(step);
        if matches!(self.node_type, NodeType::Memory) {
            let scaled = ((1.0 + self.modulation).clamp(0.1, 10.0)) * 0.9;
            self.modulation = (scaled - 1.0).clamp(-0.9, 9.0);
        }
    }

    pub fn reset_state(&mut self) {
        self.potential = 0.0;
        self.activation = 0.0;
        self.adaptation = 0.0;
        self.modulation = 0.0;
        self.last_spike_step = None;
        self.eligibility = 0.0;
        self.inhibition_accumulator = 0.0;
        self.future_exc = [0.0; MAX_DELAY];
        self.accum_exc = 0.0;
        self.accum_inh = 0.0;
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
    pub elig_fast: f32,
    pub elig_slow: f32,
    pub dt_ema: f32,
    pub used_step: usize,
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
            elig_fast: 0.0,
            elig_slow: 0.0,
            dt_ema: 0.0,
            used_step: 0,
            a_plus,
            a_minus,
            tau_plus,
            tau_minus,
            last_pre_spike: None,
            last_post_spike: None,
        }
    }
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
    input_nodes: Vec<usize>,
}

impl Network {
    pub fn new(nodes: Vec<Node>, connections: Vec<Connection>, input_nodes: Vec<usize>) -> Self {
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
            input_nodes,
        }
    }

    pub fn reset_state(&mut self) {
        for node in &mut self.nodes {
            node.reset_state();
        }
        for conn in &mut self.connections {
            conn.eligibility = 0.0;
            conn.elig_fast = 0.0;
            conn.elig_slow = 0.0;
            conn.dt_ema = 0.0;
            conn.used_step = 0;
            conn.last_post_spike = None;
            conn.last_pre_spike = None;
        }
    }

    pub fn inject(&mut self, excitations: &[(usize, f32)]) {
        for &(id, value) in excitations {
            if let Some(node) = self.nodes.get_mut(id) {
                node.accum_exc += value;
            }
        }
    }

    pub fn inject_embedding(&mut self, embedding: &[f32]) {
        for (node_id, value) in self.input_nodes.iter().zip(embedding.iter().copied()) {
            if let Some(node) = self.nodes.get_mut(*node_id) {
                node.accum_exc += value;
            }
        }
    }

    pub fn step(&mut self, step: usize) -> StepReport {
        let mut report = StepReport::default();

        for node in &mut self.nodes {
            node.step_integrate();
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
                self.register_pre_spike(conn_id, step);
                let conn = &self.connections[conn_id];
                let weight = conn.weight;
                if let Some(target) = self.nodes.get_mut(conn.to) {
                    target.modulation += weight;
                    if target.modulation < -0.9 {
                        target.modulation = -0.9;
                    }
                }
            }
            if let Some(node) = self.nodes.get_mut(node_id) {
                node.on_spike(step);
            }
            let incoming = self.incoming[node_id].clone();
            for conn_id in incoming {
                self.register_post_spike(conn_id, step);
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
                self.register_pre_spike(conn_id, step);
                let conn = &self.connections[conn_id];
                match node_type {
                    NodeType::Inhibitory => {
                        if let Some(target) = self.nodes.get_mut(conn.to) {
                            schedule_inhibition(target, conn.weight, conn.delay);
                        }
                    }
                    _ => {
                        if let Some(target) = self.nodes.get_mut(conn.to) {
                            deliver(target, conn.weight, conn.delay);
                        }
                    }
                }
            }
            if let Some(node) = self.nodes.get_mut(node_id) {
                node.on_spike(step);
            }
            let incoming = self.incoming[node_id].clone();
            for conn_id in incoming {
                self.register_post_spike(conn_id, step);
            }
        }
        report.spikes = spikes;

        report
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
        self.apply_reward_components(reward, reward, learning_rate);
    }

    pub fn apply_reward_components(
        &mut self,
        reward_fast: f32,
        reward_slow: f32,
        learning_rate: f32,
    ) {
        for conn in &mut self.connections {
            let delta_w =
                learning_rate * (reward_fast * conn.elig_fast + reward_slow * conn.elig_slow);
            if delta_w != 0.0 {
                conn.weight = (conn.weight + delta_w).clamp(0.0, conn.max_weight);
            }
            conn.eligibility = conn.elig_fast + conn.elig_slow;
            conn.elig_fast *= 0.98;
            conn.elig_slow *= 0.999;
            conn.used_step = 0;
        }
    }

    pub fn connection_weight(&self, connection_id: usize) -> f32 {
        self.connections
            .get(connection_id)
            .map(|conn| conn.weight)
            .unwrap_or(0.0)
    }

    pub fn connection_traces(&self, connection_id: usize) -> (f32, f32) {
        self.connections
            .get(connection_id)
            .map(|conn| (conn.elig_fast, conn.elig_slow))
            .unwrap_or((0.0, 0.0))
    }

    fn register_pre_spike(&mut self, conn_id: usize, step: usize) {
        let to = self.connections[conn_id].to;
        let dst_last_spike = self.nodes[to].last_spike_step;
        let dt = dst_last_spike.map(|last| step as f32 - last as f32);
        let conn = &mut self.connections[conn_id];
        if let Some(dt) = dt {
            let tau = conn.tau_minus.max(1e-6);
            let decay = (-(dt) / tau).exp();
            conn.elig_fast -= conn.a_minus * decay;
            conn.elig_slow -= 0.1 * conn.a_minus * decay;
            conn.dt_ema = 0.85 * conn.dt_ema + 0.15 * dt;
        } else {
            conn.dt_ema *= 0.85;
        }
        conn.last_pre_spike = Some(step);
        conn.used_step = step;
    }

    fn register_post_spike(&mut self, conn_id: usize, step: usize) {
        let pre_last = self.connections[conn_id].last_pre_spike;
        let conn = &mut self.connections[conn_id];
        if let Some(pre_last) = pre_last {
            let dt = step as f32 - pre_last as f32;
            if dt >= 0.0 {
                let tau = conn.tau_plus.max(1e-6);
                let factor = (-(dt) / tau).exp();
                conn.elig_fast += conn.a_plus * factor;
                conn.elig_slow += 0.1 * conn.a_plus * factor;
                conn.dt_ema = 0.85 * conn.dt_ema + 0.15 * dt;
            }
        } else {
            conn.dt_ema *= 0.85;
        }
        conn.last_post_spike = Some(step);
        conn.used_step = step;
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
                    NodeType::Excitatory | NodeType::Memory => 1.0,
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

fn deliver(node: &mut Node, weight: f32, delay: usize) {
    let start = delay.min(MAX_DELAY - 1);
    for (idx, coeff) in ALPHA_KERNEL.iter().enumerate() {
        let slot = start + idx;
        if slot >= MAX_DELAY {
            break;
        }
        node.future_exc[slot] += weight * coeff;
    }
}

fn schedule_inhibition(node: &mut Node, weight: f32, _delay: usize) {
    node.accum_inh += weight.abs();
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

pub mod test_helpers {
    use super::{Connection, Network, Node, NodeType};

    pub fn two_node_modulatory() -> Network {
        let nodes = vec![
            Node::new(0, NodeType::Modulatory, 0.5, 0.99, 0.9, 0.9, 0.0, 0.05, 0.0),
            Node::new(1, NodeType::Excitatory, 0.6, 0.99, 0.9, 0.9, 0.0, 0.05, 0.0),
        ];
        let connections = vec![Connection::new(0, 1, 1.0, 2.0, 0, 1.0, 1.0, 5.0, 5.0)];
        Network::new(nodes, connections, vec![0])
    }

    pub fn simple_pair() -> Network {
        let nodes = vec![
            Node::new(0, NodeType::Excitatory, 0.3, 0.99, 0.9, 0.9, 0.0, 0.05, 0.0),
            Node::new(1, NodeType::Excitatory, 0.5, 0.99, 0.9, 0.9, 0.0, 0.05, 0.0),
        ];
        let connections = vec![Connection::new(0, 1, 0.8, 2.0, 0, 1.0, 1.0, 5.0, 5.0)];
        Network::new(nodes, connections, vec![0])
    }
}

#[cfg(test)]
mod tests {
    use super::{Connection, Network, Node, NodeType};

    #[test]
    fn inhibitory_spikes_apply_divisive_normalization() {
        let nodes = vec![
            Node::new(0, NodeType::Excitatory, 0.3, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0),
            Node::new(1, NodeType::Inhibitory, 0.3, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0),
            Node::new(2, NodeType::Excitatory, 2.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0),
        ];
        let connections = vec![
            Connection::new(0, 2, 1.0, 2.0, 0, 1.0, 1.0, 5.0, 5.0),
            Connection::new(1, 2, 1.0, 2.0, 0, 1.0, 1.0, 5.0, 5.0),
        ];
        let mut network = Network::new(nodes, connections, vec![0, 1]);

        network.inject(&[(0, 1.0), (1, 1.0)]);
        let _ = network.step(0);
        let _ = network.step(1);

        let target = network.node(2);
        let expected = super::ALPHA_KERNEL[0] / (1.0 + 0.5 * 1.0);
        assert!((target.potential - expected).abs() < 1e-6);
        assert!(target.activation < 1.0);
    }

    #[test]
    fn memory_nodes_self_loop_retain_potential() {
        let nodes = vec![Node::new(
            0,
            NodeType::Memory,
            0.5,
            0.99,
            0.995,
            0.98,
            0.0,
            0.0,
            0.98,
        )];
        let connections = vec![Connection::new(0, 0, 0.4, 1.5, 0, 1.0, 1.0, 5.0, 5.0)];
        let mut network = Network::new(nodes, connections, vec![]);

        network.inject(&[(0, 1.0)]);
        let first = network.step(0);
        assert!(first.spikes.contains(&0));

        let after_spike_threshold = network.node(0).effective_threshold();
        assert!(after_spike_threshold < 0.5);

        let _ = network.step(1);
        let potential_after_one = network.node(0).potential;
        let expected_one = super::ALPHA_KERNEL[0] * 0.4;
        assert!((potential_after_one - expected_one).abs() < 1e-6);

        let report_two = network.step(2);
        assert!(report_two.spikes.contains(&0));
        let potential_after_two = network.node(0).potential;
        assert!(potential_after_two.abs() < 1e-6);

        let report_three = network.step(3);
        assert!(report_three.spikes.contains(&0));
    }

    #[test]
    fn memory_nodes_recover_threshold_after_inactivity() {
        let nodes = vec![Node::new(
            0,
            NodeType::Memory,
            0.6,
            0.99,
            0.99,
            0.98,
            0.0,
            0.0,
            0.7,
        )];
        let connections = vec![Connection::new(0, 0, 0.12, 1.0, 0, 1.0, 1.0, 5.0, 5.0)];
        let mut network = Network::new(nodes, connections, vec![]);

        network.inject(&[(0, 1.0)]);
        let _ = network.step(0);
        let lowered_threshold = network.node(0).effective_threshold();
        assert!(lowered_threshold < 0.6);

        for step in 1..=6 {
            let _ = network.step(step);
        }

        let recovered_threshold = network.node(0).effective_threshold();
        assert!(recovered_threshold > 0.58);
        assert!(recovered_threshold < 0.62);
    }

    #[test]
    fn excitatory_spike_delivers_alpha_kernel_over_time() {
        use super::ALPHA_KERNEL;

        let nodes = vec![
            Node::new(0, NodeType::Excitatory, 0.2, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0),
            Node::new(1, NodeType::Excitatory, 10.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0),
        ];
        let connections = vec![Connection::new(0, 1, 1.0, 2.0, 0, 1.0, 1.0, 5.0, 5.0)];
        let mut network = Network::new(nodes, connections, vec![0]);

        network.inject(&[(0, 5.0)]);
        let first = network.step(0);
        assert!(first.spikes.contains(&0));

        for (step, expected) in ALPHA_KERNEL.iter().copied().enumerate() {
            let report = network.step(step + 1);
            assert!(report.spikes.is_empty());
            let potential = network.node(1).potential;
            assert!(
                (potential - expected).abs() < 1e-6,
                "step {step}: expected {expected}, got {potential}"
            );
        }
    }

    #[test]
    fn eligibility_traces_update_and_decay() {
        let mut network = super::test_helpers::simple_pair();

        network.inject(&[(0, 1.0)]);
        let _ = network.step(0);
        network.inject(&[(1, 1.0)]);
        let report = network.step(1);
        assert!(report.spikes.contains(&1));

        let (fast_before, slow_before) = network.connection_traces(0);
        assert!(
            fast_before > 0.0,
            "fast trace should increase after causal pair"
        );
        assert!(slow_before > 0.0 && slow_before < fast_before);

        let weight_before = network.connection_weight(0);
        network.apply_reward_components(1.0, 0.5, 0.1);
        let weight_after = network.connection_weight(0);
        let expected_delta = 0.1 * (1.0 * fast_before + 0.5 * slow_before);
        assert!(
            (weight_after - (weight_before + expected_delta)).abs() < 1e-5,
            "unexpected weight update"
        );

        let (fast_after, slow_after) = network.connection_traces(0);
        assert!((fast_after - fast_before * 0.98).abs() < 1e-6);
        assert!((slow_after - slow_before * 0.999).abs() < 1e-6);
    }
}
