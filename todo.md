# TODO

## Clusterization & Narrow Channels (PR-A)
- [ ] 📦 Create `daws_protocol` crate with protobuf/capnproto schema for `MsgInference`, `MsgDelta`, `MsgResult`; ensure binary-friendly FP16/INT8 payloads.
- [ ] 🚦 Implement `router` binary crate: load NPG registry from TOML, expose flatbuffers RPC, and manage budgets per inference.
- [ ] 🎯 Add `select_experts(tags, k)` top-k policy and weighted aggregation of `MsgResult` vectors.
- [ ] 🧠 Wrap existing `Network` into `npg_server` tonic service handling iterative inference loop and streaming `MsgDelta` updates.
- [ ] ✅ Simulate E2E flow with 1 router + 3 NPG processes on synthetic XOR/brackets task and validate metrics (`iters <= 32`, `active_ratio <= 0.2`).

## Regional Context Management (PR-B)
- [ ] 🗺️ Extend `core_graph` with `RegionDescriptor`, `Network::regions`, and `gate_matrix` for selective activation.
- [ ] 🔄 Update `Network::step` to apply gating, integrate refresh pulses via `RegionalDetectorRuntime`, and cover with alternating-context test.

## Dynamic RAG within Diffusion (PR-C)
- [ ] 📉 Implement `EntropyPolicy` in `core_rules/diffusion` to trigger inhibition and `MsgDelta` fact requests on high entropy.
- [ ] 🧩 Add `FactNodes`, `inject_facts`, and `M_edit` masking to `Network` for gentle fact integration; measure ≥15% iteration reduction on SyntheticBench RAG cases.

## Online Cluster Training (PR-D)
- [ ] 🧮 Aggregate composite reward in `core_rules/reward` from logic, sparsity, consistency metrics.
- [ ] 🔁 Wire router/NPG feedback loop to call `plasticity.step` and `apply_structural_plasticity` after each request.
- [ ] 🤝 Compute inter-NPG agreement scores in router and adjust expert priorities accordingly.
- [ ] 📈 Confirm 5–10% accuracy lift on SyntheticBench analogies/brackets with visible structural plasticity logs.

## Tooling & Simulation
- [ ] 🧪 Add `daws cluster-sim` CLI command to spin router + NPG tasks under one Tokio runtime for rapid iteration.
- [ ] 📊 Integrate `tracing` + Prometheus metrics (`requests_total`, `iterations_per_req`, `active_ratio`, `reward_value`, `retriever_hits`) and document Grafana dashboard outline.
