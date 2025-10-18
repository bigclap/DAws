# Done

- Established a self-contained XOR demonstration that wires the diffusion loop, reward calculator, and spiking network core through the CLI for quick smoke testing.
- Implemented an event-driven spiking network with excitatory, inhibitory, modulatory, and memory node dynamics, divisive normalisation, alpha-kernel delivery, and reward-modulated STDP traces.
- Added configurable inhibitory pools with regional detectors to drive winner-take-all suppression and gate refresh monitoring.
- Added deterministic diffusion refinement with cosine-based convergence tracking to stabilise activation states between graph steps.
- Upgraded the diffusion loop with annealed alpha/sigma schedules, entropy-scaled gain, and ANN fact recruitment backed by tests.
- Provided basic lookup-table encoder and binary decoder utilities so the end-to-end example can run without external model weights.
- Covered the current surface area with unit tests exercising diffusion convergence, modulation, memory retention, divisive inhibition, eligibility decay, reward application, and XOR correctness.
- Documented all exposed modules with Rustdoc so that `cargo doc` surfaces architectural intent directly from the code.
- Split the prototype into a multi-crate workspace (`core_graph`, `core_rules`, `model_enc`, `model_dec`, `retriever`, `trainer`, `datasets`, `evalbench`, `cli`) with shared dependencies to mirror the E → S → D → E′ stack.
- Replaced the XOR demo plumbing with reusable graph assembly, scheduling, and profiling components shared across crates.
- Implemented structural plasticity with co-activation counters, growth/pruning rules, and delay retuning informed by eligibility traces.
- Extended node state handling with energy caps, NaN guards, and episodic reset policies aligned with production requirements.
