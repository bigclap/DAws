# Architecture Overview

This document outlines how the DAws workspace is organised and how data moves through the
reasoning pipeline. It is intended to complement the inline Rustdoc comments by providing a
single reference that links the major crates together.

## Workspace layout

- `cli/` — exposes the `daws` binary with `train`, `eval`, `infer`, and `profile`
  commands that funnel into dedicated execution helpers and shared configuration
  loading logic.【F:cli/src/main.rs†L1-L54】【F:cli/src/commands/mod.rs†L1-L23】
- `core_graph/` — hosts the event-driven spiking network, declarative assembly
  utilities, inhibitory pool definitions, and profiling helpers used by the
  diffusion scheduler.【F:core_graph/src/lib.rs†L1-L12】【F:core_graph/src/assembly.rs†L1-L72】
- `core_rules/` — contains the deterministic diffusion loop, reward shaping
  utilities, and the reasoning scheduler that coordinates graph stepping with
  diffusion refinement.【F:core_rules/src/lib.rs†L1-L6】【F:core_rules/src/scheduler.rs†L1-L64】
- `model_enc/` — provides the Candle-backed frozen encoder implementation with
  support for Hugging Face BERT checkpoints or lightweight embedding tables for
  tests.【F:model_enc/src/lib.rs†L1-L74】
- `model_dec/` — implements the cosine-aligned decoder head, binary thresholding
  helpers, and loss computation used during validation.【F:model_dec/src/lib.rs†L1-L80】
- `retriever/` — wraps an HNSW index with gate-aware refresh logic so the
  diffusion loop can recruit supporting memories and persist snapshots.【F:retriever/src/lib.rs†L1-L10】
  Search, gating, and snapshot persistence live in `store.rs`.【F:retriever/src/store.rs†L1-L118】
- `trainer/` — orchestrates offline decoder training, online reward-modulated
  plasticity, validation metrics, and AdamW scheduling utilities.【F:trainer/src/lib.rs†L1-L13】
  The optimiser schedule is provided by `optimizer.rs`.【F:trainer/src/optimizer.rs†L1-L84】
- `datasets/` — defines dataset descriptors and the mmap-backed JSONL loader used
  by the trainer.【F:datasets/src/lib.rs†L1-L88】
- `evalbench/` — packages the synthetic evaluation suites that validate the
  pipeline end-to-end and surface accuracy metrics.【F:evalbench/src/lib.rs†L1-L76】

## Reasoning pipeline

1. **Encoding (E)** — The `FrozenTextEncoder` converts input text into pooled
   embeddings using either a BERT checkpoint or an embedded lookup table
   depending on the build configuration.【F:model_enc/src/lib.rs†L31-L108】
2. **Retrieval (R)** — Encodings optionally query the HNSW-backed retriever,
   which normalises the embedding, executes a cosine search, and refreshes memory
   gates to prioritise frequently accessed entries.【F:retriever/src/store.rs†L47-L118】
3. **Signal graph (S)** — Retrieved memories and query embeddings are injected
   into the spiking `Network`, which simulates excitatory, inhibitory,
   modulatory, and memory nodes with alpha-kernel delivery, divisive
   normalisation, and reward-modulated STDP traces.【F:core_graph/src/lib.rs†L13-L120】
4. **Diffusion loop (D)** — After a configurable number of settle steps the
   `ReasoningScheduler` drives the deterministic diffusion loop, applying
   annealed alpha/sigma schedules, entropy-sensitive gain, and optional fact
   recruitment until convergence metrics stabilise.【F:core_rules/src/scheduler.rs†L28-L80】【F:core_rules/src/diffusion/mod.rs†L1-L96】
5. **Decoder (E′)** — Stabilised activations are decoded into logits or token
   probabilities by the decoder utilities, which also provide cosine-aligned
   loss functions for training and evaluation.【F:model_dec/src/lib.rs†L1-L86】

## Training and adaptation

- **Offline decoder training** uses `OfflineDecoderTrainer` to compute validation
  metrics such as cosine@median, retrieval rank@k, distinct-n, and a perplexity
  surrogate while stepping through dataset batches.【F:trainer/src/offline.rs†L1-L142】
- **Online plasticity** leverages `OnlinePlasticity` to periodically update
  decoder weights, prune dormant connections, and regrow structure based on
  reward-modulated traces while maintaining a rolling trace log.【F:trainer/src/online.rs†L1-L176】
- **Optimisation schedules** rely on `AdamWSchedule` to perform warmup and cosine
  decay, aligning the trainer with the energy-aware diffusion loop.【F:trainer/src/optimizer.rs†L1-L84】

## Evaluation and tooling

- `evalbench` exposes reusable suites such as XOR, analogies, stack/parentheses,
  and retrieval-heavy tasks to exercise the full reasoning stack.【F:evalbench/src/lib.rs†L18-L113】
- Integration tests under `core_graph/tests` and `core_rules/tests` cover
  convergence diagnostics, inhibitory pool behaviour, reward shaping, and
  diffusion recruitment, acting as executable documentation for the
  architecture.【F:core_graph/tests/diffusion_pipeline.rs†L1-L112】【F:core_rules/tests/reward_plasticity.rs†L1-L58】
- The CLI `profile` command collects CPU flamegraphs while driving the XOR demo,
  helping diagnose performance regressions alongside the in-graph profiling
  hooks.【F:cli/src/commands/profile.rs†L1-L114】【F:core_graph/src/profiling.rs†L1-L142】
