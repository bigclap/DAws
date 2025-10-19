# TODO

## Architecture and Crate Layout

## Signal Graph (S)

## Diffusion Loop (D)

## Encoder/Decoder (E/E′)

## Retriever and Memory

## Training and Adaptation

## Evaluation and Profiling
- Implement convergence reporting for diffusion (#iterations to stop, monotonicity of `J_t`) and enforce iteration hard caps.

## Tooling and Documentation
- Expand CLI commands (`train`, `eval`, `infer`, `profile`) with Clap + TOML configuration handling, surfacing profiles and checkpoints.
- Produce engineering documentation summarising ablations, convergence curves, energy comparisons against AR baselines, and final readiness criteria.
