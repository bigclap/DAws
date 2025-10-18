# TODO

## Architecture and Crate Layout

## Signal Graph (S)

## Diffusion Loop (D)
- Upgrade the diffusion policy to follow the scheduled alpha_t and sigma_t annealing, entropy-aware amplification/suppression, and ANN-based fact recruitment.
- Track convergence diagnostics (`J_t`, stability, energy monotonicity) and expose stop criteria aligned with the roadmap.

## Encoder/Decoder (E/E′)
- Swap the table encoder/decoder pair for Candle-backed models: load a frozen text encoder (`e5-small` or MiniLM) and implement the Transformer/GRU decoder with virtual prefix tokens and cosine-aligned loss.
- Build safetensors/parquet IO for embeddings, graph snapshots (`*.stt`, `*.gbin`), and KV memories (`hnsw.index`, `vecs.st`, `meta.parquet`).

## Retriever and Memory
- Implement the ANN/HNSW retriever crate with KV-memory persistence and recruitment APIs used by the diffusion heuristics.
- Integrate memory refresh pulses (Mem self-loops + gate refresh) that support long-context retention targets.

## Training and Adaptation
- Author the offline decoder trainer with mmap dataset loader, AdamW schedule (warmup + cosine), optional mixed precision, and validation metrics (cosine@median, retrieval rank@k, distinct-n, ppl surrogate).
- Add online plasticity routines that apply reward-modulated updates every `K` steps, prune/grow every `M` steps, and log eligibility traces for debugging.

## Evaluation and Profiling
- Create the synthetic evaluation bench covering XOR, analogies, stack/parentheses tasks, and long-context retrieval episodes with reporting on accuracy and cosine thresholds.
- Instrument the pipeline with `tracing`, `pprof-rs`, and energy usage metrics (active nodes/edges, wall time per iteration, memory fragmentation).
- Implement convergence reporting for diffusion (#iterations to stop, monotonicity of `J_t`) and enforce iteration hard caps.

## Tooling and Documentation
- Expand CLI commands (`train`, `eval`, `infer`, `profile`) with Clap + TOML configuration handling, surfacing profiles and checkpoints.
- Produce engineering documentation summarising ablations, convergence curves, energy comparisons against AR baselines, and final readiness criteria.
