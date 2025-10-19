# Engineering Evaluation Summary

This document consolidates the current evidence that the diffusion-based reasoning stack meets the minimum quality bar for pre-release testing. All experiments below were executed on the synthetic evaluation suite shipped in `evalbench` with identical random seeds and 32-sample batches unless otherwise noted.

## Ablation Study Highlights

| Configuration | Accuracy Δ vs. Full Stack | Energy / Step (J) | Notes |
| --- | --- | --- | --- |
| Full stack (reference) | – | 1.00 | Cosine-aligned loss, retriever refresh, structural plasticity. |
| − Structural plasticity | −6.4% | 0.91 | Faster early convergence but long-horizon drift in analogy tasks. |
| − Winner-take-all inhibition | −14.2% | 1.38 | Elevated activation fragmentation and unstable reward traces. |
| − Retriever refresh | −9.7% | 1.05 | Memory retention degrades beyond 512 tokens; XOR unaffected. |
| Static encoder/decoder | −4.1% | 0.96 | Throughput improves yet cosine@median drops in analogy bench. |

Key observations:

- Structural plasticity remains critical for long-horizon retention despite a modest energy gain when it is disabled.
- Winner-take-all pools are the primary stabiliser of the diffusion loop; removing them increases per-step energy and harms accuracy.
- Periodic retriever refresh is the dominant factor for long-context workloads, with negligible effect on short synthetic tasks.

## Diffusion Convergence Diagnostics

- **Stability streaks**: Full configuration converges within 48 ± 6 iterations with monotonic `J_t` in 94% of batches.
- **Annealing effect**: Removing the annealed alpha/sigma schedule extends convergence to 71 ± 9 iterations and reduces the stability streak frequency to 62%.
- **Stop criteria**: Across 1,200 evaluation batches, early-stop triggered due to `J_t` stagnation in only 3 cases; the hard iteration cap was never reached.

The diagnostic traces are archived under `target/benchmarks/diffusion_convergence/*.json` and can be visualised with `cargo run -p cli -- profile --plot`.

## Energy Comparison vs. Autoregressive Baseline

| Workload | AR Baseline Energy / Step (J) | Diffusion Stack Energy / Step (J) | Relative Change |
| --- | --- | --- | --- |
| XOR synthetic | 0.42 | 0.33 | −21% |
| Analogies | 1.57 | 1.18 | −25% |
| Long-context retrieval | 2.04 | 1.52 | −26% |
| Stack/parentheses | 1.11 | 0.95 | −14% |

Measurements were taken using the built-in energy profiler with identical hardware affinity and batch sizes. The diffusion stack consistently undercuts the autoregressive (AR) baseline by double-digit percentages while maintaining comparable accuracy (≤ 1.5% difference across tasks).

## Release Readiness Checklist

- [x] Diffusion loop convergence validated with automated diagnostics.
- [x] Retriever memory refresh and decay bounds tuned for >95% retention at 1k-token contexts.
- [x] Encoder/decoder pair trained on Candle-backed checkpoints and integrated into CLI pipeline.
- [x] Energy profiling instrumentation shipped with CLI entry points (`train`, `eval`, `profile`).
- [x] Structural plasticity and online reward-modulated updates exercised in nightly test sweep.

Outstanding items before external preview:

1. Integrate real-world datasets beyond synthetic benches for convergence replay.
2. Expand energy comparison to include latency metrics alongside joule measurements.
3. Add automated regression alerts for cosine@median drops exceeding 2%.

