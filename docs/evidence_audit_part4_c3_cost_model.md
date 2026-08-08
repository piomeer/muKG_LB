# Phase X Part 4 — C3 Cost Model Evidence Audit

This report is generated from existing artifacts only. It does not run training, GPU code, CPU experiments, network retrieval, or runtime modifications.

## Frozen estimand

The candidate rescue estimand is held-out prediction of complete-batch negative-sampling time under the frozen CPU baseline sampler. Current artifacts do not establish this estimand.

## Claim verdicts

### C3.1-L — D (RETRACTED)

Evidence: Part1 inventory → legacy R²=0.9008/455 wording → Phase5.5 source and artifact mismatch

Paper-safe wording: Do not report R²=0.9008, 455 measured observations, or 90% explained variance.

Upgrade condition: Reconstruct the measured target and sampling unit before any new claim.

### C3.1-R1 — C (SYNTHETIC_ONLY)

Evidence: validate_weight_assumption.py → weight_validation.md (400 numeric rows) → independent r/r² recomputation

Paper-safe wording: In a synthetic validation, candidate_size was descriptively associated with measured time; this is not out-of-sample predictive validation.

Upgrade condition: Use the frozen CPU sampler, real candidate provenance, held-out complete batches, and independent seed uncertainty.

### C3.2 — C (DESCRIPTIVE_HOLD)

Evidence: runtime_attribution.py → runtime_attribution.csv (546 rows) → CBP r sensitivity recomputation

Paper-safe wording: Within one Phase6 CBP layout, batch weight and measured negative-sampling time show a descriptive association; no causal or predictive interpretation is supported.

Upgrade condition: Independent seed-grouped runs, unrounded measurements, complete-batch estimand, and held-out evaluation.

### C3.3 — A (IMPLEMENTATION_FACT)

Evidence: features.py + cost_model.py AST → deterministic CPU fixture → cost_table artifact

Paper-safe wording: Given a supplied feature array and fixed constants, the prototype deterministically constructs a lookup table without an online profiler.

Upgrade condition: None for implementation fact; predictive validity requires the separate batch-level audit protocol.

### C3.4 — C (TRANSFER_TO_C1_CONTEXT)

Evidence: gpu_cost_microbench.py → benchmark.csv (7 aggregate points) → crossover recomputation

Paper-safe wording: The stored microbenchmark provides contextual CPU/GPU timing crossover evidence under its stated, non-equivalent operations.

Upgrade condition: Raw repeated timings and matched operations if the crossover is promoted beyond design context.

### C3.5 — D (RETRACTED)

Evidence: historical hub_count artifact → corrected analysis warning → two-value/short-batch confounding

Paper-safe wording: Do not use hub_entity_count correlation as cost-model evidence.

Upgrade condition: None; replace with a separately audited variable and estimand.

### C3.6 — A (IMPLEMENTATION_FACT)

Evidence: cost_model.py array subscript → cost_table.npy dtype/shape/nbytes inspection

Paper-safe wording: The current artifact contains 14,505 float32 entries (58,020 data bytes, excluding the .npy header) and supports constant-time array lookup.

Upgrade condition: None for storage/access fact; dataset and split provenance must be added before interpreting the table empirically.

## Recomputed values

- Synthetic `candidate_size` Pearson r: 0.900793589789 (400 numeric rows).
- Synthetic `candidate_size` r²: 0.811429091405; this is not an independently established predictive R².
- Synthetic theoretical-weight Pearson r: 0.165677717567.
- CBP runtime attribution rows: 273; the report is single-layout and descriptive.
- CBP weight/time r sensitivity: all=0.712386691066, exclude-first=0.716294415657, exclude-last=0.124900441936, exclude-both=0.130005570859.
- Stored GPU microbenchmark crossover interpolation: 153596.501070482 samples; contextual only.
- Cost-table artifact: [14505] float32, 58020 data bytes excluding header.

## X7 propagation corrections

Remove or quarantine legacy `R²=0.9008`, `90% explained`, `455 sampled observations`, relation-type candidate-pool wording, and any claim that the Phase 10 bootstrap validates measured runtime prediction.

## X6.5 minimum rescue protocol

Use unrounded complete-batch observations, frozen feature/cache provenance, independent seed groups, no within-run random split, held-out R² as primary metric, MAE/RMSE/Pearson/Spearman as secondary metrics, and intercept-only/simple-feature baselines. At least three independent repeats are required; ten seed groups are recommended.

## Gate result

Current predictive C3 does not pass the X0 A/B contribution gate. C3.3 and C3.6 remain implementation facts. Any rescue experiment requires X5.5 approval and belongs to X6.5.
