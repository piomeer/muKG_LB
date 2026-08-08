# Phase X X5 — C4 CBP Evidence Audit

本报告由现有 artifact、源码 AST 和确定性 CPU fixture 生成。X5 不运行 GPU、训练、新实验、网络或运行时代码修改。

## Frozen interpretation

当前历史比较是 RandomSorter+ChunkPacker 与 CostSorter+FFDPacker。源码和 fixture 证明 FFDPacker 对有序输入等价于 sequential ChunkPacker，因此历史结果不能识别独立 packer effect。

Composite gate: **FAIL**. A sorter-only remedial candidate is forwarded to X5.5; FFD、packing 和 CBP 独立贡献不得写入正文。

## Claim verdicts

### C4.1-L — C (HISTORICAL_PROTOCOL_LIMITED)

Evidence chain: Part1 C4.1 → Phase6 runtime_attribution.csv → all-batch SD recomputation and warm-up/partial sensitivity

Recomputed: 15.5295 ms Baseline vs 3.4086 ms CBP all rows; 1.0509 vs 1.1285 ms after first/last exclusion

Paper-safe wording: A single Phase6 trace showed lower all-batch dispersion under the historical cost-sorted layout; the effect disappears under a complete-batch estimand.

Failure condition: Do not report 78% as a validated CBP variance reduction.

Upgrade condition: Independent seed-grouped complete-batch repeats with a behaviorally distinct packer and pre-registered factor analysis.

### C4.1-R1 — B (REANALYSIS_NO_COMPRESSION)

Evidence chain: Phase6 runtime_attribution.csv → fixed exclusion filters → population SD recomputation

Recomputed: Complete interior rows: Baseline 1.0509 ms, CBP 1.1285 ms; CBP is approximately 7.4% higher.

Paper-safe wording: After excluding the first warm-up and final short batch, the historical cost-sorted layout did not reduce Phase6 negative-sampling dispersion.

Failure condition: Single trace and no independent repeats prevent an A-level runtime conclusion.

Upgrade condition: Six paired seeds, five measured epochs and a factor-isolated runtime estimand.

### C4.2 — A (IMPLEMENTATION_FACT_ONLY)

Evidence chain: schedulers.py AST/source → BaseSorter/RandomSorter/CostSorter and BasePacker/ChunkPacker/FFDPacker

Recomputed: Both sorter classes and both packer classes exist; legacy FFD fixture equals Chunk fixture for all tested ordered inputs.

Paper-safe wording: The prototype exposes composable sorter and packer interfaces; the existing FFD implementation is behaviorally equivalent to sequential chunking.

Failure condition: Do not call the four combinations behaviorally distinct without a new packer.

Upgrade condition: A distinct packer implementation plus factorial evidence separating sorter, packer and interaction effects.

### C4.3-L — C (HISTORICAL_PROTOCOL_LIMITED)

Evidence chain: Phase9 Step4.5 variance CSV → pooled all-batch SD recomputation → partial-batch sensitivity

Recomputed: Pooled SD: BL 29.4979 ms, CBP 27.0065 ms, 8.45% lower; this includes the final 2,115-sample batch.

Paper-safe wording: The stored Phase9 trace shows a descriptive all-batch difference under its stated protocol; the pooled statistic is not the primary complete-batch estimand.

Failure condition: Do not treat three nested epochs in one process as independent repeats.

Upgrade condition: Unrounded complete-batch traces from independent paired seed groups.

### C4.3-R1 — B (DESCRIPTIVE_REANALYSIS)

Evidence chain: Phase9 Step4.5 variance CSV → per-epoch full-batch ddof=0 SD → mean across three nested epochs

Recomputed: Mean epoch SD: BL 9.2381 ms, CBP 2.4537 ms; descriptive reduction 73.44%.

Paper-safe wording: Within this single process and its fixed layouts, excluding partial batches yielded lower descriptive per-epoch dispersion for the cost-sorted layout.

Failure condition: Rounded two-decimal observations and nested epochs do not support independent-repeat uncertainty.

Upgrade condition: Six paired seeds, unrounded timing, unified seeds and pre-registered factorial contrasts.

### C4.4 — C (SUMMARY_ONLY)

Evidence chain: Phase9 Step3 summary.csv → ten epoch summaries per configuration → seed/script lineage audit

Recomputed: Summary-level BL/CBP negative-time dispersion is similar; no per-step trace or independent repeat is available.

Paper-safe wording: The ten-epoch summary does not establish a robust runtime-dispersion advantage for CBP.

Failure condition: Do not treat epochs as independent runs; process-dependent hash(label) seed remains unresolved.

Upgrade condition: Independent processes, unified seeds and raw per-batch timings.

### C4.5 — C (QUALITY_TRACEABILITY_ONLY)

Evidence chain: Phase9 Step2 summary.csv → five epochs and 200-sample training holdout → protocol scope audit

Recomputed: Values are traceable but use a non-official, small sampled holdout and no qualified repeat design.

Paper-safe wording: A five-epoch sampled training-holdout quality diagnostic was recorded; it does not establish quality equivalence or non-inferiority.

Failure condition: Exclude from C4 runtime contribution and do not use non-inferiority language.

Upgrade condition: A separately approved official evaluation protocol.

### C4.6 — C (QUALITY_TRACEABILITY_ONLY)

Evidence chain: Phase9 Step2 CBP+GPU/GPU summaries → five-epoch sampled holdout → quality scope audit

Recomputed: Values are traceable but do not support CPU/GPU or CBP+GPU quality comparison.

Paper-safe wording: The stored sampled diagnostic is retained for audit history only and is excluded from the paper quality argument.

Failure condition: Exclude from C4 runtime contribution and prohibit quality non-inferiority claims.

Upgrade condition: A separately approved official evaluation protocol with independent repeats.

### C4.7-L — B (MISATTRIBUTED_LAYOUT_METRIC)

Evidence chain: integration validation summary → batch_composition.csv → within-batch predicted-cost CV recomputation

Recomputed: Stored .0552→.0124 values are reproducible, but the artifact measures static within-batch cost homogeneity.

Paper-safe wording: The integration artifact records more homogeneous predicted costs within complete cost-sorted batches under its deterministic layout.

Failure condition: Do not call this inter-batch runtime balancing or a packing effect.

Upgrade condition: Behaviorally distinct packer and runtime traces with an explicit batch-balance estimand.

### C4.7-R1 — A (DETERMINISTIC_LAYOUT_FACT)

Evidence chain: integration batch_composition.csv → full-batch filter → CV recomputation and cost-table degeneracy audit

Recomputed: 108 complete CostSorter+sequential-chunk rows have mean within-batch predicted-cost CV 0.0; partial rows remain heterogeneous.

Paper-safe wording: Under the stored deterministic layout, complete cost-sorted batches have zero within-batch CV for the predicted cost table; this is a layout fact, not runtime balance evidence.

Failure condition: Do not infer sampler-time equalization or FFD causality.

Upgrade condition: None for the layout fact; runtime attribution requires a separate factorial experiment.

## Recomputed metrics

- Phase6 all-row SD: Baseline `15.52949488` ms; CBP `3.408583408` ms.
- Phase6 complete interior SD: Baseline `1.05092422511` ms; CBP `1.12854009325` ms.
- Phase9 Step4.5 pooled SD: BL `29.4978734511` ms; CBP `27.0065301487` ms.
- Phase9 Step4.5 mean epoch complete-batch SD: BL `9.23807309853` ms; CBP `2.45367637529` ms.
- Integration full-batch within-cost CV: Baseline `0.05503420124`; CBP `0`.
- Cost table: `14505` `float32` entries, `166` unique values; dominant value `518.0` occurs `14303` times (`0.986074`).

## Mechanism mapping

- `M1` (sorter): CostSorter orders descending by max(head_cost, tail_cost) — SORTER_PLAUSIBLE_PACKER_UNIDENTIFIED.
- `M2` (packer): FFDPacker fills batch 0, then batch 1; output equals ChunkPacker — BLOCKED_EQUIVALENCE.
- `M3` (interaction): Only BL/CBP/GPU/CBP+GPU are enumerated — NOT_IDENTIFIABLE.
- `M4` (cost_table): 98.6074% of entries equal 518.0; 166 unique values — CONFOUNDING_DIAGNOSTIC.
- `M5` (batch_boundary): Warm-up and final partial batch change dispersion direction — CONFIRMED_SENSITIVITY.

## Statistical fallacy scan

- `F01` aggregation/Simpson: **CAUTION** — all-batch and complete-batch results differ; report complete-batch primary estimand and sensitivity.
- `F02` ecological inference: **CAUTION** — batch CV and epoch summaries; seed-level unit for X6.5.
- `F03` selection/Berkson: **CAUTION** — partial batches and sampled holdout; pre-register complete-batch filter and official quality protocol.
- `F04` collider: **NOT_APPLICABLE** — no collider adjustment identified; monitor in future factorial analysis.
- `F05` base-rate neglect: **NOT_APPLICABLE** — not a prevalence claim; none.
- `F06` regression to mean: **NOT_APPLICABLE** — no before/after selection design; paired seeds in X6.5.
- `F07` survivorship: **CAUTION** — only available historical artifacts are analyzed; source manifest and explicit missingness.
- `F08` look-elsewhere: **CAUTION** — multiple phases, batch filters and metrics; label historical and primary estimands separately.
- `F09` forking paths: **CAUTION** — post-hoc warm-up/partial sensitivity; freeze X6.5 filters before data.
- `F10` correlation/causation: **FAIL** — sorter and packer changed together; static CV only; 2x2 factorial with distinct packer.
- `F11` reverse causality: **NOT_APPLICABLE** — layout is chosen before measured time; retain timing boundary and seed provenance.

## X6.5 candidate protocol

若 X5.5 批准，仅执行 CPU sampler 因子的完整训练内计时：Random/Cost sorter × Chunk/GreedyLeastLoad packer，seed 42–47，每个 seed/config 独立进程、3 个 warm-up step、5 个 measured epochs，主结果为 53 个完整 batch 的逐 epoch ddof=0 SD 均值。

晋级门槛：独立因素至少降低 10%，配对 95% CI 上界低于 1，平均 neg_time 增幅不超过 5%，且布局 fixture 证明新 packer 与 Chunk 不同。

## Audit status

C4.5/C4.6 仅为质量 traceability；当前论文主路径继续采用 RandomSorter+ChunkPacker。
