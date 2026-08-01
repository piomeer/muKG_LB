# Evidence Audit Template

**Version**: 1.1
**Date**: 2026-08-01
**Purpose**: Reusable template for systematically verifying the statistical validity, variable legitimacy, and code correctness of every claim in the paper. This template will be used by Evidence Audit Parts 2–7.

---

## 1. Audit Level Definitions

Each claim receives a final credibility label from one of four audit levels:

| Level | Definition | Example |
|-------|-----------|---------|
| **A (Verified)** | Raw data, variables, protocol, statistical method, code, and wording all agree; can be written directly into the paper | A metric independently recomputed from the cited raw trace under a frozen protocol |
| **B (Re-analysis)** | Data is correct, but variables or statistical methods contain errors; requires re-analysis | unique_entities correlation |
| **C (Re-experiment)** | Experimental design has flaws; requires re-running | CBP variance compression at different batch sizes |
| **D (Invalid)** | Variable or experimental design is fundamentally invalid; must be removed from the paper | hub_entity_count correlation |

> **Audit Standard**: Assign Level A only when ALL four dimensions (data correctness, variable legitimacy, statistical method validity, code correctness) pass. Any deviation downgrades to B, C, or D depending on the severity:
> - B = data fine, analysis flawed
> - C = experimental protocol flawed (re-run needed)
> - D = fundamentally invalid (remove)

The Part 1 inventory labels `ACTIVE`, `HOLD`, and `RETRACTED` are workflow
statuses, not credibility grades. A/B/C/D may only be assigned after the
claim-level audit is complete.

---

## 2. Claim Audit Table

For each claim, fill in the following fields. One table row per claim.

| Field | Description | Audit Standard (guidance) |
|-------|-------------|---------------------------|
| **Claim ID** | e.g., C1.1, C2.3 | Must match the Claim IDs assigned in `docs/evidence_audit_part1_claim_inventory.md` |
| **Inventory Status** | ACTIVE / HOLD / RETRACTED | RETRACTED claims are audited only to document why they cannot be used |
| **Claim Statement** | One-sentence academic claim | Must be atomic: one protocol, one estimand, and one conclusion |
| **Claim Type** | Performance / variance / quality / memory / association / architecture / implementation | Determines the evidence and statistics required |
| **Frozen Protocol** | Dataset split, model, hardware, sampler semantics, scheduler, batch_size, neg_num, epochs, seeds, warm-up | All compared configurations must disclose any non-identical component |
| **Supporting Figure/Table** | Figure/Table number + path | Must map to an existing file in `paper_assets/` (verify file existence) |
| **Primary Raw Evidence** | Raw trace path and exact columns/rows | Raw evidence outranks summaries, figures, narrative reports, and story documents |
| **Derived Evidence** | Recomputed metric file, summary CSV, figure/table | Must state the transformation from primary evidence |
| **Supporting Script** | Python script path | Must map to an existing file in `src/py/` or `scripts/` (verify file existence) |
| **Key Variable(s)** | Variable name(s) used in the claim (e.g., `candidate_size`, `hub_entity_count`, `neg_std`) | List every variable that drives the claim |
| **Variable Trustworthiness** | Definition, value range (distinct values), standard deviation, continuity, physical meaning | Check: (1) Is the variable well-defined? (2) Does it have a reasonable value range? (3) Is it continuous or categorical? (4) Does it have physical meaning in the KGE context? |
| **Metric / Estimand Definition** | Unit, numerator/denominator, aggregation level, warm-up exclusion, and comparison population | Do not mix within-epoch batch dispersion, between-epoch variation, and between-run uncertainty |
| **Statistical Method Audit** | Correlation/test type, independent sample unit, repeat count, assumptions, uncertainty | Batches within one run are not independent experimental repeats; retain raw precision until final presentation |
| **Code Audit** | Variable assignment, computation logic, boundary conditions in the script | Check: (1) Is the variable computed as claimed? (2) Are boundary conditions handled? (3) Does the script reproduce the cited CSV numbers? |
| **Semantic / Fairness Audit** | Differences between compared implementations and whether wording discloses them | Required for CPU/GPU comparisons and quality claims |
| **Conclusion** | A / B / C / D | Apply the decision rules from Section 1 |
| **Paper-safe Wording** | Exact wording allowed after the audit | Must not generalize beyond the audited protocol |
| **Fix Recommendation** | If B/C: what re-analysis or re-experiment is needed | Must be specific and actionable (e.g., "Re-run with 5 repeats", "Use Spearman instead of Pearson") |
| **Responsible / Status** | Owner + open/closed | Track resolution progress |

> **Filling Guide**: The template row above is illustrative. For an actual claim, replace the guidance column with the claim's specific audit findings. Do not leave the "Audit Standard" text in filled tables.

### 2.1 Evidence Precedence

When sources disagree, use this precedence order:

1. Raw per-observation or per-step trace.
2. Independently recomputed metric from the raw trace.
3. Experiment summary CSV generated from that trace.
4. Figure/table generated from the summary.
5. Narrative report, paper draft, story freeze, handover, or memory record.

Never validate a claim by finding the same number repeated in multiple narrative
documents.

### 2.2 Frozen GPU Positioning

For this project, the GPU path is a redesigned GPU-native sampler:

- CPU baseline uses Bernoulli head/tail corruption and global collision checks.
- GPU path uses tail-only corruption and a batch-level `pos_tails` filter.
- Performance claims must disclose this semantic difference.
- “Equivalent”, “drop-in”, “same semantics”, “comparable convergence”, and
  “non-inferior” require a separate valid quality audit; runtime evidence alone
  cannot support those terms.

---

## 3. Variable Lineage Table

List all core variables used across the paper's claims, tracing their provenance.

| Variable Name | Source (Script → Function) | Used by Claims | Value Range | Definition Clarity | Audited? |
|---------------|---------------------------|----------------|-------------|--------------------|---------|
| *(e.g., `candidate_size`)* | *(e.g., `src/py/load/features.py → extract_entity_features()`)* | *(e.g., C3.1, C3.3, C4.1)* | *(e.g., [1, 10,000])* | *(Clear / Ambiguous)* | *(Yes / No)* |
| *(e.g., `neg_std`)* | *(e.g., `src/py/experiments/phase9_step3_ablation.py → compute_neg_std()`)* | *(e.g., C1.3, C4.3)* | *(e.g., [0.2, 29.5] ms)* | *(Clear / Ambiguous)* | *(Yes / No)* |
| *(e.g., `epoch_time`)* | *(e.g., `src/py/experiments/phase9_step2_benchmark.py`)* | *(e.g., C1.2, C1.4)* | *(e.g., [4.4, 25.3] s)* | *(Clear / Ambiguous)* | *(Yes / No)* |
| ... | ... | ... | ... | ... | ... |

> **Guidance**:
> - **Source**: Specify script file + function name. If the variable is computed inline without a named function, note "inline computation in `<script>`".
> - **Definition Clarity**: A variable is "Clear" if its formula/meaning is unambiguously defined in `paper/draft/method.md` or the generating script. "Ambiguous" if multiple definitions exist or the definition changed between phases (e.g., `unique_entities` vs `candidate_size`).
> - **Audited**: Set to "No" until a full audit (Section 2 rows referencing this variable) has been completed.
> - **Storage**: For array-backed variables, record the stored dtype and actual
>   artifact size. Do not substitute a hypothetical float64 size for a float32
>   artifact.

---

## 4. Cross-Claim Risk Summary Matrix

Aggregate all claim audit results into a risk matrix, grouped by contribution, to quickly see which parts of the paper are safe to write vs. which need modification.

### 4.1 Risk Matrix by Contribution

| Contribution | Claim ID | Conclusion (A/B/C/D) | Risk Level | Required Action | Status |
|--------------|----------|----------------------|------------|-----------------|--------|
| **C1: GPU Runtime** | C1.1 | *(A)* | *(Low)* | *(None)* | *(Open/Closed)* |
| | C1.2 | ... | ... | ... | ... |
| **C2: Unified Runtime Framework** | C2.1 | ... | ... | ... | ... |
| **C3: Offline Cost Model** | C3.1 | ... | ... | ... | ... |
| **C4: CBP** | C4.1 | ... | ... | ... | ... |

> **Risk Level Guidance**:
> - **Low** = Level A claim; safe to write into paper verbatim.
> - **Medium** = Level B claim; paper text must hold until re-analysis is done.
> - **High** = Level C claim; paper text must be removed or revised pending re-experiment.
> - **Critical** = Level D claim; must be removed immediately.

### 4.2 Summary Counts

| Contribution | Total Claims | A | B | C | D | % Verified |
|--------------|-------------|---|---|---|---|------------|
| C1: GPU Runtime | 9 | — | — | — | — | —% |
| C2: Unified Runtime Framework | 6 | — | — | — | — | —% |
| C3: Offline Cost Model | 6 | — | — | — | — | —% |
| C4: CBP | 7 | — | — | — | — | —% |
| **Overall** | **28** | — | — | — | — | —% |

### 4.3 Paper-Section Impact Assessment

For each paper section (from `docs/paper_outline.md`), list which claims it depends on and the resulting section-level risk.

| Paper Section | Dependent Claims | Section-Level Risk | Action Needed |
|---------------|------------------|-------------------|---------------|
| §3.1 Profiling Analysis | *(e.g., C1.9; retracted C3.5 must be removed)* | *(Low/Medium/High)* | ... |
| §3.2 Offline Cost Model | *(e.g., C3.1–C3.6)* | ... | ... |
| §3.3 CBP | *(e.g., C4.1–C4.7)* | ... | ... |
| §3.4 GPU Runtime Pipeline | *(e.g., C1.1, C1.3, C1.6)* | ... | ... |
| §3.5 Unified Runtime Framework | *(e.g., C2.1–C2.5; retracted C2.6 excluded)* | ... | ... |
| §4.2 Main Results | *(e.g., C1.2, C1.4, C2.2)* | ... | ... |
| §4.3 Ablation Study | *(e.g., C1.3, C1.7, C4.4)* | ... | ... |
| §4.4 Runtime Variance Analysis | *(e.g., C4.3)* | ... | ... |
| §4.5 Overhead & Bottleneck Shift | *(e.g., C1.9 plus a replacement scheduler-overhead claim)* | ... | ... |

> **Guidance**: A section is at High risk if it depends on any C- or D-level claim. Sections with only A-level claims are Low risk. Medium risk if B-level claims are present.

---

## 5. Usage Instructions

1. **Scope**: This template is for Evidence Audit Parts 2–7 only. It does not replace `docs/evidence_audit_part1_claim_inventory.md` (claim inventory) or `docs/evidence_matrix.md` (claim → evidence mapping).
2. **Filling Order**:
   - Part 2 → fill Sections 2, 3, 4 for C1 (GPU Runtime)
   - Part 3 → fill Sections 2, 3, 4 for C2 (Unified Runtime Framework)
   - Part 4 → fill Sections 2, 3, 4 for C3 (Offline Cost Model)
   - Part 5 → fill Sections 2, 3, 4 for C4 (CBP)
   - Part 6–7 → cross-cutting statistical validation and code audits
3. **Output Convention**: Each Part creates its own audit report file (e.g., `docs/evidence_audit_part2_c1_gpu_runtime.md`) using the tables above, with only the relevant claims.
4. **No Fabrication**: All conclusions must be based on actual data files, script inspection, and reproducible computation. No hypothetical results may be recorded.
5. **Variable Lineage Updates**: As audits proceed, update the Variable Lineage Table (Section 3) in each Part's report to build a cumulative lineage across all parts.
6. **Raw Precision**: Store full-precision measurements and compute statistics
   before rounding. Round only paper-facing tables.
7. **Claim Changes**: If an audit requires splitting or rewording a claim, retain
   the original ID with a suffix (for example C1.3a/C1.3b) and record its
   relationship to the Part 1 registry.

---

*End of Evidence Audit Template. This is a blank reusable template — it contains no audit results.*
