# Phase X X0 Research Freeze Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish the approved X0 research design as the canonical RQ, scope, contribution, and estimand freeze, then make project progress and memory point to that truth source.

**Architecture:** `docs/phase_x_x0_research_freeze.md` is the canonical scientific-governance document. The approved design spec records rationale and review history; `PROGRESS.md` and `mukg-memory.json` only summarize and link the canonical freeze through the existing memory-bouncer workflow.

**Tech Stack:** Markdown, JSONL, Python 3 standard library, `utils/memory_bouncer.py`, Git.

## Global Constraints

- Do not run training, GPU experiments, or benchmark commands.
- Do not modify runtime code, training code, the Method draft, story freeze, historical evidence audits, or figures.
- C1 is the sole primary empirical contribution; C2 is supporting architecture; C3 and C4 remain conditional or exploratory.
- Performance evidence is restricted to muKG, SimpleTransE, FB15k-237, RTX 3070, batch size 5,000, and 150 negatives.
- CPU and GPU samplers are intentionally non-equivalent runtime paths; no quality-equivalence or hardware-only-speedup claim is allowed.
- The minimum viable manuscript must remain coherent if C3 and C4 fail.
- The Triple-Single boundary means one model, one dataset, and one GPU model.
- Current C1-R1 evidence contains no publication-grade batch-size or negative-count sensitivity analysis.
- Cross-model, cross-dataset, cross-hardware single-GPU validation, and Multi-GPU Scaling are separate future branches.

---

### Task 1: Create the Canonical X0 Research Freeze

**Files:**
- Create: `docs/phase_x_x0_research_freeze.md`
- Read: `docs/superpowers/specs/2026-08-03-phase-x-x0-rq-scope-contribution-estimand-freeze-design.md`
- Read: `docs/evidence_audit_part2_c1_gpu_runtime.md`
- Read: `docs/evidence_audit_part3_c2_framework.md`

**Interfaces:**
- Consumes: the approved design spec and audited C1/C2 definitions.
- Produces: one canonical Markdown truth source for all later Phase X audits and manuscript propagation.

- [ ] **Step 1: Create the canonical freeze**

Use `apply_patch` to create `docs/phase_x_x0_research_freeze.md` with these exact top-level sections:

```markdown
# Phase X X0 — Research Question, Scope, Contribution, and Estimand Freeze

## 1. Freeze Authority and Chronology
## 2. Research Questions
## 3. Scope
## 4. Contribution Hierarchy
## 5. Primary and Supporting Estimands
## 6. Quality and External-Validity Boundaries
## 7. C3/C4 Promotion Gates and Worst-Case Manuscript
## 8. Optional Generalization Branches
## 9. Manuscript Reporting Rules
## 10. Downstream Phase Gates
```

The document must reproduce these approved facts:

```text
RQ1: end-to-end epoch time under the frozen declared runtime paths
RQ2: full-batch within-epoch negative-sampling-time dispersion
RQ3: implemented runtime roles, interfaces, and execution boundaries
EQ1: cost-model validity, pending Part 4
EQ2: sorter/packer effect, pending Part 5
EQ3: performance-quality trade-off, requiring a new protocol

E1: six paired seed-level ratios; geometric mean; log-t 95% CI;
    observed 6.013x [5.944, 6.084], manuscript 6.01x [5.94, 6.08]
E2: is_partial == False AND batch_size_actual == 5000; ddof=0 per
    epoch; mean five epoch SDs per run; six paired ratios; observed
    87.88x [72.92, 105.91], manuscript 87.9x [72.9, 105.9]
E3: GPU run-level full-batch means; 3.0026 ms, SD 0.0229 ms,
    95% CI [2.9786, 3.0266] ms
E4: conjunctive implementation evidence, not a statistical estimand
```

State that X0 is a post-result paper-level formalization, while C1-R1-v1.1 was frozen before the replacement run. Do not call X0 a prospective preregistration.

- [ ] **Step 2: Validate canonical content**

Run:

```bash
python3 - <<'PY'
from pathlib import Path

p = Path("docs/phase_x_x0_research_freeze.md")
text = p.read_text(encoding="utf-8")

required = [
    "RQ1", "RQ2", "RQ3", "EQ1", "EQ2", "EQ3",
    "E1", "E2", "E3", "E4",
    "6.013", "5.944", "6.084",
    "87.88", "72.92", "105.91",
    "3.0026", "0.0229", "2.9786", "3.0266",
    "is_partial == False", "batch_size_actual == 5000", "ddof=0",
    "SimpleTransE", "FB15k-237", "RTX 3070",
    "FFDPacker == ChunkPacker",
    "post-result", "C1-R1-v1.1",
]
missing = [item for item in required if item not in text]
assert not missing, missing

forbidden = [
    "quality equivalence is established",
    "DDP-ready",
    "general KGE speedup",
    "batch-size sensitivity is verified",
    "negative-count sensitivity is verified",
]
present = [item for item in forbidden if item in text]
assert not present, present
print("canonical X0 content: PASS")
PY
```

Expected output:

```text
canonical X0 content: PASS
```

- [ ] **Step 3: Review the document diff**

Run:

```bash
git diff --check
git diff -- docs/phase_x_x0_research_freeze.md
```

Expected: no whitespace errors; only the new canonical freeze appears.

### Task 2: Update Project Progress and Memory Through the Bouncer

**Files:**
- Create temporarily: `.memory_payload.json`
- Modify through bouncer: `PROGRESS.md`
- Modify through bouncer: `mukg-memory.json`
- Execute: `utils/memory_bouncer.py`

**Interfaces:**
- Consumes: `docs/phase_x_x0_research_freeze.md`.
- Produces: current project status plus a JSONL graph entity and relations pointing to C1/C2 audit milestones.

- [ ] **Step 1: Create the exact memory payload**

Use `apply_patch` to create `.memory_payload.json`:

```json
{
  "active_task": "Phase X X0（RQ、scope、contribution 与 estimand freeze）已完成。论文采用方案 A：C1 是唯一主实证贡献，C2 是支持性实现架构，C3/C4 在 Part 4/5 通过前保持条件性或探索性。下一步进入 X1.5 文献与新颖性审计。",
  "new_constraints": [
    "X0 canonical freeze：论文最小可行主线不依赖 C3/C4；默认按二者审计失败仍可完整成稿",
    "外部有效性边界：现有性能证据仅覆盖 muKG、SimpleTransE、FB15k-237、RTX 3070、batch_size=5000、neg_num=150",
    "当前无论文级 batch-size/neg-num sensitivity；Phase 10 舍入数据不得替代；跨模型、跨数据集、跨 GPU 型号单卡复现与多 GPU scaling 必须分别注册新 Claim/协议",
    "RQ1/RQ2 是 post-result paper-level formalization 下的 primary RQ/estimand；C1-R1-v1.1 replacement protocol 在补跑前冻结，不得把 X0 称为前瞻性预注册"
  ],
  "progress_and_blockers": "已批准并冻结 docs/phase_x_x0_research_freeze.md。RQ1 对应 E1 六 paired seeds 的 epoch speedup 6.013× [5.944, 6.084]；RQ2 对应 E2 full-batch within-epoch SD compression 87.88× [72.92, 105.91]；E3 为 GPU full-batch neg time 3.0026ms [2.9786, 3.0266]；RQ3/E4 限于实现事实。C3 需 Part 4 恢复 target provenance、排除 leakage 并建立 out-of-sample estimand；C4 受 FFDPacker==ChunkPacker 阻塞。三单外部有效性仍是投稿风险，但不通过无证据措辞扩张 scope。",
  "next_steps": "1. 执行 X1.5 文献与新颖性审计，建立 KGE runtime systems / GPU negative sampling / modular reproducibility 的 systematic mapping 与 novelty matrix。\\n2. Part 4 审计 C3，决定 CostModel 是贡献、实现细节还是撤回。\\n3. Part 5 审计 C4，并按最坏情况骨架决定 CBP 进入正文、附录或删除。\\n4. 只有完成前三项贡献裁决后，才设计可选的跨模型、跨数据集、跨 GPU 型号或多 GPU gap-closing experiments。",
  "l2_graph_updates": [
    {
      "type": "entity",
      "name": "PhaseX_X0_Research_Freeze",
      "entityType": "ResearchGovernanceMilestone",
      "observations": [
        "[2026-08-03] 方案 A 冻结：C1 是唯一主实证贡献，C2 是支持性架构，C3/C4 保持条件性或探索性。",
        "[2026-08-03] RQ1/E1 冻结六 paired seeds 的 epoch speedup：6.013×，95% CI [5.944, 6.084]；RQ2/E2 冻结 full-batch within-epoch SD compression：87.88×，95% CI [72.92, 105.91]。",
        "[2026-08-03] Scope 限于 muKG、SimpleTransE、FB15k-237、RTX 3070、batch_size=5000、neg_num=150；不主张 sampler 语义等价或质量 non-inferiority。",
        "[2026-08-03] 最坏情况论文骨架不依赖 CostModel/CBP 成功；当前无合格 batch-size/neg-num sensitivity。",
        "[2026-08-03] 跨模型、跨数据集、跨 GPU 型号的单卡复现与多 GPU scaling 被定义为不同的可选扩展分支。"
      ]
    },
    {
      "type": "relation",
      "from": "PhaseX_X0_Research_Freeze",
      "to": "C1_R1_Combined_Rerun",
      "relationType": "FREEZES_PRIMARY_EVIDENCE"
    },
    {
      "type": "relation",
      "from": "PhaseX_X0_Research_Freeze",
      "to": "Evidence_Audit_Part3_C2_Framework",
      "relationType": "SCOPES_SUPPORTING_CONTRIBUTION"
    }
  ]
}
```

- [ ] **Step 2: Run the memory bouncer**

Run:

```bash
python3 utils/memory_bouncer.py
```

Expected output contains:

```text
[BOUNCER] ✅ Payload 校验通过 (3 条 L2 更新)
[BOUNCER] ✅ L2 记忆合并完成: 追加 3 行到 mukg-memory.json
[BOUNCER] ✅ L3 进度沉淀完成: PROGRESS.md
[BOUNCER] 🧹 已删除 .memory_payload.json
```

- [ ] **Step 3: Validate progress and JSONL memory**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path

progress = Path("PROGRESS.md").read_text(encoding="utf-8")
for item in [
    "Phase X X0",
    "C1 是唯一主实证贡献",
    "X1.5 文献与新颖性审计",
    "FFDPacker==ChunkPacker",
]:
    assert item in progress, item

records = [
    json.loads(line)
    for line in Path("mukg-memory.json").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
freeze = [
    item for item in records
    if item.get("type") == "entity"
    and item.get("name") == "PhaseX_X0_Research_Freeze"
]
assert len(freeze) == 1, len(freeze)
relations = [
    item for item in records
    if item.get("type") == "relation"
    and item.get("from") == "PhaseX_X0_Research_Freeze"
]
assert {
    (item["to"], item["relationType"]) for item in relations
} == {
    ("C1_R1_Combined_Rerun", "FREEZES_PRIMARY_EVIDENCE"),
    ("Evidence_Audit_Part3_C2_Framework", "SCOPES_SUPPORTING_CONTRIBUTION"),
}
print("progress and memory: PASS")
PY
```

Expected output:

```text
progress and memory: PASS
```

### Task 3: Final Verification and Scoped Commit

**Files:**
- Add: `docs/phase_x_x0_research_freeze.md`
- Modify: `PROGRESS.md`
- Modify: `mukg-memory.json`

**Interfaces:**
- Consumes: Tasks 1 and 2.
- Produces: one verified Git commit containing only X0 documentation and state updates.

- [ ] **Step 1: Confirm the modification boundary**

Run:

```bash
git status --short
```

Expected paths only:

```text
 M PROGRESS.md
 M mukg-memory.json
?? docs/phase_x_x0_research_freeze.md
```

- [ ] **Step 2: Run final checks**

Run:

```bash
git diff --check
python3 - <<'PY'
import json
from pathlib import Path
for number, line in enumerate(
    Path("mukg-memory.json").read_text(encoding="utf-8").splitlines(), 1
):
    if line.strip():
        json.loads(line)
print("JSONL parse: PASS")
PY
```

Expected: `git diff --check` has no output and JSON parsing prints `PASS`.

- [ ] **Step 3: Stage only scoped files**

Run:

```bash
git add \
  PROGRESS.md \
  mukg-memory.json \
  docs/phase_x_x0_research_freeze.md
git diff --cached --check
git diff --cached --stat
```

Expected: three scoped files and no whitespace errors.

- [ ] **Step 4: Commit**

Run:

```bash
git commit -m "docs: freeze Phase X research scope and estimands"
```

Expected: one commit on `production`. Pushing is intentionally outside this
plan unless the user separately authorizes publishing the completed X0 commit.
