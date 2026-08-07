# Phase X X0.5 Legacy Narrative Quarantine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an auditable firewall between superseded Phase X narratives/data and the canonical evidence used for manuscript writing.

**Architecture:** A central Markdown register provides human-readable authority, Claim-ID, data-lineage, release-state, and safe-writing guidance. A standalone standard-library Python checker validates the register and five legacy HTML-comment headers and emits deterministic JSON; it does not modify the completed Part 3 audit. Part 1 remains byte-for-byte frozen, with reverse traceability supplied by X0.5.

**Tech Stack:** Markdown, Python 3 standard library, `unittest`, deterministic JSON, `utils/memory_bouncer.py`, Git.

## Global Constraints

- Do not run GPU/CPU training, benchmarks, profiling, or experiment scripts.
- Do not modify runtime/training code, result CSVs, figures, or Part 1.
- Do not rewrite the substantive body of Method, outline, story freeze, runtime spec, or Phase 8 architecture freeze.
- Use the exact HTML marker `<!-- LEGACY-NON-AUTHORITATIVE: Phase X X0.5 -->`.
- Keep `docs/evidence_audit_part1_claim_inventory.md` byte-for-byte unchanged.
- Use `output/results/c1_r1_combined_rerun/` as the C1-R1 v1.1 evidence path; `output/results/c1_r1_v1.1/` does not exist.
- Treat `output/results/unified_runtime/`, `phase9_step2/`, and `phase9_step3/` as legacy/audit-only for superseded C1 performance claims.
- Treat C3/C4 source directories as pending audit, not current paper evidence.
- Checker output must contain no timestamp and must be byte-identical across repeated runs.
- X0.5 records governance state; it does not promote a Claim or release quarantined wording.

---

### Task 1: Implement the Fail-Closed Quarantine Checker

**Files:**
- Create: `scripts/check_x0_5_quarantine.py`
- Create: `tests/test_check_x0_5_quarantine.py`

**Interfaces:**
- Consumes: a repository root containing the X0.5 register, Part 1 inventory, canonical sources, data directories, and five legacy documents.
- Produces: `build_checks(repo_root: Path) -> dict[str, object]`, `write_output(result: dict[str, object], output_path: Path) -> None`, and a CLI returning zero only when every check passes.

- [ ] **Step 1: Write fixture helpers and failing tests**

Create `tests/test_check_x0_5_quarantine.py` with `unittest` and `tempfile`. The fixture must build a minimal repository containing:

```python
HEADER_MARKER = "<!-- LEGACY-NON-AUTHORITATIVE: Phase X X0.5 -->"
REGISTER_PATH = "docs/phase_x_x0_5_legacy_narrative_quarantine.md"
LEGACY_FILES = (
    "paper/draft/method.md",
    "docs/paper_outline.md",
    "docs/paper_story_freeze.md",
    "docs/runtime_framework_spec.md",
    "docs/phase8_architecture_freeze.md",
)
EXPECTED_Q_IDS = tuple(f"Q-{number:02d}" for number in range(1, 12))
```

The fixture register must use this table header exactly:

```markdown
| Quarantine ID | Affected Claim IDs | Legacy Expression | Affected Documents/Data | Disposition | Canonical Source | Paper-Use Rule | Audit Owner / State | Explicit Release Condition |
```

Add `make_valid_fixture(root: Path) -> None` that creates every path declared by
the checker, writes one marker into each legacy file, writes Q-01 through Q-11,
and writes the full reverse-index table. Add
`replace_once(path: Path, old: str, new: str) -> None` that asserts the old
substring occurs exactly once before replacing it. Then add these tests:

```python
class X05QuarantineCheckerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.repo_root = Path(self.tempdir.name)
        make_valid_fixture(self.repo_root)

    def test_valid_fixture_passes_all_checks(self) -> None:
        result = build_checks(self.repo_root)
        self.assertEqual(result["overall_status"], "PASS")
        self.assertTrue(all(row["status"] == "PASS" for row in result["checks"]))

    def test_missing_or_duplicate_header_fails_closed(self) -> None:
        method = self.repo_root / LEGACY_FILES[0]
        replace_once(method, HEADER_MARKER, "")
        self.assertEqual(build_checks(self.repo_root)["overall_status"], "FAIL")

        make_valid_fixture(self.repo_root)
        method.write_text(
            method.read_text(encoding="utf-8") + HEADER_MARKER + "\n",
            encoding="utf-8",
        )
        self.assertEqual(build_checks(self.repo_root)["overall_status"], "FAIL")

    def test_missing_q_row_or_claim_reverse_mapping_fails_closed(self) -> None:
        register = self.repo_root / REGISTER_PATH
        lines = register.read_text(encoding="utf-8").splitlines()
        register.write_text(
            "\n".join(line for line in lines if not line.startswith("| Q-11 |")) + "\n",
            encoding="utf-8",
        )
        self.assertEqual(build_checks(self.repo_root)["overall_status"], "FAIL")

        make_valid_fixture(self.repo_root)
        replace_once(register, "| C2.3 | Q-11 |", "| C2.3 |  |")
        self.assertEqual(build_checks(self.repo_root)["overall_status"], "FAIL")

    def test_missing_safe_source_or_data_directory_fails_closed(self) -> None:
        missing = self.repo_root / "docs/phase_x_x0_research_freeze.md"
        missing.unlink()
        self.assertEqual(build_checks(self.repo_root)["overall_status"], "FAIL")

    def test_blank_release_field_fails_closed(self) -> None:
        register = self.repo_root / REGISTER_PATH
        lines = register.read_text(encoding="utf-8").splitlines()
        q01 = next(line for line in lines if line.startswith("| Q-01 |"))
        cells = [cell.strip() for cell in q01.strip().strip("|").split("|")]
        cells[-1] = ""
        blank_release = "| " + " | ".join(cells) + " |"
        replace_once(register, q01, blank_release)
        self.assertEqual(build_checks(self.repo_root)["overall_status"], "FAIL")

    def test_json_output_is_byte_deterministic(self) -> None:
        result = build_checks(self.repo_root)
        first = self.repo_root / "first.json"
        second = self.repo_root / "second.json"
        write_output(result, first)
        write_output(result, second)
        self.assertEqual(first.read_bytes(), second.read_bytes())
```

- [ ] **Step 2: Run the tests and verify the checker is absent**

Run:

```bash
python3 -m unittest tests/test_check_x0_5_quarantine.py -v
```

Expected: `ERROR` with `ModuleNotFoundError: No module named 'scripts.check_x0_5_quarantine'`.

- [ ] **Step 3: Implement constants, parsing, and checks**

Create `scripts/check_x0_5_quarantine.py` using only the standard library. Define:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

HEADER_MARKER = "<!-- LEGACY-NON-AUTHORITATIVE: Phase X X0.5 -->"
REGISTER_PATH = "docs/phase_x_x0_5_legacy_narrative_quarantine.md"

LEGACY_FILES = (
    "paper/draft/method.md",
    "docs/paper_outline.md",
    "docs/paper_story_freeze.md",
    "docs/runtime_framework_spec.md",
    "docs/phase8_architecture_freeze.md",
)

SAFE_SOURCE_PATHS = (
    "docs/phase_x_x0_research_freeze.md",
    "docs/evidence_audit_part2_c1_gpu_runtime.md",
    "docs/evidence_audit_part3_c2_framework.md",
    "docs/unified_runtime_architecture_freeze.md",
    "output/results/evidence_audit_part2",
    "output/results/evidence_audit_part3",
    "output/results/c1_r1_combined_rerun",
)

DATA_SOURCE_STATUS = {
    "output/results/c1_r1_combined_rerun": "CURRENT-EVIDENCE",
    "output/results/evidence_audit_part2": "CURRENT-EVIDENCE",
    "output/results/evidence_audit_part3": "CURRENT-EVIDENCE",
    "output/results/unified_runtime": "LEGACY-DATA / AUDIT-ONLY",
    "output/results/phase9_step2": "LEGACY-DATA / AUDIT-ONLY",
    "output/results/phase9_step3": "LEGACY-DATA / AUDIT-ONLY",
    "output/results/runtime_attribution": "PENDING-AUDIT-DATA",
    "output/results/phase9_step4_5": "PENDING-AUDIT-DATA",
    "output/results/integration_validation": "PENDING-AUDIT-DATA",
}

EXPECTED_CLAIM_MAP = {
    "Q-01": {"C1.1"},
    "Q-02": {"C1.4"},
    "Q-03": {"C1.2"},
    "Q-04": {"C1.3"},
    "Q-05": {"C3.1"},
    "Q-06": {"C2.4", "C2.6"},
    "Q-07": {"C4.1", "C4.3", "C4.4", "C4.7"},
    "Q-08": {"C2.1"},
    "Q-09": {"C1.5", "C1.8", "C4.5", "C4.6"},
    "Q-10": {"C1.6", "C1.9", "C2.5"},
    "Q-11": {"C2.3"},
}
```

Implement these functions:

```python
def parse_markdown_table(text: str, heading: str) -> list[dict[str, str]]:
    """Return rows from the first pipe table after heading; reject ragged rows."""


def check_headers(repo_root: Path) -> dict[str, object]:
    """Require one marker in the entire file and require it within the first eight lines."""


def check_matrix(register_text: str) -> dict[str, object]:
    """Require Q-01..Q-11, exact Claim mappings, and nonempty nine-column fields."""


def check_reverse_index(register_text: str) -> dict[str, object]:
    """Require every Claim in EXPECTED_CLAIM_MAP to map back to exactly one Q row."""


def check_paths_and_statuses(repo_root: Path, register_text: str) -> dict[str, object]:
    """Require every safe/data path to exist and carry its declared status."""


def build_checks(repo_root: Path) -> dict[str, object]:
    """Return sorted {check_id, status, detail} rows and overall PASS/FAIL."""


def write_output(result: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    output_path.write_text(payload, encoding="utf-8")
```

The CLI must accept:

```python
parser.add_argument("--repo-root", type=Path, default=Path("."))
parser.add_argument("--output", type=Path)
```

It must print the same sorted JSON it writes and return `0` for `PASS`, `1` for `FAIL`.

- [ ] **Step 4: Run the focused tests**

Run:

```bash
python3 -m unittest tests/test_check_x0_5_quarantine.py -v
```

Expected: all six tests pass against their temporary fixtures.

- [ ] **Step 5: Compile and inspect the checker**

Run:

```bash
python3 -m py_compile scripts/check_x0_5_quarantine.py
git diff --check
git diff -- scripts/check_x0_5_quarantine.py tests/test_check_x0_5_quarantine.py
```

Expected: compilation and whitespace checks pass; the diff contains only the checker and its tests.

- [ ] **Step 6: Commit the checker**

Run:

```bash
git add scripts/check_x0_5_quarantine.py tests/test_check_x0_5_quarantine.py
git commit -m "test: add X0.5 quarantine integrity gate"
```

---

### Task 2: Build the Human-Readable Register and Legacy Headers

**Files:**
- Create: `docs/phase_x_x0_5_legacy_narrative_quarantine.md`
- Modify: `paper/draft/method.md`
- Modify: `docs/paper_outline.md`
- Modify: `docs/paper_story_freeze.md`
- Modify: `docs/runtime_framework_spec.md`
- Modify: `docs/phase8_architecture_freeze.md`
- Create: `output/results/evidence_audit_x0_5/quarantine_checks.json`
- Read only: `docs/evidence_audit_part1_claim_inventory.md`

**Interfaces:**
- Consumes: X0, Parts 1–3, the Part 3 architecture freeze, current/legacy data directories, and the Task 1 checker.
- Produces: a writing-facing evidence firewall and a deterministic machine-readable PASS/FAIL artifact.

- [ ] **Step 1: Record the frozen Part 1 hash before editing**

Run:

```bash
sha256sum docs/evidence_audit_part1_claim_inventory.md
```

Record the hash in the implementation notes. Do not stage or modify Part 1.

- [ ] **Step 2: Create the central register**

Use `apply_patch` to create `docs/phase_x_x0_5_legacy_narrative_quarantine.md` with these top-level headings:

```markdown
# Phase X X0.5 — Legacy Narrative Quarantine Register

## 1. Safe Writing Sources
## 2. Purpose, Boundary, and Authority Order
## 3. Document Status Register
## 4. Data Source Status Register
## 5. Claim Propagation Matrix
## 6. Part 1 Claim-ID Reverse Index
## 7. Writing Prohibitions and Release-State Tracker
## 8. Part 7 Handoff and Release Procedure
## 9. Reproduction
```

The Safe Writing Sources section must state:

```text
Numerical or architectural wording may be drafted directly only from the
listed canonical sources. Any other source requires an explicit lookup in
the Claim Propagation Matrix before reuse.
```

List every path in `SAFE_SOURCE_PATHS`, identifying:

- X0 for RQs/scope/contribution hierarchy/estimands;
- Part 2 and its artifacts for C1;
- Part 3 and its artifacts for C2;
- `unified_runtime_architecture_freeze.md` for the figure/interface boundary;
- C1-R1 for unrounded performance observations.

The authority order must be X0 → completed claim audit → Part 1 lineage only → legacy narrative.

- [ ] **Step 3: Add document and data status tables**

The document table must include at minimum:

```text
CANONICAL:
  docs/phase_x_x0_research_freeze.md
  docs/evidence_audit_part2_c1_gpu_runtime.md
  docs/evidence_audit_part3_c2_framework.md
  docs/unified_runtime_architecture_freeze.md

FROZEN-INVENTORY:
  docs/evidence_audit_part1_claim_inventory.md

LEGACY-NON-AUTHORITATIVE:
  paper/draft/method.md
  docs/paper_outline.md
  docs/paper_story_freeze.md
  docs/runtime_framework_spec.md
  docs/phase8_architecture_freeze.md
  docs/baseline_freeze.md
  docs/validation_plan.md
  docs/evidence_matrix.md
```

The data table must reproduce `DATA_SOURCE_STATUS` exactly and include this rule:

Use these columns so the supersession lineage is explicit:

```markdown
| Path | Status | Affected Claim IDs | Allowed Use | Superseded/Controlled By |
```

```text
New C1 manuscript analysis must read from output/results/c1_r1_combined_rerun/
or output/results/evidence_audit_part2/. Legacy data may be used only for
historical auditing, C2.2 lineage repair, or a newly registered Claim whose
consumer explicitly names the legacy protocol and limitation.
```

- [ ] **Step 4: Populate Q-01 through Q-11 with explicit release logic**

Use the nine-column matrix header from Task 1. Populate every field. Apply these release rules:

```text
Q-01/Q-02:
  PERMANENTLY-EXCLUDED. The historical number is never released.
  A new matched component/step claim requires a new Claim ID, a new Part 2
  audit, and Part 7 propagation.

Q-03/Q-04:
  SUPERSEDED-PENDING-PROPAGATION. The old number is never released.
  Part 7 must replace it with C1.2-R1/E1 or C1.3-R1/E2 wording.

Q-05:
  PENDING-PART4. Release only if C3.1 or its replacement receives Part 4
  grade A or B and Part 7 propagates the audited wording.

Q-06:
  PARTIALLY-PROHIBITED. Deterministic construction/O(1) lookup may use C2.4
  wording now; an end-to-end negligible-overhead statement requires a new
  measured Claim graded A/B and Part 7 propagation. C2.6 remains D.

Q-07:
  PENDING-PART5. Release only after Part 5 establishes a true sorter/packer
  treatment contrast, assigns the relevant Claim grade A/B, and Part 7
  propagates the wording.

Q-08:
  SUPERSEDED-PENDING-PROPAGATION. Part 7 replaces the old layer narrative
  with C2.1-R1; RuntimePolicy/GPUExecution remain future extensions.

Q-09:
  PROHIBITED. Release requires a new full-convergence, valid official-test
  quality Claim audited at A/B and Part 7 propagation.

Q-10:
  PROHIBITED/OUT-OF-SCOPE. Sampler-only VRAM requires an isolated A/B Claim;
  bottleneck shift requires unified timing boundaries and A/B audit;
  DDP-ready requires a real distributed execution Claim; general/SOTA
  wording requires separately registered generalization/comparator evidence.
  Each branch also requires Part 7 propagation.

Q-11:
  PROHIBITED. Current Part 7 wording must say the training loop explicitly
  selects the backend. Transparent/automatic/drop-in backend wording requires
  a newly implemented and audited automatic-selection interface.
```

- [ ] **Step 5: Add the reverse index**

Create a two-column table:

```markdown
| Part 1 Claim ID | Quarantine ID |
```

List every Claim from `EXPECTED_CLAIM_MAP` once. Do not add replacement IDs such as C1.2-R1 to Part 1; mention them only in the propagation matrix.

- [ ] **Step 6: Add the identical HTML header block to five files**

Immediately after the first Markdown title, insert exactly:

```html
<!-- LEGACY-NON-AUTHORITATIVE: Phase X X0.5 -->
<!--
Canonical scope: docs/phase_x_x0_research_freeze.md
Quarantine register: docs/phase_x_x0_5_legacy_narrative_quarantine.md
Do not reuse numerical, architecture, or contribution claims without the register.
-->
```

Do not change any other line in those files.

- [ ] **Step 7: Verify header-only legacy diffs**

Run:

```bash
git diff --unified=0 -- \
  paper/draft/method.md \
  docs/paper_outline.md \
  docs/paper_story_freeze.md \
  docs/runtime_framework_spec.md \
  docs/phase8_architecture_freeze.md
```

Expected: each file has one identical six-line HTML comment block after its title and no deletion/replacement.

- [ ] **Step 8: Run the checker twice and compare bytes**

Run:

```bash
python3 scripts/check_x0_5_quarantine.py \
  --repo-root . \
  --output output/results/evidence_audit_x0_5/quarantine_checks.json
cp output/results/evidence_audit_x0_5/quarantine_checks.json /tmp/x0_5_checks_first.json
python3 scripts/check_x0_5_quarantine.py \
  --repo-root . \
  --output output/results/evidence_audit_x0_5/quarantine_checks.json
cmp /tmp/x0_5_checks_first.json \
  output/results/evidence_audit_x0_5/quarantine_checks.json
```

Expected: both runs report `"overall_status": "PASS"` and `cmp` returns zero.

- [ ] **Step 9: Verify Part 1 is unchanged**

Run the same `sha256sum` command from Step 1.

Expected: the hash is identical and `git status --short` does not list Part 1.

- [ ] **Step 10: Run focused validation**

Run:

```bash
python3 -m unittest tests/test_check_x0_5_quarantine.py -v
python3 -m py_compile scripts/check_x0_5_quarantine.py
python3 -m json.tool \
  output/results/evidence_audit_x0_5/quarantine_checks.json >/dev/null
git diff --check
```

Expected: tests, compilation, JSON parsing, and whitespace checks pass.

- [ ] **Step 11: Commit the register and guardrails**

Run:

```bash
git add \
  docs/phase_x_x0_5_legacy_narrative_quarantine.md \
  paper/draft/method.md \
  docs/paper_outline.md \
  docs/paper_story_freeze.md \
  docs/runtime_framework_spec.md \
  docs/phase8_architecture_freeze.md \
  output/results/evidence_audit_x0_5/quarantine_checks.json
git commit -m "docs: quarantine legacy Phase X narratives"
```

---

### Task 3: Update Progress and Project Memory

**Files:**
- Create temporarily: `.memory_payload.json`
- Modify through bouncer: `PROGRESS.md`
- Modify through bouncer: `mukg-memory.json`
- Execute: `utils/memory_bouncer.py`

**Interfaces:**
- Consumes: the completed X0.5 register and PASS checker artifact.
- Produces: current project state identifying X1.5 as the next mandatory gate and preserving X0.5 as a governance milestone.

- [ ] **Step 1: Create the exact memory payload**

Use `apply_patch` to create `.memory_payload.json`:

```json
{
  "active_task": "Phase X X0.5（Legacy Narrative Quarantine）已完成：旧叙事、旧 C1 数据和未裁决 C3/C4 证据已通过中央注册表、五个 HTML 标头和 deterministic checker 隔离。下一步进入 X1.5 文献与新颖性审计。",
  "new_constraints": [
    "X0.5 隔离门：Part 7 前，LEGACY-NON-AUTHORITATIVE 标头不得删除；scripts/check_x0_5_quarantine.py 必须 PASS",
    "Part 1 保持冻结且不回写 SUPERSEDED；Claim→Quarantine 反向追踪由 X0.5 register 维护",
    "新 C1 论文分析只允许读取 output/results/c1_r1_combined_rerun/ 或 evidence_audit_part2；旧 Phase 8/9 C1 数据仅供审计或显式注册用途",
    "透明迁移、自动注入、drop-in backend 等措辞禁止；当前后端由训练循环显式选择"
  ],
  "progress_and_blockers": "已完成 docs/phase_x_x0_5_legacy_narrative_quarantine.md。Q-01–Q-11 均具有 Part 1 Claim 映射、canonical source、release state、audit owner 和明确释放条件；五个旧叙事文件具有 HTML 隔离标头；output/results/evidence_audit_x0_5/quarantine_checks.json 为 deterministic PASS。C3 仍等待 Part 4，C4 仍等待 Part 5，旧 198×/8.5×/5.7×/142× 不得回流。",
  "next_steps": "1. 执行 X1.5 systematic mapping 与 novelty matrix。\\n2. Part 4 审计 C3，并据结果更新 Q-05。\\n3. Part 5 审计 C4，并据结果更新 Q-07。\\n4. 贡献裁决后由 Part 7 按 X0.5 register 传播获准措辞并重新运行 checker。",
  "l2_graph_updates": [
    {
      "type": "entity",
      "name": "PhaseX_X0_5_Legacy_Narrative_Quarantine",
      "entityType": "ResearchGovernanceMilestone",
      "observations": [
        "[2026-08-07] 建立 Q-01–Q-11 的 Claim-ID、证据源、release state 与 Part 7 传播门。",
        "[2026-08-07] Part 1 保持冻结；双向追踪由 X0.5 reverse index 提供。",
        "[2026-08-07] 五个旧叙事文件加入 LEGACY-NON-AUTHORITATIVE HTML 标头，并由 deterministic checker fail-closed 验证。",
        "[2026-08-07] C1-R1 combined rerun 是当前 C1 数据源；Phase 8/9 历史目录为 LEGACY-DATA/AUDIT-ONLY。"
      ]
    },
    {
      "type": "relation",
      "from": "PhaseX_X0_5_Legacy_Narrative_Quarantine",
      "to": "PhaseX_X0_Research_Freeze",
      "relationType": "ENFORCES"
    },
    {
      "type": "relation",
      "from": "PhaseX_X0_5_Legacy_Narrative_Quarantine",
      "to": "Evidence_Audit_Part2_C1_GPU_Runtime",
      "relationType": "ROUTES_TO_CANONICAL_EVIDENCE"
    },
    {
      "type": "relation",
      "from": "PhaseX_X0_5_Legacy_Narrative_Quarantine",
      "to": "Evidence_Audit_Part3_C2_Framework",
      "relationType": "ROUTES_TO_CANONICAL_EVIDENCE"
    }
  ]
}
```

- [ ] **Step 2: Run the memory bouncer**

Run:

```bash
python3 utils/memory_bouncer.py
```

Expected: payload validation passes with four graph updates, the bouncer appends four JSONL records, updates the four `PROGRESS.md` sections, and deletes `.memory_payload.json`.

- [ ] **Step 3: Validate only the newly appended memory tail**

The repository contains two pre-existing malformed historical lines in
`mukg-memory.json`; X0.5 must not repair or reinterpret them. Validate the four
new records only:

```bash
tail -n 4 mukg-memory.json | python3 -c \
  'import json,sys; rows=[json.loads(line) for line in sys.stdin if line.strip()]; assert len(rows)==4; assert rows[0]["name"]=="PhaseX_X0_5_Legacy_Narrative_Quarantine"; print("X0.5 memory tail: PASS")'
```

Expected:

```text
X0.5 memory tail: PASS
```

Validate progress:

```bash
python3 -c 'from pathlib import Path; text=Path("PROGRESS.md").read_text(); required=["Phase X X0.5","X1.5","LEGACY-NON-AUTHORITATIVE","c1_r1_combined_rerun","透明迁移"]; missing=[x for x in required if x not in text]; assert not missing, missing; print("X0.5 progress: PASS")'
```

- [ ] **Step 4: Commit state updates**

Run:

```bash
git add PROGRESS.md mukg-memory.json
git diff --cached --check
git commit -m "docs: record X0.5 quarantine completion"
```

---

### Task 4: Final Scope, Determinism, and Regression Verification

**Files:**
- Verify all files created or modified in Tasks 1–3.
- Do not modify new files unless a verification failure identifies a defect.

**Interfaces:**
- Consumes: all X0.5 implementation commits.
- Produces: evidence that X0.5 is complete without scientific, runtime, or Part 1 drift.

- [ ] **Step 1: Run all CPU-only tests**

Run:

```bash
python3 -m unittest discover -s tests -v
```

Expected: existing Part 3 audit tests and new X0.5 tests pass.

- [ ] **Step 2: Re-run the repository checker**

Run:

```bash
python3 scripts/check_x0_5_quarantine.py \
  --repo-root . \
  --output output/results/evidence_audit_x0_5/quarantine_checks.json
python3 -m json.tool \
  output/results/evidence_audit_x0_5/quarantine_checks.json >/dev/null
```

Expected: `"overall_status": "PASS"`.

- [ ] **Step 3: Verify forbidden modification boundaries**

Run:

```bash
git diff 15c2f7e..HEAD --name-only
git diff 15c2f7e..HEAD -- docs/evidence_audit_part1_claim_inventory.md
git diff 15c2f7e..HEAD -- src paper_assets output/results \
  ':(exclude)output/results/evidence_audit_x0_5/quarantine_checks.json'
```

Expected:

- Part 1 diff is empty.
- No `src/` or `paper_assets/` change exists.
- No pre-existing result artifact changed.
- The only new result artifact is the X0.5 checker JSON.
- Legacy manuscript/document changes are header additions only.

- [ ] **Step 4: Confirm no experiment process was invoked**

Run:

```bash
nvidia-smi
```

Expected: no X0.5 GPU/training process. This is an environment snapshot, not an experiment.

- [ ] **Step 5: Check Git hygiene**

Run:

```bash
git diff --check
git status --short
git log -6 --oneline --decorate
```

Expected: clean worktree and three scoped implementation commits after the design/plan commits.

- [ ] **Step 6: Push only with explicit authorization**

Do not push as part of the implementation plan unless the user explicitly requests it. If authorized:

```bash
git push origin production
```
