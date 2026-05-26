#!/usr/bin/env python3
"""
======================================================================
MuKG Memory Bouncer — 物理级记忆流转门禁系统
======================================================================

【核心职责】
在执行任务收尾前，Cline 必须先生成 .memory_payload.json，再由本脚本执行：
  1. 强制校验 payload 完整性
  2. L2 记忆合并（JSONL 追加到 mukg-memory.json）
  3. L3 进度沉淀（追加到 PROGRESS.md）
  4. 自动 Git 同步（环境感知分支策略）
  5. Python 语法门禁（py_compile 快速扫描）
  6. 清理 payload 文件

【退出码约定】
  - Exit 0: 成功，允许流转
  - Exit 1: 校验失败或流程出错，拒绝流转
======================================================================
"""

import json
import sys
import os
import subprocess
import py_compile
import tempfile
import re
from datetime import datetime, timezone
from pathlib import Path


# ====================================================================
# 常量
# ====================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAYLOAD_FILE = PROJECT_ROOT / ".memory_payload.json"
MEMORY_FILE = PROJECT_ROOT / "mukg-memory.json"
PROGRESS_FILE = PROJECT_ROOT / "PROGRESS.md"
ENV_IDENTITY_FILE = PROJECT_ROOT / "env_identity.json"

REQUIRED_FIELDS = [
    "l1_reflection",
    "l2_graph_updates",
    "l3_progress_notes",
    "next_steps",
    "commit_message",
]

from datetime import timedelta

JST = timezone.utc  # We'll format manually for JST


def get_jst_timestamp() -> str:
    """返回 JST 时间戳字符串 [YYYY-MM-DD HH:MM:SS]"""
    utc_now = datetime.now(timezone.utc)
    jst_now = utc_now.astimezone(timezone(timedelta(hours=9)))
    return jst_now.strftime("%Y-%m-%d %H:%M:%S")


def get_jst_date() -> str:
    """返回 JST 日期字符串 [YYYY-MM-DD]"""
    utc_now = datetime.now(timezone.utc)
    jst_now = utc_now.astimezone(timezone(timedelta(hours=9)))
    return jst_now.strftime("%Y-%m-%d")


# ====================================================================
# Step 1: 强制校验（Field Validation）
# ====================================================================
def validate_payload(payload: dict) -> None:
    """校验 payload 中所有必需字段是否完整。缺失任一字段 → sys.exit(1)"""
    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing:
        print(f"[BOUNCER] ❌ PAYLOAD 字段缺失: {missing}", file=sys.stderr)
        print(f"[BOUNCER]    必需字段: {REQUIRED_FIELDS}", file=sys.stderr)
        sys.exit(1)

    # 校验 l2_graph_updates 类型
    if not isinstance(payload["l2_graph_updates"], list):
        print("[BOUNCER] ❌ l2_graph_updates 必须是数组类型", file=sys.stderr)
        sys.exit(1)

    # 校验每个更新条目
    for i, entry in enumerate(payload["l2_graph_updates"]):
        if "type" not in entry:
            print(f"[BOUNCER] ❌ l2_graph_updates[{i}] 缺少 type 字段", file=sys.stderr)
            sys.exit(1)
        if entry["type"] not in ("entity", "relation"):
            print(
                f"[BOUNCER] ❌ l2_graph_updates[{i}] type 必须是 'entity' 或 'relation'",
                file=sys.stderr,
            )
            sys.exit(1)
        if entry["type"] == "entity":
            for field in ("name", "entityType", "observations"):
                if field not in entry:
                    print(
                        f"[BOUNCER] ❌ l2_graph_updates[{i}] (entity) 缺少 {field}",
                        file=sys.stderr,
                    )
                    sys.exit(1)
        elif entry["type"] == "relation":
            for field in ("from", "to", "relationType"):
                if field not in entry:
                    print(
                        f"[BOUNCER] ❌ l2_graph_updates[{i}] (relation) 缺少 {field}",
                        file=sys.stderr,
                    )
                    sys.exit(1)

    print(f"[BOUNCER] ✅ Payload 校验通过 ({len(payload['l2_graph_updates'])} 条更新)")


# ====================================================================
# Step 2: L2 记忆合并（JSONL Append）
# ====================================================================
def merge_l2_memory(payload: dict) -> None:
    """
    将 l2_graph_updates 以 JSONL 格式追加到 mukg-memory.json。

    现有文件格式为 JSON Lines（每行一个独立 JSON 对象，无外层括号）。
    新增条目时确保每行是紧凑的 JSON（无内部换行）。
    """
    updates = payload["l2_graph_updates"]
    if not updates:
        print("[BOUNCER] ⏭️  L2 无更新，跳过记忆合并")
        return

    timestamp = get_jst_timestamp()
    lines_appended = 0

    with open(MEMORY_FILE, "a", encoding="utf-8") as f:
        for entry in updates:
            # 构建 metadata
            record = {
                "_meta": {
                    "timestamp": timestamp,
                    "source": "memory_bouncer",
                    "commit_message": payload.get("commit_message", ""),
                }
            }

            if entry["type"] == "entity":
                record["type"] = "entity"
                record["name"] = entry["name"]
                record["entityType"] = entry["entityType"]
                record["observations"] = entry["observations"]
            elif entry["type"] == "relation":
                record["type"] = "relation"
                record["from"] = entry["from"]
                record["to"] = entry["to"]
                record["relationType"] = entry["relationType"]

            # 强制无内部换行的紧凑 JSON
            json_line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            f.write(json_line + "\n")
            lines_appended += 1

    print(f"[BOUNCER] ✅ L2 记忆合并完成: 追加 {lines_appended} 行到 {MEMORY_FILE.name}")


# ====================================================================
# Step 3: L3 进度沉淀（Append to PROGRESS.md）
# ====================================================================
def merge_l3_progress(payload: dict) -> None:
    """
    将 l3_progress_notes 和 next_steps 格式化后追加到 PROGRESS.md 的适当位置。
    追加到 `## 7. 待办事项 (Backlog)` 之后（文件末尾）。
    """
    notes = payload.get("l3_progress_notes", "").strip()
    next_steps = payload.get("next_steps", "")
    commit_msg = payload.get("commit_message", "")
    date_str = get_jst_date()
    env_tag = detect_environment()

    if isinstance(next_steps, list):
        next_steps = "\n".join(f"- {s}" for s in next_steps)

    # 构建追加区块
    section = f"""
---

## 8. 自动记忆流转记录 [({date_str})]

**环境**: `{env_tag}`

### 本次任务总结
{notes}

### 下一步计划
{next_steps}

### Git 提交
```
{commit_msg}
```
"""

    with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
        f.write(section)

    print(f"[BOUNCER] ✅ L3 进度沉淀完成: 追加到 {PROGRESS_FILE.name}")


# ====================================================================
# Step 4: 环境感知分支策略
# ====================================================================
def detect_environment() -> str:
    """读取 env_identity.json 判断当前环境"""
    try:
        with open(ENV_IDENTITY_FILE, "r", encoding="utf-8") as f:
            env_data = json.load(f)
        env = env_data.get("env", "unknown")
        return env
    except (FileNotFoundError, json.JSONDecodeError):
        return "unknown"


def determine_branch() -> str:
    """
    根据环境决定目标分支：
      - wsl   → main
      - node4 → production
      - node6 → production
      - 其他   → main（安全默认）
    """
    env = detect_environment()
    if env == "local_wsl":
        return "main"
    elif env in ("server_node4", "server_node6"):
        return "production"
    else:
        print(f"[BOUNCER] ⚠️  未知环境 '{env}'，默认推送到 main 分支")
        return "main"


# ====================================================================
# Step 5: Python 语法门禁（py_compile 扫描）
# ====================================================================
def check_python_syntax() -> None:
    """
    扫描 Git 暂存区中的所有 .py 文件，使用 py_compile 进行基本语法检查。
    如果发现 SyntaxError，直接报错并拒绝流转。
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        staged_files = result.stdout.strip().split("\n")
    except subprocess.SubprocessError as e:
        print(f"[BOUNCER] ⚠️  无法获取暂存区文件列表: {e}", file=sys.stderr)
        print("[BOUNCER]    跳过语法检查（不阻止流转）")
        return

    py_files = [f for f in staged_files if f.endswith(".py") and os.path.isfile(f)]

    if not py_files:
        print("[BOUNCER] ⏭️  暂存区无 .py 文件，跳过语法检查")
        return

    errors = []
    for py_file in py_files:
        try:
            py_compile.compile(py_file, doraise=True)
            print(f"[BOUNCER]   ✅ 语法通过: {py_file}")
        except py_compile.PyCompileError as e:
            errors.append((py_file, str(e)))
            print(f"[BOUNCER]   ❌ 语法错误: {py_file} — {e}", file=sys.stderr)

    if errors:
        print("\n[BOUNCER] 🔴 语法门禁拦截！以下文件存在语法错误：", file=sys.stderr)
        for fname, err in errors:
            print(f"  - {fname}: {err}", file=sys.stderr)
        print("\n[BOUNCER] 请修复上述语法错误后重试。流转已拒绝。", file=sys.stderr)
        sys.exit(1)

    print(f"[BOUNCER] ✅ Python 语法门禁通过: {len(py_files)} 个文件")


# ====================================================================
# Step 6: 自动 Git 同步
# ====================================================================
def auto_git_sync(commit_message: str) -> None:
    """
    执行 git add . → git commit → git push origin <branch>。
    分支根据环境自动判断。
    """
    branch = determine_branch()
    env = detect_environment()

    try:
        # 1. git add
        print(f"[BOUNCER] 🔄 执行 git add .")
        subprocess.run(
            ["git", "add", "."],
            capture_output=True,
            text=True,
            check=True,
            cwd=PROJECT_ROOT,
        )

        # 2. git commit
        print(f"[BOUNCER] 🔄 执行 git commit -m \"{commit_message}\"")
        result = subprocess.run(
            ["git", "commit", "-m", commit_message],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        if result.returncode == 0:
            print(f"[BOUNCER]   ✅ Commit 成功")
        elif "nothing to commit" in result.stdout or "nothing to commit" in result.stderr:
            print("[BOUNCER]   ⏭️  无变更需要提交")
        else:
            print(f"[BOUNCER]   ⚠️  Commit 可能异常: {result.stderr.strip()}")

        # 3. git push
        print(f"[BOUNCER] 🔄 执行 git push origin {branch}")
        push_result = subprocess.run(
            ["git", "push", "origin", branch],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        if push_result.returncode == 0:
            print(f"[BOUNCER]   ✅ Push 成功 → origin/{branch}")
        else:
            # 尝试设置 upstream 并重试
            stderr = push_result.stderr.strip()
            if "has no upstream" in stderr or "upstream" in stderr:
                print(f"[BOUNCER]   ⚠️  无上游分支，尝试 git push -u origin {branch}")
                retry = subprocess.run(
                    ["git", "push", "-u", "origin", branch],
                    capture_output=True,
                    text=True,
                    cwd=PROJECT_ROOT,
                )
                if retry.returncode == 0:
                    print(f"[BOUNCER]   ✅ Push 成功（上游分支已设置）→ origin/{branch}")
                else:
                    print(
                        f"[BOUNCER]   ❌ Push 失败: {retry.stderr.strip()}",
                        file=sys.stderr,
                    )
                    print("[BOUNCER]     请手动检查 Git 配置。流转继续进行。")
            else:
                print(f"[BOUNCER]   ⚠️  Push 异常: {stderr}")
                print("[BOUNCER]     流转继续进行（非致命异常）")

        # 记录 Git 信息
        print(f"[BOUNCER] ✅ Git 同步完成 (环境: {env} → 分支: {branch})")

    except subprocess.CalledProcessError as e:
        print(f"[BOUNCER] ❌ Git 操作失败: {e.stderr}", file=sys.stderr)
        print("[BOUNCER]    流转继续进行（Git 失败不阻止流程）")


# ====================================================================
# Step 7: 清理 Payload
# ====================================================================
def cleanup_payload() -> None:
    """删除 .memory_payload.json 文件"""
    if PAYLOAD_FILE.exists():
        PAYLOAD_FILE.unlink()
        print(f"[BOUNCER] 🧹 已删除 {PAYLOAD_FILE.name}")
    else:
        print(f"[BOUNCER] ⏭️  {PAYLOAD_FILE.name} 不存在，跳过清理")


# ====================================================================
# Main Entry Point
# ====================================================================
def main():
    print("=" * 60)
    print("  MuKG Memory Bouncer — 记忆流转门禁系统")
    print(f"  工作目录: {PROJECT_ROOT}")
    print("=" * 60)

    # 检查 payload 文件是否存在
    if not PAYLOAD_FILE.exists():
        print(
            f"[BOUNCER] ❌ 未找到 {PAYLOAD_FILE.name}！",
            file=sys.stderr,
        )
        print(
            "[BOUNCER]    请先生成 .memory_payload.json 再执行此脚本。",
            file=sys.stderr,
        )
        sys.exit(1)

    # 读取 payload
    try:
        with open(PAYLOAD_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)
        print(f"[BOUNCER] 📄 已读取 {PAYLOAD_FILE.name}")
    except json.JSONDecodeError as e:
        print(f"[BOUNCER] ❌ Payload JSON 解析失败: {e}", file=sys.stderr)
        sys.exit(1)

    # Step 1: 强制校验
    print("\n── Step 1/5: 强制校验 ──")
    validate_payload(payload)

    # Step 2: L2 记忆合并
    print("\n── Step 2/5: L2 记忆合并 ──")
    merge_l2_memory(payload)

    # Step 3: L3 进度沉淀
    print("\n── Step 3/5: L3 进度沉淀 ──")
    merge_l3_progress(payload)

    # Step 4: Python 语法门禁
    print("\n── Step 4/5: Python 语法门禁 ──")
    check_python_syntax()

    # Step 5: 自动 Git 同步 + 清理
    print("\n── Step 5/5: 自动 Git 同步 ──")
    auto_git_sync(payload.get("commit_message", "Auto-commit: memory_bouncer"))

    # 清理
    cleanup_payload()

    print("\n" + "=" * 60)
    print("  ✅ 记忆流转完成！Exit Code 0")
    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()
