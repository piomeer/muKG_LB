#!/usr/bin/env python3
"""
======================================================================
MuKG Memory Bouncer — 物理级记忆流转门禁系统 (v4)
======================================================================
【核心职责】
在执行任务收尾前，Cline 必须先生成 .memory_payload.json，再由本脚本执行：
  1. 强制校验 payload 完整性（新 L3 切面状态机 Schema）
  2. L2 记忆合并（JSONL 追加到 mukg-memory.json）
  3. L3 进度沉淀（纯切面模式：精准替换 PROGRESS.md 的 4 个固定板块）
  4. 清理 payload 文件

【L3 切面状态机设计模式】
  - PROGRESS.md 永远只包含 4 个固定板块：## 1., ## 2., ## 3., ## 4.
  - 每次执行时，用 Payload 中的新内容精准替换/追加到对应板块下方。
  - 不再有永久锚点 / 历史归档 / 追加记录模式。

【v4 重大变更】
  - Payload Schema 改为 L3 切面模式：active_task, new_constraints, progress_and_blockers, next_steps
  - L3 merge 改为正则精确定位 4 板块 + 替换内容
  - new_constraints 支持追加到 ## 2. 现有约束列表下方（L1 宪法 → L3 动态映射）
  - 移除永久锚点 / 历史归档逻辑
  - 移除 Git 自动提交/推送逻辑

【退出码约定】
  - Exit 0: 成功，允许流转
  - Exit 1: 校验失败或流程出错，拒绝流转
======================================================================
"""

import json
import sys
import os
import re
from datetime import datetime, timezone, timedelta
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
    "active_task",
    "new_constraints",
    "progress_and_blockers",
    "next_steps",
    "l2_graph_updates",
]


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

    if not isinstance(payload["new_constraints"], list):
        print("[BOUNCER] ❌ new_constraints 必须是数组类型", file=sys.stderr)
        sys.exit(1)

    if not isinstance(payload["l2_graph_updates"], list):
        print("[BOUNCER] ❌ l2_graph_updates 必须是数组类型", file=sys.stderr)
        sys.exit(1)

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

    print(f"[BOUNCER] ✅ Payload 校验通过 ({len(payload['l2_graph_updates'])} 条 L2 更新)")


# ====================================================================
# Step 2: L2 记忆合并（JSONL Append）
# ====================================================================
def merge_l2_memory(payload: dict) -> None:
    """
    将 l2_graph_updates 以 JSONL 格式追加到 mukg-memory.json。
    每行是紧凑 JSON（无内部换行），带 _meta 元信息。
    """
    updates = payload["l2_graph_updates"]
    if not updates:
        print("[BOUNCER] ⏭️  L2 无更新，跳过记忆合并")
        return

    timestamp = get_jst_timestamp()
    commit_msg = payload.get("commit_message", "")
    lines_appended = 0

    with open(MEMORY_FILE, "a", encoding="utf-8") as f:
        for entry in updates:
            record = {
                "_meta": {
                    "timestamp": timestamp,
                    "source": "memory_bouncer",
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

            json_line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            f.write(json_line + "\n")
            lines_appended += 1

    print(f"[BOUNCER] ✅ L2 记忆合并完成: 追加 {lines_appended} 行到 {MEMORY_FILE.name}")


# ====================================================================
# Step 3: L3 进度沉淀 — 纯切面模式
# ====================================================================
def _build_constraints_block(payload: dict) -> str:
    """
    从现有 ## 2. 内容和 new_constraints 构建新的约束板块。
    如果 new_constraints 为空，保持原内容不变。
    不为空时，将新约束作为列表项追加到原内容下方。
    """
    new_constraints = payload.get("new_constraints", [])
    if not new_constraints:
        return None  # 表示不修改 ## 2.

    constraints_lines = []
    for c in new_constraints:
        constraints_lines.append(f"- **{c}**")
    return "\n".join(constraints_lines)


def _replace_section(content: str, heading: str, new_body: str) -> str:
    """
    用 new_body 替换指定标题下方的全部内容（直到下一个同级标题或文件末尾）。
    保留标题行本身，只替换标题下方的内容。

    参数：
      content: PROGRESS.md 全文
      heading: 如 "## 1."、"## 2." 等
      new_body: 要替换的新内容（不含标题）
    """
    # 转义标题中的特殊正则字符
    escaped_heading = re.escape(heading)

    # 匹配模式：标题行 + 其后内容直到下一个 ## 或文件末尾
    # 分组1：标题行 + 换行
    # 分组2：标题下方内容（被替换的部分）
    pattern = rf"({escaped_heading}[^\n]*\n)(.*?)(?=\n## |\Z)"

    def replacement(match):
        # match.group(1) = 标题行 + 换行
        # match.group(2) = 原标题下方的旧内容
        return match.group(1) + new_body.rstrip() + "\n"

    replaced_count = 0
    new_content, count = re.subn(pattern, replacement, content, count=1, flags=re.DOTALL)
    if count == 0:
        # 如果没找到匹配（标题不存在），在文件末尾追加
        print(f"[BOUNCER] ⚠️  未找到标题 '{heading}'，将在文件末尾追加")
        new_content = content.rstrip() + f"\n\n{heading}\n" + new_body.rstrip() + "\n"
    else:
        replaced_count = count

    return new_content


def merge_l3_progress(payload: dict) -> None:
    """
    纯切面模式写入 PROGRESS.md。

    逻辑：
      1. 读取 PROGRESS.md 全文
      2. 用 active_task 替换 ## 1. 下方的内容
      3. 若 new_constraints 非空，追加到 ## 2. 内容下方（L1 → L3 映射）
      4. 用 progress_and_blockers 替换 ## 3. 下方的内容
      5. 用 next_steps 替换 ## 4. 下方的内容
      6. 写回文件
    """
    active_task = payload.get("active_task", "[待分配]").strip()
    progress_and_blockers = payload.get("progress_and_blockers", "[待分配]").strip()
    next_steps = payload.get("next_steps", "[待分配]").strip()
    new_constraints = payload.get("new_constraints", [])

    if not PROGRESS_FILE.exists():
        print(f"[BOUNCER] ❌ 未找到 {PROGRESS_FILE.name}！", file=sys.stderr)
        sys.exit(1)

    with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    # --- 替换 ## 1. 当前活动目标 ---
    content = _replace_section(content, "## 1.", active_task)

    # --- 处理 ## 2. 活跃约束提醒（追加模式） ---
    if new_constraints:
        # 提取 ## 2. 下方现有的内容
        pattern_2 = r"(## 2\.[^\n]*\n)(.*?)(?=\n## |\Z)"
        m = re.search(pattern_2, content, re.DOTALL)
        if m:
            existing_body = m.group(2).strip()
            # 构建要追加的新约束行
            extra_lines = []
            for c in new_constraints:
                extra_lines.append(f"- **{c}**  *(自动映射自 L1 宪法)*")
            extra_block = "\n".join(extra_lines)
            # 在现有内容后追加
            new_body_2 = existing_body + "\n" + extra_block
            content = _replace_section(content, "## 2.", new_body_2)
            print(f"[BOUNCER] 📋 追加 {len(new_constraints)} 条新约束到 ## 2.")
    # 如果 new_constraints 为空，保持 ## 2. 不变

    # --- 替换 ## 3. 当前进度与卡点 ---
    content = _replace_section(content, "## 3.", progress_and_blockers)

    # --- 替换 ## 4. 下一步计划 ---
    content = _replace_section(content, "## 4.", next_steps)

    # --- 写回文件 ---
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[BOUNCER] ✅ L3 进度沉淀完成: {PROGRESS_FILE.name}")
    print(f"[BOUNCER]     ## 1. → active_task")
    if new_constraints:
        print(f"[BOUNCER]     ## 2. → 追加 {len(new_constraints)} 条新约束")
    print(f"[BOUNCER]     ## 3. → progress_and_blockers")
    print(f"[BOUNCER]     ## 4. → next_steps")


# ====================================================================
# Step 4: 环境检测
# ====================================================================
def detect_environment() -> str:
    """读取 env_identity.json 判断当前环境"""
    try:
        with open(ENV_IDENTITY_FILE, "r", encoding="utf-8") as f:
            env_data = json.load(f)
        return env_data.get("env", "unknown")
    except (FileNotFoundError, json.JSONDecodeError):
        return "unknown"


# ====================================================================
# Step 5: 清理 Payload
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
    print("  MuKG Memory Bouncer — 记忆流转门禁系统 v4")
    print(f"  [L3切面状态机模式]")
    print(f"  工作目录: {PROJECT_ROOT}")
    print("=" * 60)

    # 检查 payload 文件
    if not PAYLOAD_FILE.exists():
        print(f"[BOUNCER] ❌ 未找到 {PAYLOAD_FILE.name}！", file=sys.stderr)
        print("[BOUNCER]    请先生成 .memory_payload.json 再执行此脚本。", file=sys.stderr)
        sys.exit(1)

    try:
        with open(PAYLOAD_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)
        print(f"[BOUNCER] 📄 已读取 {PAYLOAD_FILE.name}")
    except json.JSONDecodeError as e:
        print(f"[BOUNCER] ❌ Payload JSON 解析失败: {e}", file=sys.stderr)
        sys.exit(1)

    # Step 1: 强制校验
    print("\n── Step 1/4: 强制校验 ──")
    validate_payload(payload)

    # Step 2: L2 记忆合并
    print("\n── Step 2/4: L2 记忆合并 ──")
    merge_l2_memory(payload)

    # Step 3: L3 进度沉淀（纯切面模式）
    print("\n── Step 3/4: L3 进度沉淀（切面替换）──")
    merge_l3_progress(payload)

    # Step 4: 清理 Payload
    print("\n── Step 4/4: 清理 ──")
    cleanup_payload()

    print("\n" + "=" * 60)
    print("  ✅ 记忆流转完成！Exit Code 0")
    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()
