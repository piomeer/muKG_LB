#!/usr/bin/env python3
"""Profile the true ID Mapping / data loading time (one-shot, no training)."""

import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.py.args_handler import load_args
from src.py.load.kgs import read_kgs_from_folder

# Load args to get data path
curPath = os.path.abspath(os.path.dirname(__file__)) + "/src/py/experiments"
model_name = 'transe'
args = load_args(curPath + "/args_kge/" + model_name + r"_fb15k237_args.json")

print("=" * 60)
print(f"Data folder: {args.training_data}")
print(f"batch_size={args.batch_size}, neg_triple_num={args.neg_triple_num}")
print("=" * 60)

# === Profile the ONE TRUE data loading + ID mapping ===
t_start = time.time()

kgs = read_kgs_from_folder(
    'lp',
    args.training_data,
    args.dataset_division,
    args.alignment_module,
    args.ordered,
    remove_unlinked=False
)

t_end = time.time()

print("=" * 60)
print(f"[系统初始化] 真·原始三元组→ID映射+KG构建总耗时: {t_end - t_start:.4f} 秒")
print(f"  → 训练三元组数: {len(kgs.relation_triples_list) if hasattr(kgs, 'relation_triples_list') else kgs.relation_triples_num}")
print(f"  → 实体数: {kgs.entities_num}")
print(f"  → 关系数: {kgs.relations_num}")
print("=" * 60)
print("\n[完成] 数据加载分析完毕，可以 Ctrl+C 退出（但脚本已结束）。")
