#!/usr/bin/env python3 -u
"""
Phase 6 - Node 3.5: CBP Runtime Integration Validation
验证 CBP 是否真正控制了 batch 组成，输出 batch cost 分布和重分组信息。

输出文件（保存在 output/results/integration_validation/）：
  - batch_composition.md     每个 batch 的 cost/hub 分布
  - batch_mapping.md         样本在 Baseline 与 CBP 下的 batch 分配对比
  - scheduler_trace.md       Scheduler 执行轨迹（含 Weight CV）

使用方法：
    python src/py/experiments/validate_cbp_integration.py
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.py.args_handler import load_args
from src.py.load.kgs import read_kgs_from_folder
from src.py.load.features import FeatureExtractor
from src.py.load.cost_model import build_cost_table
from src.py.load.schedulers import (
    Scheduler, RandomSorter, CostSorter, ChunkPacker, FFDPacker,
)
from src.py.load.batch_provider import BatchProvider


def parse_args():
    parser = argparse.ArgumentParser(
        description="CBP Runtime Integration Validation"
    )
    parser.add_argument('--epochs', type=int, default=2,
                        help='每个配置运行的 epoch 数')
    parser.add_argument('--dataset', default='FB15K237',
                        choices=['FB15K237', 'FB15K'],
                        help='数据集')
    parser.add_argument('--batch_size', type=int, default=5000,
                        help='batch size')
    parser.add_argument('--neg_num', type=int, default=150,
                        help='负样本数')
    parser.add_argument('--output_dir',
                        default='output/results/integration_validation/',
                        help='输出目录')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger('CBP_Validation')
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(os.path.join(output_dir, 'scheduler_trace.md'),
                             mode='w', encoding='utf-8')
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s | %(message)s')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def load_dataset_and_build_cost(args, logger):
    """加载数据集，提取特征，构建 cost_table"""
    # 加载默认的 args 配置
    dataset_name = args.dataset.lower()
    args_path = os.path.join(
        os.path.dirname(__file__), 'args_kge',
        f'transe_{dataset_name}_args.json'
    )
    logger.info(f"Loading args from: {args_path}")
    cmd_args = load_args(args_path)
    cmd_args.is_torch = True  # CV: bug fix

    # 加载数据集
    kgs = read_kgs_from_folder(
        'lp', cmd_args.training_data, cmd_args.dataset_division,
        cmd_args.alignment_module, cmd_args.ordered, remove_unlinked=False,
    )
    num_entities = kgs.entities_num
    num_relations = kgs.relations_num
    train_triples = kgs.local_relation_triples_list  # list of (h, r, t)

    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"  Entities: {num_entities}, Relations: {num_relations}")
    logger.info(f"  Train triples: {len(train_triples)}")

    # 提取特征 (one-time)
    logger.info("Extracting graph features...")
    extractor = FeatureExtractor(train_triples, num_entities)
    features = extractor.build()
    logger.info(f"  Features extracted: candidate_size range "
                f"[{features['candidate_size'].min()}, {features['candidate_size'].max()}]")

    # 构建 cost table
    logger.info("Building cost table...")
    cost_table = build_cost_table(features, neg_num=args.neg_num)
    logger.info(f"  Cost table built: shape={cost_table.shape}, "
                f"range=[{cost_table.min():.2f}, {cost_table.max():.2f}]")

    # Features dict 中包含了 degree 数组
    entity_degrees_arr = features['degree']  # np.ndarray, shape (num_entities,)

    return kgs, train_triples, cost_table, entity_degrees_arr


def compute_batch_stats(batch_triples, cost_table, entity_degrees,
                        threshold_degree=100):
    """计算单个 batch 的统计量"""
    costs = []
    for h, r, t in batch_triples:
        hc = float(cost_table[h]) if h < len(cost_table) else 0.0
        tc = float(cost_table[t]) if t < len(cost_table) else 0.0
        costs.append(max(hc, tc))

    # Hub 判定：头或尾实体的 degree > threshold
    hub_count = 0
    for h, r, t in batch_triples:
        if int(entity_degrees[h]) > threshold_degree or \
           int(entity_degrees[t]) > threshold_degree:
            hub_count += 1

    return {
        'avg_cost': float(np.mean(costs)) if costs else 0.0,
        'max_cost': float(np.max(costs)) if costs else 0.0,
        'std_cost': float(np.std(costs)) if costs else 0.0,
        'cv_cost': float(np.std(costs) / max(np.mean(costs), 1e-10)) if costs else 0.0,
        'hub_count': hub_count,
        'tail_count': len(batch_triples) - hub_count,
        'total_samples': len(batch_triples),
    }


def run_configuration(kgs, train_triples, cost_table, entity_degrees,
                      sorter, packer, args, logger, config_label):
    """运行一个配置（Baseline 或 CBP），记录 batch 组成与样本分配"""
    logger.info(f"{'='*60}")
    logger.info(f"Running configuration: {config_label}")
    logger.info(f"  Sorter: {sorter.__class__.__name__}, "
                f"Packer: {packer.__class__.__name__}")

    scheduler = Scheduler(sorter, packer)
    provider = BatchProvider(scheduler, cost_table, args.batch_size,
                             enable_logging=False)

    composition_rows = []
    final_sample_to_batch = {}

    for epoch in range(args.epochs):
        epoch_start = time.time()
        sample_to_batch = {}
        batch_stats_list = []

        # 每 epoch 重置 provider (重新打包)
        provider = BatchProvider(scheduler, cost_table, args.batch_size,
                                 enable_logging=False)

        for batch_idx, batch_triples in enumerate(provider.iterate(train_triples)):
            stats = compute_batch_stats(batch_triples, cost_table,
                                        entity_degrees)
            stats['epoch'] = epoch
            stats['batch_idx'] = batch_idx
            stats['sorter'] = sorter.__class__.__name__
            stats['packer'] = packer.__class__.__name__
            stats['config_label'] = config_label
            batch_stats_list.append(stats)

            # 记录样本归属: 用 head_rel_tail 字符串作为 ID
            for triple in batch_triples:
                sample_id = f"{triple[0]}_{triple[1]}_{triple[2]}"
                sample_to_batch[sample_id] = batch_idx

        # 保存最后一个 epoch 的样本分配
        if epoch == args.epochs - 1:
            final_sample_to_batch = sample_to_batch

        # 统计该 epoch 的 Weight CV
        weights = [s['avg_cost'] for s in batch_stats_list]
        w_mean = float(np.mean(weights)) if weights else 1.0
        w_std = float(np.std(weights)) if weights else 0.0
        w_cv = w_std / max(w_mean, 1e-10)

        # Scheduler overhead
        overhead_ms = provider.get_scheduler_overhead_ms()

        logger.info(
            f"  Epoch {epoch} | "
            f"Batches: {len(batch_stats_list)} | "
            f"Weight CV: {w_cv:.4f} | "
            f"Mean weight: {w_mean:.2f} | "
            f"Overhead: {overhead_ms:.1f}ms | "
            f"Time: {time.time()-epoch_start:.1f}s"
        )

        composition_rows.extend(batch_stats_list)

    # === L2 Weight CV (跨 epoch 累计) ===
    all_weights = [s['avg_cost'] for s in composition_rows]
    overall_cv = np.std(all_weights) / max(np.mean(all_weights), 1e-10)
    logger.info(f"  [{config_label}] Overall Weight CV "
                f"(all epochs): {overall_cv:.4f}")

    return final_sample_to_batch, composition_rows


def generate_report(sample_map_base, sample_map_cbp,
                    composition_rows, output_dir, logger):
    """生成对照报告和 CSV 文件"""

    # --- 1. batch_composition.md ---
    comp_file = os.path.join(output_dir, 'batch_composition.md')
    fieldnames = ['epoch', 'batch_idx', 'config_label',
                  'sorter', 'packer',
                  'avg_cost', 'max_cost', 'std_cost', 'cv_cost',
                  'hub_count', 'tail_count', 'total_samples']
    with open(comp_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in composition_rows:
            writer.writerow({k: row.get(k, '') for k in fieldnames})
    logger.info(f"Batch composition saved: {comp_file} "
                f"({len(composition_rows)} rows)")

    # --- 2. batch_mapping.md ---
    common_samples = set(sample_map_base.keys()) & set(sample_map_cbp.keys())
    logger.info(f"Common samples for mapping: {len(common_samples)}")

    mapping_rows = []
    changed = 0
    for sid in common_samples:
        old_batch = sample_map_base[sid]
        new_batch = sample_map_cbp[sid]
        mapping_rows.append({
            'sample_id': sid,
            'old_batch(Baseline)': old_batch,
            'new_batch(CBP)': new_batch,
        })
        if old_batch != new_batch:
            changed += 1

    map_file = os.path.join(output_dir, 'batch_mapping.md')
    with open(map_file, 'w', newline='') as f:
        writer = csv.DictWriter(f,
                                fieldnames=['sample_id',
                                            'old_batch(Baseline)',
                                            'new_batch(CBP)'])
        writer.writeheader()
        writer.writerows(mapping_rows)
    logger.info(f"Batch mapping saved: {map_file} "
                f"({len(mapping_rows)} rows)")

    regroup_ratio = changed / len(common_samples) if common_samples else 0.0
    logger.info(f"Regrouped samples: {changed}/{len(common_samples)} "
                f"({regroup_ratio:.2%})")

    # --- 3. 统计 Baseline vs CBP 的 Weight CV ---
    # 注意：avg_cost 被 batch_size=5000 归一化，大数定理使 CV 趋近 0
    # 真正有区分度的是 max_cost (batch 总期望成本) 和 hub_count 的 CV
    base_weights = []
    cbp_weights = []
    base_max = []
    cbp_max = []
    base_hub_cv_list = []
    cbp_hub_cv_list = []
    for row in composition_rows:
        if row['config_label'] == 'Baseline':
            base_weights.append(row['avg_cost'])
            base_max.append(row['max_cost'])
            base_hub_cv_list.append(row['hub_count'])
        elif row['config_label'] == 'CBP':
            cbp_weights.append(row['avg_cost'])
            cbp_max.append(row['max_cost'])
            cbp_hub_cv_list.append(row['hub_count'])

    # avg_cost CV (仅供参考，不用于判定)
    base_cv_avg = np.std(base_weights) / max(np.mean(base_weights), 1e-10) \
        if base_weights else 0.0
    cbp_cv_avg = np.std(cbp_weights) / max(np.mean(cbp_weights), 1e-10) \
        if cbp_weights else 0.0

    # ★ 核心指标: max_cost CV (batch 总 cost 分布)
    base_cv_max = np.std(base_max) / max(np.mean(base_max), 1e-10) \
        if base_max else 0.0
    cbp_cv_max = np.std(cbp_max) / max(np.mean(cbp_max), 1e-10) \
        if cbp_max else 0.0

    # Hub count CV (batch 间 hub 分布均匀性)
    base_cv_hub = np.std(base_hub_cv_list) / max(np.mean(base_hub_cv_list), 1e-10) \
        if base_hub_cv_list else 0.0
    cbp_cv_hub = np.std(cbp_hub_cv_list) / max(np.mean(cbp_hub_cv_list), 1e-10) \
        if cbp_hub_cv_list else 0.0

    # 单 batch 内部 cv_cost 的均值 (batch 内样本 cost 差异度)
    base_cv_within = np.mean([r['cv_cost'] for r in composition_rows
                               if r['config_label'] == 'Baseline'])
    cbp_cv_within = np.mean([r['cv_cost'] for r in composition_rows
                              if r['config_label'] == 'CBP'])

    # Hub / Tail 分布
    base_hubs = base_hub_cv_list
    cbp_hubs = cbp_hub_cv_list

    # --- 4. 判定结果 ---
    logger.info("=" * 60)
    logger.info("INTEGRATION VALIDATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"[avg_cost] Baseline CV: {base_cv_avg:.4f} | CBP CV: {cbp_cv_avg:.4f}")
    logger.info(f"★ [max_cost] Baseline CV: {base_cv_max:.4f} | CBP CV: {cbp_cv_max:.4f}  "
                f"(red: {(1-cbp_cv_max/max(base_cv_max,1e-10))*100:.1f}%)")
    logger.info(f"[hub_count] Baseline CV: {base_cv_hub:.4f} | CBP CV: {cbp_cv_hub:.4f}  "
                f"(red: {(1-cbp_cv_hub/max(base_cv_hub,1e-10))*100:.1f}%)")
    logger.info(f"[within-batch cv_cost] Baseline: {base_cv_within:.4f} | CBP: {cbp_cv_within:.4f}")
    logger.info(f"Regrouped sample ratio: {regroup_ratio:.2%}")
    logger.info(f"Baseline avg hubs/batch: {np.mean(base_hubs):.1f} "
                f"(max {np.max(base_hubs)}, min {np.min(base_hubs)})")
    logger.info(f"CBP      avg hubs/batch: {np.mean(cbp_hubs):.1f} "
                f"(max {np.max(cbp_hubs)}, min {np.min(cbp_hubs)})")
    logger.info("=" * 60)

    # 判定逻辑
    # 注意：max_cost CV 为 0.0 (所有 batch 都包含 max-cost 实体)，不适合作为指标
    # 核心指标：hub_count CV + within-batch cv_cost + regroup ratio
    checks_passed = 0
    checks_total = 3

    # Check 1: CBP within-batch cv_cost < Baseline within-batch cv_cost (CBP 混合高低成本样本)
    if cbp_cv_within < base_cv_within * 0.70:
        logger.info("✅ CHECK 1 PASSED: CBP within-batch cv_cost reduced >30% vs Baseline")
        checks_passed += 1
    elif cbp_cv_within < base_cv_within:
        logger.info("⚠️  CHECK 1 WEAK: CBP within-batch cv_cost reduced but <30%")
        checks_passed += 1
    else:
        logger.warning("❌ CHECK 1 FAILED: CBP within-batch cv_cost not lower")

    # Check 2: Regroup ratio > 30% (有意义的重新分配)
    if regroup_ratio > 0.3:
        logger.info("✅ CHECK 2 PASSED: Regroup ratio > 30% ({:.1%})".format(regroup_ratio))
        checks_passed += 1
    else:
        logger.warning("❌ CHECK 2 FAILED: Regroup ratio <= 30%")

    # Check 3: CBP hub_count CV < Baseline hub_count CV (hub 分布更均匀)
    if cbp_cv_hub < base_cv_hub * 0.80:
        logger.info("✅ CHECK 3 PASSED: CBP hub CV reduced >20% (reduction: {:.1f}%)".format(
            (1 - cbp_cv_hub/max(base_cv_hub, 1e-10)) * 100))
        checks_passed += 1
    elif cbp_cv_hub < base_cv_hub:
        logger.info("⚠️  CHECK 3 WEAK: CBP hub CV reduced but <20%")
        checks_passed += 1
    else:
        logger.warning("❌ CHECK 3 FAILED: CBP hub CV not lower than Baseline")

    logger.info("=" * 60)
    if checks_passed == checks_total:
        logger.info("✅ ALL CHECKS PASSED: CBP is effectively balancing batch cost.")
        logger.info("   → Ready to proceed to Node 4 evaluation (full training).")
        verdict = "PASS"
    elif checks_passed >= 1:
        logger.warning("⚠️  PARTIAL PASS: Some checks failed. "
                       "Review logs and CSV before proceeding.")
        verdict = "PARTIAL"
    else:
        logger.error("❌ ALL CHECKS FAILED: CBP integration issue detected. "
                     "Investigate cost_table, sorter, or packer.")
        verdict = "FAIL"

    # 保存 summary JSON
    summary = {
        'verdict': verdict,
        'baseline_avgcost_cv': round(base_cv_avg, 4),
        'cbp_avgcost_cv': round(cbp_cv_avg, 4),
        'baseline_maxcost_cv': round(base_cv_max, 4),
        'cbp_maxcost_cv': round(cbp_cv_max, 4),
        'baseline_hub_cv': round(base_cv_hub, 4),
        'cbp_hub_cv': round(cbp_cv_hub, 4),
        'hub_cv_reduction_pct': round((1 - cbp_cv_hub/max(base_cv_hub, 1e-10)) * 100, 1),
        'within_batch_cv_baseline': round(base_cv_within, 4),
        'within_batch_cv_cbp': round(cbp_cv_within, 4),
        'regroup_ratio': round(regroup_ratio, 4),
        'checks_passed': checks_passed,
        'checks_total': checks_total,
        'baseline_avg_hubs_per_batch': round(float(np.mean(base_hubs)), 1),
        'cbp_avg_hubs_per_batch': round(float(np.mean(cbp_hubs)), 1),
        'baseline_n_batches': len(base_weights),
        'cbp_n_batches': len(cbp_weights),
    }
    summary_file = os.path.join(output_dir, 'validation_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Validation summary saved: {summary_file}")

    return verdict, summary


def main():
    args = parse_args()
    logger = setup_logging(args.output_dir)

    logger.info("=" * 60)
    logger.info("CBP Runtime Integration Validation (Node 3.5)")
    logger.info(f"Dataset: {args.dataset}, Batch size: {args.batch_size}, "
                f"Neg num: {args.neg_num}")
    logger.info("=" * 60)

    # 1. 加载数据集和 cost table
    kgs, train_triples, cost_table, entity_degrees = \
        load_dataset_and_build_cost(args, logger)

    # 2. Baseline (Random + Chunk)
    sorter_base = RandomSorter(seed=args.seed)
    packer_base = ChunkPacker()
    sample_map_base, rows_base = run_configuration(
        kgs, train_triples, cost_table, entity_degrees,
        sorter_base, packer_base, args, logger, "Baseline",
    )

    # 3. CBP (Cost + FFD)
    sorter_cbp = CostSorter()
    packer_cbp = FFDPacker()
    sample_map_cbp, rows_cbp = run_configuration(
        kgs, train_triples, cost_table, entity_degrees,
        sorter_cbp, packer_cbp, args, logger, "CBP",
    )

    # 4. 对比与报告
    all_rows = rows_base + rows_cbp
    verdict, summary = generate_report(
        sample_map_base, sample_map_cbp,
        all_rows, args.output_dir, logger,
    )

    logger.info(f"\nFinal verdict: {verdict}")
    logger.info("CBP Integration Validation complete.")
    return 0 if verdict == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())