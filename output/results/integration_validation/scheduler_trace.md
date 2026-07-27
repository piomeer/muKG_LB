2026-07-20 16:26:34,159 | ============================================================
2026-07-20 16:26:34,159 | CBP Runtime Integration Validation (Node 3.5)
2026-07-20 16:26:34,159 | Dataset: FB15K237, Batch size: 5000, Neg num: 150
2026-07-20 16:26:34,159 | ============================================================
2026-07-20 16:26:34,159 | Loading args from: /home/hma/muKG_LB/src/py/experiments/args_kge/transe_fb15k237_args.json
2026-07-20 16:26:35,614 | Dataset: FB15K237
2026-07-20 16:26:35,614 |   Entities: 14541, Relations: 237
2026-07-20 16:26:35,614 |   Train triples: 272115
2026-07-20 16:26:35,614 | Extracting graph features...
2026-07-20 16:26:35,632 |   Features extracted: candidate_size range [1, 5984]
2026-07-20 16:26:35,632 | Building cost table...
2026-07-20 16:26:35,634 |   Cost table built: shape=(14505,), range=[53.13, 518.00]
2026-07-20 16:26:35,638 | ============================================================
2026-07-20 16:26:35,638 | Running configuration: Baseline
2026-07-20 16:26:35,638 |   Sorter: RandomSorter, Packer: ChunkPacker
2026-07-20 16:26:36,076 |   Epoch 0 | Batches: 55 | Weight CV: 0.0008 | Mean weight: 515.80 | Overhead: 66.9ms | Time: 0.4s
2026-07-20 16:26:36,506 |   Epoch 1 | Batches: 55 | Weight CV: 0.0008 | Mean weight: 515.80 | Overhead: 56.6ms | Time: 0.4s
2026-07-20 16:26:36,507 |   [Baseline] Overall Weight CV (all epochs): 0.0008
2026-07-20 16:26:36,507 | ============================================================
2026-07-20 16:26:36,507 | Running configuration: CBP
2026-07-20 16:26:36,507 |   Sorter: CostSorter, Packer: FFDPacker
2026-07-20 16:26:37,201 |   Epoch 0 | Batches: 55 | Weight CV: 0.0735 | Mean weight: 512.87 | Overhead: 335.9ms | Time: 0.7s
2026-07-20 16:26:37,913 |   Epoch 1 | Batches: 55 | Weight CV: 0.0735 | Mean weight: 512.87 | Overhead: 350.7ms | Time: 0.7s
2026-07-20 16:26:37,913 |   [CBP] Overall Weight CV (all epochs): 0.0735
2026-07-20 16:26:37,914 | Batch composition saved: output/results/integration_validation/batch_composition.csv (220 rows)
2026-07-20 16:26:37,963 | Common samples for mapping: 272115
2026-07-20 16:26:38,378 | Batch mapping saved: output/results/integration_validation/batch_mapping.csv (272115 rows)
2026-07-20 16:26:38,378 | Regrouped samples: 267049/272115 (98.14%)
2026-07-20 16:26:38,379 | ============================================================
2026-07-20 16:26:38,379 | INTEGRATION VALIDATION RESULTS
2026-07-20 16:26:38,379 | ============================================================
2026-07-20 16:26:38,379 | [avg_cost] Baseline CV: 0.0008 | CBP CV: 0.0735
2026-07-20 16:26:38,379 | ★ [max_cost] Baseline CV: 0.0000 | CBP CV: 0.0000  (red: 100.0%)
2026-07-20 16:26:38,379 | [hub_count] Baseline CV: 0.0794 | CBP CV: 0.0464  (red: 41.6%)
2026-07-20 16:26:38,379 | [within-batch cv_cost] Baseline: 0.0552 | CBP: 0.0124
2026-07-20 16:26:38,379 | Regrouped sample ratio: 98.14%
2026-07-20 16:26:38,379 | Baseline avg hubs/batch: 2949.2 (max 3056, min 1244)
2026-07-20 16:26:38,379 | CBP      avg hubs/batch: 2949.2 (max 3034, min 1973)
2026-07-20 16:26:38,379 | ============================================================
2026-07-20 16:26:38,379 | ✅ CHECK 1 PASSED: CBP within-batch cv_cost reduced >30% vs Baseline
2026-07-20 16:26:38,379 | ✅ CHECK 2 PASSED: Regroup ratio > 30% (98.1%)
2026-07-20 16:26:38,379 | ✅ CHECK 3 PASSED: CBP hub CV reduced >20% (reduction: 41.6%)
2026-07-20 16:26:38,379 | ============================================================
2026-07-20 16:26:38,379 | ✅ ALL CHECKS PASSED: CBP is effectively balancing batch cost.
2026-07-20 16:26:38,379 |    → Ready to proceed to Node 4 evaluation (full training).
2026-07-20 16:26:38,379 | Validation summary saved: output/results/integration_validation/validation_summary.json
2026-07-20 16:26:38,404 | 
Final verdict: PASS
2026-07-20 16:26:38,404 | CBP Integration Validation complete.
