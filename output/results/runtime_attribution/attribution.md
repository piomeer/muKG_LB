2026-07-20 18:00:41,689 | ============================================================
2026-07-20 18:00:41,689 | Runtime Attribution Analysis
2026-07-20 18:00:41,689 | ============================================================
2026-07-20 18:00:41,689 | Dataset: FB15K237, Batch size: 1000, Neg num: 150, Embed dim: 400
2026-07-20 18:00:41,689 | Loading args from: /home/hma/muKG_LB/src/py/experiments/args_kge/transe_fb15k237_args.json
2026-07-20 18:00:43,098 | Dataset: FB15K237
2026-07-20 18:00:43,099 |   Entities: 14541, Relations: 237
2026-07-20 18:00:43,099 |   Train triples: 272115
2026-07-20 18:00:43,099 | Extracting graph features...
2026-07-20 18:00:43,114 | Building cost table...
2026-07-20 18:00:43,117 |   Cost table: shape=(14505,), range=[53.13, 518.00]
2026-07-20 18:00:43,165 | Building TransE model...
2026-07-20 18:00:43,313 |   Model params: 5911200
2026-07-20 18:00:45,655 |   [Baseline] batch 20: weight=514.78, neg=61.1ms, fwd=7.8ms
2026-07-20 18:00:47,548 |   [Baseline] batch 40: weight=514.25, neg=61.1ms, fwd=7.8ms
2026-07-20 18:00:49,436 |   [Baseline] batch 60: weight=514.14, neg=59.7ms, fwd=7.8ms
2026-07-20 18:00:51,300 |   [Baseline] batch 80: weight=516.19, neg=60.7ms, fwd=7.8ms
2026-07-20 18:00:53,190 |   [Baseline] batch 100: weight=515.10, neg=61.1ms, fwd=7.8ms
2026-07-20 18:00:55,083 |   [Baseline] batch 120: weight=514.54, neg=61.6ms, fwd=7.8ms
2026-07-20 18:00:56,974 |   [Baseline] batch 140: weight=515.36, neg=59.9ms, fwd=7.7ms
2026-07-20 18:00:58,853 |   [Baseline] batch 160: weight=515.78, neg=60.4ms, fwd=7.7ms
2026-07-20 18:01:00,748 |   [Baseline] batch 180: weight=517.21, neg=62.9ms, fwd=7.8ms
2026-07-20 18:01:02,642 |   [Baseline] batch 200: weight=514.53, neg=61.3ms, fwd=7.8ms
2026-07-20 18:01:04,539 |   [Baseline] batch 220: weight=515.84, neg=61.6ms, fwd=7.8ms
2026-07-20 18:01:06,428 |   [Baseline] batch 240: weight=514.78, neg=60.8ms, fwd=7.8ms
2026-07-20 18:01:08,321 |   [Baseline] batch 260: weight=516.38, neg=61.2ms, fwd=7.8ms
2026-07-20 18:01:09,486 |   [Baseline] Completed 273 batches.
2026-07-20 18:01:12,683 |   [CBP] batch 20: weight=518.00, neg=60.6ms, fwd=7.8ms
2026-07-20 18:01:14,568 |   [CBP] batch 40: weight=518.00, neg=60.9ms, fwd=7.8ms
2026-07-20 18:01:16,526 |   [CBP] batch 60: weight=518.00, neg=63.7ms, fwd=7.8ms
2026-07-20 18:01:18,489 |   [CBP] batch 80: weight=518.00, neg=64.0ms, fwd=7.8ms
2026-07-20 18:01:20,421 |   [CBP] batch 100: weight=518.00, neg=62.8ms, fwd=7.8ms
2026-07-20 18:01:22,387 |   [CBP] batch 120: weight=518.00, neg=63.1ms, fwd=7.8ms
2026-07-20 18:01:24,348 |   [CBP] batch 140: weight=518.00, neg=62.3ms, fwd=7.8ms
2026-07-20 18:01:26,282 |   [CBP] batch 160: weight=518.00, neg=63.2ms, fwd=7.8ms
2026-07-20 18:01:28,242 |   [CBP] batch 180: weight=518.00, neg=63.5ms, fwd=7.8ms
2026-07-20 18:01:30,166 |   [CBP] batch 200: weight=518.00, neg=61.3ms, fwd=7.7ms
2026-07-20 18:01:32,126 |   [CBP] batch 220: weight=518.00, neg=63.6ms, fwd=7.8ms
2026-07-20 18:01:34,084 |   [CBP] batch 240: weight=518.00, neg=63.6ms, fwd=7.8ms
2026-07-20 18:01:36,018 |   [CBP] batch 260: weight=518.00, neg=61.5ms, fwd=7.7ms
2026-07-20 18:01:37,166 |   [CBP] Completed 273 batches.
2026-07-20 18:01:37,168 | CSV saved: output/results/runtime_attribution/runtime_attribution.csv (546 rows)
2026-07-20 18:01:37,170 | Report saved: output/results/runtime_attribution/attribution_report.txt
