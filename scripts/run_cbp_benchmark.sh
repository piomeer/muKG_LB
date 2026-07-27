#!/bin/bash
set -e

echo "============================================"
echo "CBP Benchmark: Exp-1 (Random+Chunk) + Exp-2 (Cost+FFD)"
echo "============================================"

SCRIPT="src/py/experiments/run_cbp_evaluation.py"
EPOCHS=10

# Clean previous runs
rm -rf output/results/exp_Baseline output/results/exp_CBP
mkdir -p output/results/exp_Baseline output/results/exp_CBP

# ============================================
# Exp-1: Baseline (Random Sorter + Chunk Packer)
# ============================================
echo ""
echo "============================================"
echo "Starting Exp-1: Baseline (Random+Chunk)"
echo "============================================"
echo ""

python3 -u "$SCRIPT" \
    --sorter Random --packer Chunk \
    --epochs "$EPOCHS" \
    --exp-label Baseline 2>&1 | tee output/results/exp_Baseline/training.md

echo ""
echo "============================================"
echo "Exp-1 Done! Starting Exp-2: CBP (Cost+FFD)"
echo "============================================"
echo ""

# ============================================
# Exp-2: CBP (Cost Sorter + FFD Packer)
# ============================================
python3 -u "$SCRIPT" \
    --sorter Cost --packer FFD \
    --epochs "$EPOCHS" \
    --exp-label CBP 2>&1 | tee output/results/exp_CBP/training.md

echo ""
echo "============================================"
echo "Both experiments completed!"
echo "============================================"
echo ""
echo "Results:"
echo "  Exp-1 (Baseline): output/results/exp_Baseline/"
echo "  Exp-2 (CBP):      output/results/exp_CBP/"