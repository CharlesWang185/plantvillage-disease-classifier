#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# run_all.sh
#
# Runs the full pipeline:
#   1) Train
#   2) Evaluate on test
#   3) Generate figures (curves, confusion matrix, misclassified gallery)
#   4) Generate Grad-CAM overlays
#
# ✅ To switch models, change ONLY the CONFIG path below:
#   - configs/resnet18.yaml
#   - configs/efficientnet_b0.yaml
# ============================================================

# -----------------------------
# (CHANGE THIS TO SWITCH MODEL)
# -----------------------------
# CONFIG="configs/resnet18.yaml"
CONFIG="configs/efficientnet_b0.yaml"

# Optional: set python executable (useful if you have multiple pythons/venvs)
PYTHON="python"

echo "=============================================="
echo "Using config: ${CONFIG}"
echo "Python: ${PYTHON}"
echo "=============================================="

echo ""
echo "== [1/4] Training =="
${PYTHON} -m src.train --config "${CONFIG}"

echo ""
echo "== [2/4] Evaluating on test split =="
${PYTHON} -m src.eval --config "${CONFIG}"

echo ""
echo "== [3/4] Generating figures =="
# Creates:
#   reports/figures/training_curves.png
#   reports/figures/confusion_matrix_norm.png
#   reports/figures/misclassified_test.png
${PYTHON} -m src.visualize --normalize_cm --make_gallery --config "${CONFIG}"

echo ""
echo "== [4/4] Generating Grad-CAM overlays =="
# Creates:
#   reports/figures/gradcam/*.png
${PYTHON} -m src.explain.gradcam --config "${CONFIG}"

echo ""
echo "✅ Done!"
echo "Check outputs in:"
echo "  - reports/results.json"
echo "  - reports/test_results.json"
echo "  - reports/figures/"
echo "  - reports/figures/gradcam/"
