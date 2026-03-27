#!/bin/bash
# Run complete stability experiment: Agent vs Baseline LLM

set -e

echo "==============================================================="
echo "EchoPilot Stability Experiment"
echo "==============================================================="
echo ""
echo "This will:"
echo "  1. Run agent 10 times on 50 examples (using echoprime)"
echo "  2. Run baseline LLM 10 times on same 50 examples"
echo "  3. Compare stability metrics"
echo ""
echo "Estimated time: ~30-60 minutes"
echo "==============================================================="

# Configuration
NUM_ITERATIONS=10
NUM_EXAMPLES=50
START_IDX=0
MEASUREMENT_TOOL="echoprime"

# Step 1: Run agent stability test
echo ""
echo "[1/3] Running agent stability test..."
echo "---------------------------------------------------------------"
uv run python experiment/test_agent_stability.py \
  --num-iterations $NUM_ITERATIONS \
  --num-examples $NUM_EXAMPLES \
  --start-idx $START_IDX \
  --measurement-tool $MEASUREMENT_TOOL

# Get the latest agent report
AGENT_REPORT=$(ls -t experiment/stability_tests/agent_${MEASUREMENT_TOOL}_*/stability_report.json | head -1)
AGENT_DIR=$(dirname "$AGENT_REPORT")

echo ""
echo "✓ Agent test complete"
echo "  Report: $AGENT_REPORT"

# Step 2: Run LLM baseline stability test
echo ""
echo "[2/3] Running LLM baseline stability test..."
echo "---------------------------------------------------------------"
uv run python experiment/test_llm_baseline.py \
  --num-iterations $NUM_ITERATIONS \
  --num-examples $NUM_EXAMPLES \
  --start-idx $START_IDX

# Get the latest LLM report
LLM_REPORT=$(ls -t experiment/stability_tests/llm_baseline_*/stability_report.json | head -1)
LLM_DIR=$(dirname "$LLM_REPORT")

echo ""
echo "✓ LLM test complete"
echo "  Report: $LLM_REPORT"

# Step 3: Compare results
echo ""
echo "[3/3] Comparing stability metrics..."
echo "---------------------------------------------------------------"
uv run python experiment/compare_stability.py \
  --agent-report "$AGENT_REPORT" \
  --llm-report "$LLM_REPORT"

echo ""
echo "==============================================================="
echo "Experiment Complete!"
echo "==============================================================="
echo ""
echo "Results saved to:"
echo "  Agent:    $AGENT_DIR"
echo "  LLM:      $LLM_DIR"
echo ""
echo "Directory structure:"
echo "  experiment/stability_tests/"
echo "  ├── agent_echoprime_TIMESTAMP/"
echo "  │   ├── stability_report.json"
echo "  │   ├── iteration_01_summary.json"
echo "  │   ├── iteration_02_summary.json"
echo "  │   └── individual_results/"
echo "  │       ├── iter01_000_<id>.json"
echo "  │       └── ..."
echo "  ├── llm_baseline_TIMESTAMP/"
echo "  │   ├── stability_report.json"
echo "  │   ├── iteration_01_summary.json"
echo "  │   └── individual_results/"
echo "  └── comparison_TIMESTAMP.json"
echo ""
echo "==============================================================="

