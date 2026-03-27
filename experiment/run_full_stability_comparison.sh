#!/bin/bash
# Complete stability comparison experiment
# Runs all 4 testing modes and compares results

set -e  # Exit on error

# Parse command line arguments
START_IDX=${1:-0}
NUM_EXAMPLES=${2:-50}
NUM_ITERATIONS=${3:-10}
MEASUREMENT_TOOL="echoprime"
OUTPUT_BASE="experiment/stability_tests"

echo "================================================================================"
echo "COMPLETE STABILITY COMPARISON EXPERIMENT"
echo "================================================================================"
echo "This will run 4 experiments:"
echo "  1. Baseline LLM (no tools, no agent)"
echo "  2. LLM + Tools (no agent framework)"
echo "  3. Agent with Single LLM (no self-contrast)"
echo "  4. Agent with Self-Contrast"
echo ""
echo "Configuration:"
echo "  Start index: $START_IDX"
echo "  Examples per test: $NUM_EXAMPLES"
echo "  Iterations per test: $NUM_ITERATIONS"
echo "  Measurement tool: $MEASUREMENT_TOOL"
echo ""
echo "Total evaluations: $((NUM_EXAMPLES * NUM_ITERATIONS * 4))"
echo "================================================================================"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Aborted."
    exit 1
fi

echo ""
echo "Starting experiments..."
echo ""

# Track start time
START_TIME=$(date +%s)

# ============================================================================
# Experiment 1: Baseline LLM
# ============================================================================
echo "[1/4] Running Baseline LLM test..."
echo "---------------------------------------------------------------"
python experiment/test_llm_baseline.py \
    --num-examples $NUM_EXAMPLES \
    --num-iterations $NUM_ITERATIONS \
    --start-idx $START_IDX \
    --output-dir $OUTPUT_BASE

# Find the most recent baseline report
BASELINE_REPORT=$(ls -t $OUTPUT_BASE/llm_baseline_*/stability_report.json | head -1)
echo "✓ Baseline test complete"
echo "  Report: $BASELINE_REPORT"
echo ""

# ============================================================================
# Experiment 2: LLM + Tools
# ============================================================================
echo "[2/4] Running LLM + Tools test..."
echo "---------------------------------------------------------------"
python experiment/test_llm_with_tools.py \
    --num-examples $NUM_EXAMPLES \
    --num-iterations $NUM_ITERATIONS \
    --start-idx $START_IDX \
    --measurement-tool $MEASUREMENT_TOOL \
    --output-dir $OUTPUT_BASE

# Find the most recent LLM+tools report
LLM_TOOLS_REPORT=$(ls -t $OUTPUT_BASE/llm_with_tools_*/stability_report.json | head -1)
echo "✓ LLM + Tools test complete"
echo "  Report: $LLM_TOOLS_REPORT"
echo ""

# ============================================================================
# Experiment 3: Agent (Single LLM)
# ============================================================================
echo "[3/4] Running Agent (Single LLM) test..."
echo "---------------------------------------------------------------"
python experiment/test_agent_single_llm.py \
    --num-examples $NUM_EXAMPLES \
    --num-iterations $NUM_ITERATIONS \
    --start-idx $START_IDX \
    --measurement-tool $MEASUREMENT_TOOL \
    --output-dir $OUTPUT_BASE

# Find the most recent agent single LLM report
AGENT_SINGLE_REPORT=$(ls -t $OUTPUT_BASE/agent_single_llm_*/stability_report.json | head -1)
echo "✓ Agent (Single LLM) test complete"
echo "  Report: $AGENT_SINGLE_REPORT"
echo ""

# ============================================================================
# Experiment 4: Agent (Self-Contrast)
# ============================================================================
echo "[4/4] Running Agent (Self-Contrast) test..."
echo "---------------------------------------------------------------"
python experiment/test_agent_stability_fresh.py \
    --num-examples $NUM_EXAMPLES \
    --num-iterations $NUM_ITERATIONS \
    --start-idx $START_IDX \
    --measurement-tool $MEASUREMENT_TOOL \
    --output-dir $OUTPUT_BASE

# Find the most recent agent self-contrast report
AGENT_CONTRAST_REPORT=$(ls -t $OUTPUT_BASE/agent_fresh_*/stability_report.json | head -1)
echo "✓ Agent (Self-Contrast) test complete"
echo "  Report: $AGENT_CONTRAST_REPORT"
echo ""

# ============================================================================
# Compare Results
# ============================================================================
echo "================================================================================"
echo "COMPARING ALL RESULTS"
echo "================================================================================"

python experiment/compare_all_stability.py \
    --baseline "$BASELINE_REPORT" \
    --llm-tools "$LLM_TOOLS_REPORT" \
    --agent-single "$AGENT_SINGLE_REPORT" \
    --agent-contrast "$AGENT_CONTRAST_REPORT"

# Calculate total time
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))

echo ""
echo "================================================================================"
echo "Experiment Complete!"
echo "================================================================================"
echo ""
echo "Results saved to:"
echo "  Baseline LLM:       $BASELINE_REPORT"
echo "  LLM + Tools:        $LLM_TOOLS_REPORT"
echo "  Agent (Single):     $AGENT_SINGLE_REPORT"
echo "  Agent (Contrast):   $AGENT_CONTRAST_REPORT"
echo ""
echo "Total time: ${HOURS}h ${MINUTES}m"
echo ""
echo "================================================================================"
