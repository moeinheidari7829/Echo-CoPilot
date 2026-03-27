#!/bin/bash
# Complete accuracy comparison - runs all 4 modes on entire dataset ONCE
# This tests accuracy/performance, not stability

set -e  # Exit on error

# Parse arguments
NUM_EXAMPLES=${1:-622}  # Default: all examples
START_IDX=${2:-0}
MEASUREMENT_TOOL="echoprime"
OUTPUT_BASE="experiment/accuracy_comparison"
DATASET="mimic-iv-echo-ext-mimicechoqa-a-benchmark-dataset-for-echocardiogram-based-visual-question-answering-1.0.0/MIMICEchoQA/MIMICEchoQA.json"

# Create timestamped run directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${OUTPUT_BASE}/run_${TIMESTAMP}"
mkdir -p "$RUN_DIR"

echo "================================================================================"
echo "FULL DATASET ACCURACY COMPARISON"
echo "================================================================================"
echo "This will run 4 experiments on the full dataset (1 iteration each):"
echo "  1. Baseline LLM (no tools, no agent)"
echo "  2. LLM + Tools (no agent framework)"
echo "  3. Agent with Single LLM (no self-contrast)"
echo "  4. Agent with Self-Contrast"
echo ""
echo "Configuration:"
echo "  Dataset: MIMICEchoQA"
echo "  Start index: $START_IDX"
echo "  Number of examples: $NUM_EXAMPLES"
echo "  Iterations: 1 (single run for accuracy)"
echo "  Measurement tool: $MEASUREMENT_TOOL"
echo "  Output: $RUN_DIR"
echo ""
echo "Total evaluations: $((NUM_EXAMPLES * 4))"
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
START_TIME=$(date +%s)

# ============================================================================
# Experiment 1: Baseline LLM
# ============================================================================
echo "[1/4] Running Baseline LLM..."
echo "---------------------------------------------------------------"
python experiment/test_llm_baseline.py \
    --num-examples $NUM_EXAMPLES \
    --num-iterations 1 \
    --start-idx $START_IDX \
    --output-dir "$RUN_DIR"

BASELINE_REPORT=$(ls -t $RUN_DIR/llm_baseline_*/stability_report.json | head -1)
echo "✓ Baseline complete: $BASELINE_REPORT"
echo ""

# ============================================================================
# Experiment 2: LLM + Tools
# ============================================================================
echo "[2/4] Running LLM + Tools..."
echo "---------------------------------------------------------------"
python experiment/test_llm_with_tools.py \
    --num-examples $NUM_EXAMPLES \
    --num-iterations 1 \
    --start-idx $START_IDX \
    --measurement-tool $MEASUREMENT_TOOL \
    --output-dir "$RUN_DIR"

LLM_TOOLS_REPORT=$(ls -t $RUN_DIR/llm_with_tools_*/stability_report.json | head -1)
echo "✓ LLM + Tools complete: $LLM_TOOLS_REPORT"
echo ""

# ============================================================================
# Experiment 3: Agent (Single LLM)
# ============================================================================
echo "[3/4] Running Agent (Single LLM)..."
echo "---------------------------------------------------------------"
python experiment/test_agent_single_llm.py \
    --num-examples $NUM_EXAMPLES \
    --num-iterations 1 \
    --start-idx $START_IDX \
    --measurement-tool $MEASUREMENT_TOOL \
    --output-dir "$RUN_DIR"

AGENT_SINGLE_REPORT=$(ls -t $RUN_DIR/agent_single_llm_*/stability_report.json | head -1)
echo "✓ Agent (Single) complete: $AGENT_SINGLE_REPORT"
echo ""

# ============================================================================
# Experiment 4: Agent (Self-Contrast)
# ============================================================================
echo "[4/4] Running Agent (Self-Contrast)..."
echo "---------------------------------------------------------------"
python experiment/test_agent_stability_fresh.py \
    --num-examples $NUM_EXAMPLES \
    --num-iterations 1 \
    --start-idx $START_IDX \
    --measurement-tool $MEASUREMENT_TOOL \
    --output-dir "$RUN_DIR"

AGENT_CONTRAST_REPORT=$(ls -t $RUN_DIR/agent_fresh_*/stability_report.json | head -1)
echo "✓ Agent (Self-Contrast) complete: $AGENT_CONTRAST_REPORT"
echo ""

# ============================================================================
# Compare Results
# ============================================================================
echo "================================================================================"
echo "COMPARING ACCURACY ACROSS ALL MODES"
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
echo "Accuracy Comparison Complete!"
echo "================================================================================"
echo ""
echo "Results saved to: $RUN_DIR"
echo "Total time: ${HOURS}h ${MINUTES}m"
echo ""
echo "================================================================================"
