#!/bin/bash
# Rerun ONLY Agent+Self-Contrast test on full dataset

set -e

OUTPUT_DIR="experiment/accuracy_comparison/run_20260216_092719"
NUM_EXAMPLES=622
MEASUREMENT_TOOL="echoprime"

echo "================================================================================"
echo "Rerunning Agent+Self-Contrast on Full Dataset"
echo "================================================================================"
echo "This will run the self-contrast agent on all 622 examples (1 iteration)"
echo ""
echo "Configuration:"
echo "  Examples: $NUM_EXAMPLES"
echo "  Measurement tool: $MEASUREMENT_TOOL"
echo "  Output dir: $OUTPUT_DIR"
echo "================================================================================"
echo ""

# Remove the old failed run
OLD_RUN="$OUTPUT_DIR/agent_fresh_echoprime_20260217_204243"
if [ -d "$OLD_RUN" ]; then
    echo "Removing old failed run: $OLD_RUN"
    rm -rf "$OLD_RUN"
fi

echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Aborted."
    exit 1
fi

echo ""
echo "Starting Agent+Self-Contrast test..."
START_TIME=$(date +%s)

python experiment/test_agent_stability_fresh.py \
    --num-examples $NUM_EXAMPLES \
    --num-iterations 1 \
    --start-idx 0 \
    --measurement-tool $MEASUREMENT_TOOL \
    --output-dir "$OUTPUT_DIR"

# Find the new report
AGENT_CONTRAST_REPORT=$(ls -t $OUTPUT_DIR/agent_fresh_*/stability_report.json | head -1)

echo ""
echo "✓ Agent+Self-Contrast complete!"
echo "  Report: $AGENT_CONTRAST_REPORT"
echo ""

# Calculate time
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))

echo "Time taken: ${HOURS}h ${MINUTES}m"
echo ""

# Now compare with existing results
echo "================================================================================"
echo "Comparing with Existing Results"
echo "================================================================================"

BASELINE_REPORT=$(ls $OUTPUT_DIR/llm_baseline_*/stability_report.json)
LLM_TOOLS_REPORT=$(ls $OUTPUT_DIR/llm_with_tools_*/stability_report.json)
AGENT_SINGLE_REPORT=$(ls $OUTPUT_DIR/agent_single_llm_*/stability_report.json)

python experiment/compare_all_stability.py \
    --baseline "$BASELINE_REPORT" \
    --llm-tools "$LLM_TOOLS_REPORT" \
    --agent-single "$AGENT_SINGLE_REPORT" \
    --agent-contrast "$AGENT_CONTRAST_REPORT"

echo ""
echo "================================================================================"
echo "Complete!"
echo "================================================================================"
echo "All results in: $OUTPUT_DIR"
echo "================================================================================"
