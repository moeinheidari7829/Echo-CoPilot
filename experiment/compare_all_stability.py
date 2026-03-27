#!/usr/bin/env python3
"""
Compare stability across all four modes:
1. Baseline LLM (no tools, no agent)
2. LLM with tools (tools, no agent framework)
3. Agent with single LLM (agent + tools, no self-contrast)
4. Agent with self-contrast (agent + tools + 3 perspectives)
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any


def load_report(report_path: str) -> Dict[str, Any]:
    """Load a stability report."""
    with open(report_path, 'r') as f:
        return json.load(f)


def compare_stability_reports(
    llm_baseline_path: str,
    llm_with_tools_path: str,
    agent_single_llm_path: str,
    agent_self_contrast_path: str
):
    """Compare four stability reports side by side."""
    
    print("Loading reports...")
    baseline = load_report(llm_baseline_path)
    llm_tools = load_report(llm_with_tools_path)
    single_llm = load_report(agent_single_llm_path)
    self_contrast = load_report(agent_self_contrast_path)
    
    baseline_metrics = baseline['stability_metrics']['overall']
    llm_tools_metrics = llm_tools['stability_metrics']['overall']
    single_metrics = single_llm['stability_metrics']['overall']
    contrast_metrics = self_contrast['stability_metrics']['overall']
    
    print("\n" + "="*110)
    print("STABILITY COMPARISON: Four Testing Modes")
    print("="*110)
    
    print("\nTest Configuration:")
    print(f"  Questions tested: {baseline_metrics['total_questions']}")
    print(f"  Iterations per system: {baseline_metrics['total_iterations']}")
    
    print("\n" + "="*110)
    print("STABILITY METRICS")
    print("="*110)
    print(f"{'Metric':<40} {'Baseline':<12} {'LLM+Tools':<12} {'Agent':<12} {'Agent+SC':<12}")
    print("-"*110)
    
    # Stability rate
    print(f"{'Stability Rate':<40} "
          f"{baseline_metrics['stability_rate']:>10.1%}  "
          f"{llm_tools_metrics['stability_rate']:>10.1%}  "
          f"{single_metrics['stability_rate']:>10.1%}  "
          f"{contrast_metrics['stability_rate']:>10.1%}")
    
    # Avg unique answers
    print(f"{'Avg Unique Answers':<40} "
          f"{baseline_metrics['avg_unique_answers']:>10.2f}  "
          f"{llm_tools_metrics['avg_unique_answers']:>10.2f}  "
          f"{single_metrics['avg_unique_answers']:>10.2f}  "
          f"{contrast_metrics['avg_unique_answers']:>10.2f}")
    
    # Avg changes
    print(f"{'Avg Changes':<40} "
          f"{baseline_metrics['avg_changes_per_question']:>10.2f}  "
          f"{llm_tools_metrics['avg_changes_per_question']:>10.2f}  "
          f"{single_metrics['avg_changes_per_question']:>10.2f}  "
          f"{contrast_metrics['avg_changes_per_question']:>10.2f}")
    
    # Accuracy
    print(f"{'Accuracy':<40} "
          f"{baseline_metrics['mode_accuracy']:>10.1%}  "
          f"{llm_tools_metrics['mode_accuracy']:>10.1%}  "
          f"{single_metrics['mode_accuracy']:>10.1%}  "
          f"{contrast_metrics['mode_accuracy']:>10.1%}")
    
    print("\n" + "="*110)
    print("WINNERS")
    print("="*110)
    
    # Most stable
    stability_rates = {
        "Baseline LLM": baseline_metrics['stability_rate'],
        "LLM + Tools": llm_tools_metrics['stability_rate'],
        "Agent (Single)": single_metrics['stability_rate'],
        "Agent (Self-Contrast)": contrast_metrics['stability_rate']
    }
    most_stable = max(stability_rates, key=stability_rates.get)
    print(f"  Most Stable: {most_stable} ({stability_rates[most_stable]:.1%})")
    
    # Fewest changes
    changes = {
        "Baseline LLM": baseline_metrics['avg_changes_per_question'],
        "LLM + Tools": llm_tools_metrics['avg_changes_per_question'],
        "Agent (Single)": single_metrics['avg_changes_per_question'],
        "Agent (Self-Contrast)": contrast_metrics['avg_changes_per_question']
    }
    fewest_changes = min(changes, key=changes.get)
    print(f"  Fewest Changes: {fewest_changes} ({changes[fewest_changes]:.2f})")
    
    # Best accuracy
    accuracies = {
        "Baseline LLM": baseline_metrics['mode_accuracy'],
        "LLM + Tools": llm_tools_metrics['mode_accuracy'],
        "Agent (Single)": single_metrics['mode_accuracy'],
        "Agent (Self-Contrast)": contrast_metrics['mode_accuracy']
    }
    best_accuracy = max(accuracies, key=accuracies.get)
    print(f"  Best Accuracy: {best_accuracy} ({accuracies[best_accuracy]:.1%})")
    
    print("\n" + "="*110)
    print("ANALYSIS")
    print("="*110)
    
    # Compare single LLM vs self-contrast
    stability_diff = single_metrics['stability_rate'] - contrast_metrics['stability_rate']
    accuracy_diff = single_metrics['mode_accuracy'] - contrast_metrics['mode_accuracy']
    
    print("\nSingle LLM vs Self-Contrast:")
    print(f"  Stability difference: {stability_diff:+.1%}")
    print(f"  Accuracy difference: {accuracy_diff:+.1%}")
    
    if stability_diff > 0:
        print(f"  → Single LLM is MORE stable than self-contrast")
        print(f"  → Self-contrast adds variability without improving accuracy" if accuracy_diff >= 0 else "")
    else:
        print(f"  → Self-contrast is MORE stable (unexpected)")
    
    # Compare agent (single) vs baseline
    stability_diff2 = single_metrics['stability_rate'] - baseline_metrics['stability_rate']
    accuracy_diff2 = single_metrics['mode_accuracy'] - baseline_metrics['mode_accuracy']
    
    print("\nSingle LLM Agent vs Baseline LLM:")
    print(f"  Stability difference: {stability_diff2:+.1%}")
    print(f"  Accuracy difference: {accuracy_diff2:+.1%}")
    
    if stability_diff2 < 0:
        print(f"  → Tools add variability (less stable than baseline)")
    if accuracy_diff2 > 0:
        print(f"  → Tools improve accuracy (worth the variability?)")
    
    # Compare baseline vs LLM+tools
    print("\nBaseline vs LLM+Tools (isolates tool impact):")
    tool_stability_diff = llm_tools_metrics['stability_rate'] - baseline_metrics['stability_rate']
    tool_accuracy_diff = llm_tools_metrics['mode_accuracy'] - baseline_metrics['mode_accuracy']
    print(f"  Stability difference: {tool_stability_diff:+.1%}")
    print(f"  Accuracy difference: {tool_accuracy_diff:+.1%}")
    if tool_stability_diff < 0:
        print(f"  → Tools DECREASE stability")
    if tool_accuracy_diff > 0:
        print(f"  → Tools IMPROVE accuracy")
    
    print("\n" + "="*110)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare stability reports")
    parser.add_argument("--baseline", required=True, help="Path to baseline LLM report")
    parser.add_argument("--llm-tools", required=True, help="Path to LLM with tools report")
    parser.add_argument("--agent-single", required=True, help="Path to single LLM agent report")
    parser.add_argument("--agent-contrast", required=True, help="Path to self-contrast agent report")
    
    args = parser.parse_args()
    
    compare_stability_reports(
        llm_baseline_path=args.baseline,
        llm_with_tools_path=args.llm_tools,
        agent_single_llm_path=args.agent_single,
        agent_self_contrast_path=args.agent_contrast
    )


if __name__ == "__main__":
    main()
