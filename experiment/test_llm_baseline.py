#!/usr/bin/env python3
"""
Test baseline LLM (without tools) by running multiple times on the same dataset.
This serves as a baseline to compare against the agent's stability.
"""

import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from config import Config
from experiment.extract_answer import extract_answer_from_text


BASELINE_SYSTEM_PROMPT = """
You are an expert echocardiography assistant. Answer questions about echocardiography videos based on your medical knowledge.

Provide your answer in the following JSON format:
```json
{
  "final_answer": "A/B/C/D",
  "confidence": "low/medium/high",
  "reasoning": "Brief explanation"
}
```
"""


def run_single_iteration(
    llm: ChatOpenAI,
    examples: List[Dict[str, Any]],
    iteration_num: int,
    output_dir: Path
) -> List[Dict[str, Any]]:
    """Run LLM on all examples once."""
    print(f"\n{'='*60}")
    print(f"Iteration {iteration_num}")
    print(f"{'='*60}")

    results = []

    for idx, example in enumerate(examples):
        messages_id = example.get("messages_id", f"unknown_{idx}")
        question = example.get("question", "")

        print(f"[{idx+1}/{len(examples)}] {messages_id[:20]}... ", end="", flush=True)

        # Format query with options
        options = []
        for opt in ['A', 'B', 'C', 'D']:
            opt_key = f'option_{opt}'
            if opt_key in example and example[opt_key]:
                options.append(f"{opt}. {example[opt_key]}")

        if options:
            query = f"{question}\n\nOptions:\n" + "\n".join(options)
        else:
            query = question

        try:
            messages = [
                {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {query}\n\nNote: This is about an echocardiography video. Please answer based on your medical knowledge."}
            ]

            response = llm.invoke(messages)
            response_text = response.content

            extracted_answer = extract_answer_from_text(response_text)

            result = {
                "messages_id": messages_id,
                "iteration": iteration_num,
                "question": question,
                "extracted_answer": extracted_answer,
                "correct_option": example.get("correct_option", ""),
                "response_text": response_text[:500],
                "success": extracted_answer is not None,
            }

            is_correct = (extracted_answer == example.get("correct_option", "")) if extracted_answer else None
            result["is_correct"] = is_correct

            print(f"✓ {extracted_answer or 'NO_ANS'}" + (f" ({'✓' if is_correct else '✗'})" if is_correct is not None else ""))

        except Exception as e:
            result = {
                "messages_id": messages_id,
                "iteration": iteration_num,
                "question": question,
                "extracted_answer": None,
                "correct_option": example.get("correct_option", ""),
                "success": False,
                "error": str(e),
            }
            print(f"✗ Error")

        results.append(result)

        # Save individual result
        individual_file = output_dir / "individual_results" / f"iter{iteration_num:02d}_{idx:03d}_{messages_id}.json"
        individual_file.parent.mkdir(parents=True, exist_ok=True)
        with open(individual_file, 'w') as f:
            json.dump(result, f, indent=2)

        time.sleep(0.5)  # Small delay to avoid rate limits

    return results


def calculate_stability_metrics(all_iterations: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Calculate stability metrics across iterations."""

    # Group by messages_id
    by_question = {}
    for iteration_results in all_iterations:
        for result in iteration_results:
            msg_id = result["messages_id"]
            if msg_id not in by_question:
                by_question[msg_id] = {
                    "question": result.get("question", ""),
                    "correct_option": result.get("correct_option", ""),
                    "answers": [],
                    "iterations": []
                }
            by_question[msg_id]["answers"].append(result.get("extracted_answer"))
            by_question[msg_id]["iterations"].append(result)

    # Calculate per-question metrics
    question_metrics = []
    total_unique_answers = []
    total_changes = []

    for msg_id, data in by_question.items():
        answers = data["answers"]

        # Count unique answers
        answer_counts = Counter(answers)
        unique_answers = len(answer_counts)
        most_common_answer = answer_counts.most_common(1)[0][0] if answer_counts else None
        most_common_count = answer_counts.most_common(1)[0][1] if answer_counts else 0

        # Count changes
        changes = sum(1 for i in range(1, len(answers)) if answers[i] != answers[i-1])

        # Calculate agreement rate
        agreement_rate = most_common_count / len(answers) if answers else 0

        # Check if most common answer is correct
        correct_answer = data["correct_option"]
        is_mode_correct = (most_common_answer == correct_answer) if most_common_answer else None

        question_metrics.append({
            "messages_id": msg_id,
            "question": data["question"][:80] + "...",
            "correct_option": correct_answer,
            "unique_answers": unique_answers,
            "most_common_answer": most_common_answer,
            "most_common_count": most_common_count,
            "agreement_rate": agreement_rate,
            "changes": changes,
            "is_mode_correct": is_mode_correct,
            "all_answers": answers,
        })

        total_unique_answers.append(unique_answers)
        total_changes.append(changes)

    # Overall metrics
    avg_unique_answers = sum(total_unique_answers) / len(total_unique_answers)
    avg_changes = sum(total_changes) / len(total_changes)

    # Calculate standard deviation
    import numpy as np
    std_unique_answers = np.std(total_unique_answers)
    std_changes = np.std(total_changes)

    # Count questions by stability
    stable_questions = sum(1 for q in question_metrics if q["unique_answers"] == 1)
    unstable_questions = len(question_metrics) - stable_questions

    # Calculate accuracy for most common answers
    mode_correct = sum(1 for q in question_metrics if q["is_mode_correct"] is True)
    mode_incorrect = sum(1 for q in question_metrics if q["is_mode_correct"] is False)
    mode_accuracy = mode_correct / (mode_correct + mode_incorrect) if (mode_correct + mode_incorrect) > 0 else 0

    return {
        "overall": {
            "total_questions": len(question_metrics),
            "total_iterations": len(all_iterations),
            "stable_questions": stable_questions,
            "unstable_questions": unstable_questions,
            "stability_rate": stable_questions / len(question_metrics),
            "avg_unique_answers": avg_unique_answers,
            "std_unique_answers": std_unique_answers,
            "avg_changes_per_question": avg_changes,
            "std_changes": std_changes,
            "mode_accuracy": mode_accuracy,
        },
        "per_question": question_metrics,
    }


def run_llm_baseline_test(
    dataset_path: str,
    num_iterations: int = 10,
    num_examples: int = 50,
    start_idx: int = 0,
    output_base_dir: str = "experiment/stability_tests"
) -> None:
    """Run baseline LLM stability test."""

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(output_base_dir) / f"llm_baseline_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"LLM Baseline Stability Test")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_path}")
    print(f"Examples: {num_examples} (starting at {start_idx})")
    print(f"Iterations: {num_iterations}")
    print(f"Model: {Config.OPENAI_MODEL}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    # Load dataset
    with open(dataset_path, 'r') as f:
        full_dataset = json.load(f)

    # Extract subset (same random sampling as agent test)
    import random
    random.seed(42)  # Same seed as agent test for fair comparison

    all_indices = list(range(len(full_dataset)))
    random.shuffle(all_indices)
    selected_indices = all_indices[start_idx:start_idx + num_examples]

    examples = []
    for idx in selected_indices:
        item = full_dataset[idx]

        if not item.get("videos"):
            continue

        examples.append({
            "messages_id": item.get("messages_id", f"unknown_{idx}"),
            "question": item.get("question", ""),
            "option_A": item.get("option_A", ""),
            "option_B": item.get("option_B", ""),
            "option_C": item.get("option_C", ""),
            "option_D": item.get("option_D", ""),
            "correct_option": item.get("correct_option", ""),
        })

    print(f"\nFound {len(examples)} valid examples")

    # Initialize LLM
    print(f"\nInitializing LLM...")
    llm = ChatOpenAI(
        api_key=Config.OPENAI_API_KEY,
        model=Config.OPENAI_MODEL,
        temperature=Config.OPENAI_TEMPERATURE,
        max_tokens=Config.OPENAI_MAX_TOKENS,
        base_url=Config.OPENAI_BASE_URL,
    )
    print(f"✓ LLM ready\n")

    # Run iterations
    all_iterations = []
    start_time = time.time()

    for iteration in range(1, num_iterations + 1):
        iteration_results = run_single_iteration(
            llm=llm,
            examples=examples,
            iteration_num=iteration,
            output_dir=output_dir
        )
        all_iterations.append(iteration_results)

        # Save iteration summary
        iteration_file = output_dir / f"iteration_{iteration:02d}_summary.json"
        iteration_data = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "num_examples": len(examples),
            "results": iteration_results,
        }
        with open(iteration_file, 'w') as f:
            json.dump(iteration_data, f, indent=2)

    # Calculate stability metrics
    print(f"\n{'='*60}")
    print(f"Calculating stability metrics...")
    print(f"{'='*60}")

    stability_metrics = calculate_stability_metrics(all_iterations)

    # Save final report
    final_report = {
        "metadata": {
            "test_type": "llm_baseline",
            "model": Config.OPENAI_MODEL,
            "temperature": Config.OPENAI_TEMPERATURE,
            "dataset_path": dataset_path,
            "num_iterations": num_iterations,
            "num_examples": len(examples),
            "start_idx": start_idx,
            "timestamp": timestamp,
            "total_time_seconds": time.time() - start_time,
        },
        "stability_metrics": stability_metrics,
    }

    report_file = output_dir / "stability_report.json"
    with open(report_file, 'w') as f:
        json.dump(final_report, f, indent=2)

    # Print summary
    metrics = stability_metrics["overall"]
    print(f"\nLLM Baseline Stability Results:")
    print(f"  Total questions: {metrics['total_questions']}")
    print(f"  Total iterations: {metrics['total_iterations']}")
    print(f"  Stable questions (same answer every time): {metrics['stable_questions']} ({metrics['stability_rate']:.1%})")
    print(f"  Unstable questions: {metrics['unstable_questions']}")
    print(f"  Avg unique answers per question: {metrics['avg_unique_answers']:.2f} ± {metrics['std_unique_answers']:.2f}")
    print(f"  Avg changes per question: {metrics['avg_changes_per_question']:.2f} ± {metrics['std_changes']:.2f}")
    print(f"  Mode accuracy (most common answer): {metrics['mode_accuracy']:.1%}")
    print(f"\n[SAVED] Full report: {report_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test baseline LLM stability across multiple runs")
    parser.add_argument("--dataset", default="mimic-iv-echo-ext-mimicechoqa-a-benchmark-dataset-for-echocardiogram-based-visual-question-answering-1.0.0/MIMICEchoQA/MIMICEchoQA.json")
    parser.add_argument("--num-iterations", type=int, default=10, help="Number of times to run on same examples")
    parser.add_argument("--num-examples", type=int, default=50, help="Number of examples to test")
    parser.add_argument("--start-idx", type=int, default=0, help="Starting index in dataset")
    parser.add_argument("--output-dir", default="experiment/stability_tests")

    args = parser.parse_args()

    run_llm_baseline_test(
        dataset_path=args.dataset,
        num_iterations=args.num_iterations,
        num_examples=args.num_examples,
        start_idx=args.start_idx,
        output_base_dir=args.output_dir
    )


if __name__ == "__main__":
    main()

