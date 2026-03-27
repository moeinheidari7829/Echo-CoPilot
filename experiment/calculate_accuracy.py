"""
Calculate accuracy from evaluation results file.
Supports both single-file results and multi-chunk results.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List


def calculate_accuracy_from_single_file(results: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate accuracy from a single results file."""
    all_results = results.get('results', [])

    total = len(all_results)
    correct = sum(1 for r in all_results if r.get('is_correct') == True)
    incorrect = sum(1 for r in all_results if r.get('is_correct') == False)
    no_answer = sum(1 for r in all_results if r.get('is_correct') is None and r.get('matched', True))
    failed = sum(1 for r in all_results if not r.get('success', True))

    accuracy = correct / (correct + incorrect) if (correct + incorrect) > 0 else 0.0

    return {
        "total_examples": total,
        "correct": correct,
        "incorrect": incorrect,
        "no_answer": no_answer,
        "failed": failed,
        "accuracy": accuracy
    }


def calculate_accuracy_from_chunks(data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate accuracy from multi-chunk results file."""
    chunk_names = ['first_chunk', 'second_chunk', 'third_chunk', 'fourth_chunk',
                   'fifth_chunk', 'sixth_chunk', 'seventh_chunk', 'eighth_chunk']

    all_results = []
    chunk_stats = {}

    # Aggregate all results from all chunks
    for chunk_name in chunk_names:
        if chunk_name in data:
            chunk_results = data[chunk_name].get('results', [])
            all_results.extend(chunk_results)

            # Calculate stats for this chunk
            chunk_correct = sum(1 for r in chunk_results if r.get('is_correct') == True)
            chunk_incorrect = sum(1 for r in chunk_results if r.get('is_correct') == False)
            chunk_accuracy = chunk_correct / (chunk_correct + chunk_incorrect) if (chunk_correct + chunk_incorrect) > 0 else 0.0

            chunk_stats[chunk_name] = {
                "total": len(chunk_results),
                "correct": chunk_correct,
                "incorrect": chunk_incorrect,
                "accuracy": chunk_accuracy
            }

    # Calculate overall statistics
    total = len(all_results)
    correct = sum(1 for r in all_results if r.get('is_correct') == True)
    incorrect = sum(1 for r in all_results if r.get('is_correct') == False)
    no_answer = sum(1 for r in all_results if r.get('is_correct') is None and r.get('matched', True))
    failed = sum(1 for r in all_results if not r.get('success', True))

    accuracy = correct / (correct + incorrect) if (correct + incorrect) > 0 else 0.0

    return {
        "total_examples": total,
        "correct": correct,
        "incorrect": incorrect,
        "no_answer": no_answer,
        "failed": failed,
        "accuracy": accuracy,
        "chunk_breakdown": chunk_stats
    }


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate accuracy from evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calculate from single results file
  python experiment/calculate_accuracy.py experiment/pmc_vqa_results_20260106.json

  # Calculate from multi-chunk results
  python experiment/calculate_accuracy.py experiment/final_result_Jan_6.json
        """
    )
    parser.add_argument(
        "--results_file",
        help="Path to results JSON file"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed breakdown by chunk"
    )

    args = parser.parse_args()

    # Load results file
    print(f"Loading results from {args.results_file}...")
    with open(args.results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Detect format (single file vs chunks)
    if 'first_chunk' in data or 'second_chunk' in data:
        print("Detected multi-chunk format\n")
        stats = calculate_accuracy_from_chunks(data)
    else:
        print("Detected single-file format\n")
        stats = calculate_accuracy_from_single_file(data)

    # Print results
    print("="*60)
    print("ACCURACY REPORT")
    print("="*60)
    print(f"Total Examples:     {stats['total_examples']}")
    print(f"Correct:            {stats['correct']}")
    print(f"Incorrect:          {stats['incorrect']}")
    print(f"No Answer:          {stats['no_answer']}")
    print(f"Failed:             {stats['failed']}")
    print(f"\nAccuracy:           {stats['accuracy']:.2%}")
    print("="*60)

    # Print chunk breakdown if available and requested
    if args.detailed and 'chunk_breakdown' in stats:
        print("\nCHUNK BREAKDOWN:")
        print("-"*60)
        for chunk_name, chunk_stat in stats['chunk_breakdown'].items():
            print(f"{chunk_name}:")
            print(f"  Total: {chunk_stat['total']}, "
                  f"Correct: {chunk_stat['correct']}, "
                  f"Incorrect: {chunk_stat['incorrect']}, "
                  f"Accuracy: {chunk_stat['accuracy']:.2%}")
        print("-"*60)


if __name__ == "__main__":
    main()

