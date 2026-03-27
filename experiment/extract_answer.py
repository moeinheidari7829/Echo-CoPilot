"""
Extract final answer from agent response/trajectory files.
Supports multiple extraction patterns to handle different response formats.
"""

import json
import re
from pathlib import Path
from typing import Optional, Dict, Any, List


def extract_answer_from_text(text: str) -> Optional[str]:
    """
    Extract the final answer option (A, B, C, or D) from agent response text.
    
    Supports multiple patterns:
    - JSON format: {"final_answer": "A", ...}
    - "Answer: B) No"
    - "The answer is B) No"
    - "Therefore, the final answer is: Answer: B) No"
    - "B) No" (standalone)
    - "Answer: B"
    - "The answer is B"
    - Malformed starts like: B". or A\".
    
    Args:
        text: Agent response text
        
    Returns:
        Extracted answer option (A, B, C, or D) or None if not found
    """
    if not text or len(text.strip()) < 2:
        return None
    
    # FIRST: Check for malformed starts like: B". or A\". (common error pattern)
    malformed_start = re.match(r'^([A-D])[\"\.]', text.strip(), re.IGNORECASE)
    if malformed_start:
        return malformed_start.group(1).upper()
    
    # First, try to extract from JSON format (new structured format)
    # Look for JSON block in code fences or plain JSON
    try:
        # Try to find JSON code blocks first
        json_block_pattern = r'```(?:json)?\s*(\{.*?"final_answer".*?\})\s*```'
        json_match = re.search(json_block_pattern, text, re.DOTALL | re.IGNORECASE)
        if json_match:
            json_str = json_match.group(1)
            json_obj = json.loads(json_str)
            final_answer = str(json_obj.get("final_answer", "")).upper().strip()
            # Remove any parentheses or extra text (e.g., "A)" -> "A")
            final_answer = re.sub(r'[^A-D]', '', final_answer)
            if final_answer in ['A', 'B', 'C', 'D']:
                return final_answer
        
        # Try to find plain JSON object (not in code block)
        json_obj_pattern = r'\{\s*"final_answer"\s*:\s*"([A-D])"[^}]*\}'
        json_obj_match = re.search(json_obj_pattern, text, re.IGNORECASE)
        if json_obj_match:
            answer = json_obj_match.group(1).upper()
            if answer in ['A', 'B', 'C', 'D']:
                return answer
    except (json.JSONDecodeError, AttributeError, KeyError, IndexError):
        # If JSON parsing fails, fall through to text pattern matching
        pass
    
    # Pattern 1: "Answer: X) ..." or "Answer: X" or "**X) ...**" (markdown bold)
    pattern1 = r'(?:^|\n|\.|,|:)\s*[Aa]nswer\s*:?\s*([A-D])\)?\s*[:\-]?\s*[^\n]*'
    match1 = re.search(pattern1, text, re.MULTILINE | re.IGNORECASE)
    if match1:
        return match1.group(1).upper()
    
    # Pattern 1b: "**X) ...**" (markdown bold answer at end)
    pattern1b = r'\*\*([A-D])\)\s+[^\*\n]+\*\*'
    # Look for this pattern near the end of the text
    lines = text.split('\n')
    for line in reversed(lines[-5:]):  # Check last 5 lines
        match1b = re.search(pattern1b, line, re.IGNORECASE)
        if match1b:
            return match1b.group(1).upper()
    
    # Pattern 2: "The answer is X) ..." or "The answer is X"
    pattern2 = r'[Tt]he\s+[Aa]nswer\s+is\s*:?\s*([A-D])\)?\s*[:\-]?\s*[^\n]*'
    match2 = re.search(pattern2, text, re.MULTILINE | re.IGNORECASE)
    if match2:
        return match2.group(1).upper()
    
    # Pattern 3: "Final answer: X) ..." or "Final answer is X" or "**Final Answer**: ... **X) ...**"
    pattern3 = r'[Ff]inal\s+[Aa]nswer\s*:?\s*is\s*:?\s*([A-D])\)?\s*[:\-]?\s*[^\n]*'
    match3 = re.search(pattern3, text, re.MULTILINE | re.IGNORECASE)
    if match3:
        return match3.group(1).upper()
    
    # Pattern 3b: "**Final Answer**: ... **X) ...**" (with markdown bold)
    pattern3b = r'\*\*[Ff]inal\s+[Aa]nswer\s*\*\*\s*:?\s*[^.]*\*\*([A-D])\)\s*[^\*\n]+\*\*'
    match3b = re.search(pattern3b, text, re.MULTILINE | re.IGNORECASE)
    if match3b:
        return match3b.group(1).upper()
    
    # Pattern 4: "Therefore, ... Answer: X) ..."
    pattern4 = r'[Tt]herefore[^.]*[Aa]nswer\s*:?\s*([A-D])\)?\s*[:\-]?\s*[^\n]*'
    match4 = re.search(pattern4, text, re.MULTILINE | re.IGNORECASE)
    if match4:
        return match4.group(1).upper()
    
    # Pattern 5: Standalone "X) ..." at the end (most common format)
    # Look for lines ending with "X) ..." where X is A-D
    lines = text.split('\n')
    for line in reversed(lines):  # Check from end
        line = line.strip()
        if re.match(r'^([A-D])\)\s+', line, re.IGNORECASE):
            match5 = re.match(r'^([A-D])\)', line, re.IGNORECASE)
            if match5:
                return match5.group(1).upper()
    
    # Pattern 6: "Conclusion: ... X) ..."
    pattern6 = r'[Cc]onclusion\s*:?\s*[^.]*([A-D])\)\s*[^\n]*'
    match6 = re.search(pattern6, text, re.MULTILINE | re.IGNORECASE)
    if match6:
        return match6.group(1).upper()
    
    return None


def extract_answer_from_trajectory(trajectory_path: str) -> Optional[Dict[str, Any]]:
    """
    Extract answer from a trajectory JSON file.
    
    Args:
        trajectory_path: Path to trajectory JSON file
        
    Returns:
        Dictionary with extracted answer and metadata, or None if not found
    """
    trajectory_path = Path(trajectory_path)
    
    if not trajectory_path.exists():
        print(f"✗ Trajectory file not found: {trajectory_path}")
        return None
    
    with open(trajectory_path, 'r', encoding='utf-8') as f:
        trajectory = json.load(f)
    
    # Get final response
    final_response = trajectory.get("final_response", "")
    if not final_response:
        print(f"✗ No final_response found in trajectory")
        return None
    
    # Extract answer
    extracted_answer = extract_answer_from_text(final_response)
    
    # Get metadata
    metadata = trajectory.get("metadata", {})
    query = metadata.get("query", "")
    video_path = metadata.get("video_path", "")
    
    return {
        "extracted_answer": extracted_answer,
        "query": query,
        "video_path": video_path,
        "final_response": final_response[:200] + "..." if len(final_response) > 200 else final_response,
        "trajectory_path": str(trajectory_path)
    }


def extract_answers_from_directory(
    trajectory_dir: str = "logs",
    output_path: str = "experiment/extracted_answers.json"
) -> List[Dict[str, Any]]:
    """
    Extract answers from all trajectory files in a directory.
    
    Args:
        trajectory_dir: Directory containing trajectory JSON files
        output_path: Path to save extracted answers
        
    Returns:
        List of extracted answers with metadata
    """
    trajectory_dir = Path(trajectory_dir)
    output_path = Path(output_path)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Find all trajectory files
    trajectory_files = list(trajectory_dir.glob("trajectory_*.json"))
    
    print(f"Found {len(trajectory_files)} trajectory files")
    
    extracted_answers = []
    for traj_file in trajectory_files:
        result = extract_answer_from_trajectory(str(traj_file))
        if result:
            result["trajectory_file"] = traj_file.name
            extracted_answers.append(result)
            print(f"✓ {traj_file.name}: Extracted answer: {result['extracted_answer']}")
        else:
            print(f"✗ {traj_file.name}: Could not extract answer")
    
    # Save results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(extracted_answers, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved {len(extracted_answers)} extracted answers to {output_path}")
    
    return extracted_answers


def evaluate_answers(
    extracted_answers_path: str = "experiment/extracted_answers.json",
    dataset_path: str = "MIMICEchoQA/MIMICEchoQA.json",
    output_path: str = "experiment/evaluation_results.json"
) -> Dict[str, Any]:
    """
    Evaluate extracted answers against ground truth.
    
    Args:
        extracted_answers_path: Path to extracted answers JSON
        dataset_path: Path to original dataset with ground truth
        output_path: Path to save evaluation results
        
    Returns:
        Evaluation results dictionary
    """
    extracted_answers_path = Path(extracted_answers_path)
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    
    # Load extracted answers
    with open(extracted_answers_path, 'r', encoding='utf-8') as f:
        extracted_answers = json.load(f)
    
    # Load dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Create lookup for ground truth by video path and filename
    ground_truth_lookup = {}
    for item in dataset:
        if item.get("videos"):
            video_path = item["videos"][0]
            # Normalize path for matching
            ground_truth_lookup[video_path] = {
                "correct_option": item.get("correct_option", ""),
                "answer": item.get("answer", ""),
                "question": item.get("question", ""),
                "messages_id": item.get("messages_id", ""),
                "video_path": video_path
            }
            # Also index by filename for easier matching
            video_filename = Path(video_path).name
            if video_filename not in ground_truth_lookup:
                ground_truth_lookup[video_filename] = ground_truth_lookup[video_path]
    
    # Evaluate each extracted answer
    results = []
    correct_count = 0
    total_count = 0
    
    for extracted in extracted_answers:
        video_path = extracted.get("video_path", "")
        extracted_answer = extracted.get("extracted_answer", "")
        
        # Try to match video path
        # Extract relative path from absolute path
        rel_path = None
        if "MIMICEchoQA" in video_path:
            rel_path = video_path.split("MIMICEchoQA")[-1].replace("\\", "/").lstrip("/")
            if rel_path.startswith("0.1/files/"):
                rel_path = "mimic-iv-echo/" + rel_path
        
        ground_truth = None
        for key, value in ground_truth_lookup.items():
            # Try multiple matching strategies
            if key in video_path or video_path.endswith(key):
                ground_truth = value
                break
            if rel_path and (rel_path in key or key in rel_path):
                ground_truth = value
                break
            # Try matching by filename
            video_filename = Path(video_path).name
            if video_filename in key or key.endswith(video_filename):
                ground_truth = value
                break
        
        if ground_truth:
            correct_option = ground_truth.get("correct_option", "").upper()
            is_correct = extracted_answer == correct_option
            
            result = {
                "trajectory_file": extracted.get("trajectory_file", ""),
                "query": extracted.get("query", ""),
                "extracted_answer": extracted_answer,
                "correct_option": correct_option,
                "ground_truth_answer": ground_truth.get("answer", ""),
                "is_correct": is_correct,
                "question": ground_truth.get("question", ""),
                "messages_id": ground_truth.get("messages_id", "")
            }
            results.append(result)
            
            if is_correct:
                correct_count += 1
            total_count += 1
        else:
            # Could not match video path
            result = {
                "trajectory_file": extracted.get("trajectory_file", ""),
                "query": extracted.get("query", ""),
                "extracted_answer": extracted_answer,
                "correct_option": None,
                "ground_truth_answer": None,
                "is_correct": None,
                "question": None,
                "messages_id": None,
                "error": "Could not match video path to dataset"
            }
            results.append(result)
    
    # Calculate metrics
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    
    evaluation_results = {
        "total_evaluated": total_count,
        "correct": correct_count,
        "incorrect": total_count - correct_count,
        "accuracy": accuracy,
        "results": results
    }
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Evaluation Results")
    print(f"{'='*60}")
    print(f"Total evaluated: {total_count}")
    print(f"Correct: {correct_count}")
    print(f"Incorrect: {total_count - correct_count}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"{'='*60}")
    
    return evaluation_results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Extract from specific trajectory file
        trajectory_path = sys.argv[1]
        result = extract_answer_from_trajectory(trajectory_path)
        if result:
            print(f"Extracted answer: {result['extracted_answer']}")
            print(f"Query: {result['query']}")
    else:
        # Extract from all trajectory files and evaluate
        print("Extracting answers from trajectory files...")
        extract_answers_from_directory()
        
        print("\nEvaluating answers...")
        evaluate_answers()
