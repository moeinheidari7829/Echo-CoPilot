#!/usr/bin/env python3
"""
Visualize self-contrast mechanism with detailed perspective breakdown.
Creates a markdown report showing how the voting works.
"""

import json
import sys
import re
from pathlib import Path
from collections import Counter


def extract_answer_simple(text: str) -> str:
    """Simple answer extraction."""
    if not text:
        return '?'
    
    # JSON format
    json_match = re.search(r'"final_answer"\s*:\s*"([A-D])"', text, re.I)
    if json_match:
        return json_match.group(1).upper()
    
    # Explicit answer
    patterns = [
        r'\*\*Answer[:\s]+([A-D])\s*[–-]',
        r'Answer[:\s]+([A-D])',
        r'([A-D])\s*[–-]\s*(?:Normal|Mild|Moderate|Severe|Yes|No)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return match.group(1).upper()
    
    return '?'


def create_markdown_report(trajectory_path: str, output_path: str = None):
    """Create a markdown report showing self-contrast voting."""
    
    with open(trajectory_path, 'r') as f:
        traj = json.load(f)
    
    metadata = traj.get('metadata', {})
    query = metadata.get('query', '')
    success = metadata.get('success', False)
    execution_time = metadata.get('execution_time_seconds', 0)
    
    # Start markdown
    md = []
    md.append("# Self-Contrast Voting Example\n")
    md.append(f"**File**: `{Path(trajectory_path).name}`\n")
    md.append(f"**Status**: {'✓ Success' if success else '✗ Failed'}\n")
    md.append(f"**Execution Time**: {execution_time:.1f}s\n")
    md.append("\n---\n")
    
    # Question
    md.append("## Question\n")
    md.append(f"```\n{query}\n```\n")
    md.append("\n---\n")
    
    # Perspectives
    perspectives = traj.get('self_contrast', {}).get('perspectives', [])
    
    if perspectives:
        md.append("## Three Perspectives\n")
        
        perspective_votes = []
        
        for i, p in enumerate(perspectives, 1):
            name = p.get('name', f'Perspective {i}')
            response = p.get('response', '')
            tools_used = p.get('tools_used', [])
            
            suggested_answer = extract_answer_simple(response)
            perspective_votes.append(suggested_answer)
            
            md.append(f"### {i}. {name}\n")
            md.append(f"**Tools used**: {', '.join(tools_used) if tools_used else 'None'}\n\n")
            md.append(f"**Suggested answer**: **{suggested_answer}**\n\n")
            md.append(f"**Reasoning**:\n")
            md.append(f"```\n{response[:400]}...\n```\n")
            md.append("\n")
        
        md.append("---\n")
        md.append("## Voting Summary\n")
        
        vote_counts = Counter(perspective_votes)
        for answer, count in sorted(vote_counts.items()):
            bar = "█" * count
            md.append(f"- **Option {answer}**: {count} vote(s) {bar}\n")
        
        majority = vote_counts.most_common(1)[0][0]
        md.append(f"\n**Majority vote**: **{majority}**\n")
        md.append("\n---\n")
    
    # Final decision
    final_response = traj.get('final_response', '')
    final_answer = extract_answer_simple(final_response)
    
    md.append("## Final Decision\n")
    md.append(f"**Selected answer**: **{final_answer}**\n\n")
    md.append(f"**Full reasoning**:\n")
    md.append(f"```\n{final_response[:800]}\n...\n```\n")
    
    # Combine markdown
    markdown_content = "\n".join(md)
    
    # Print to console
    print(markdown_content)
    
    # Save to file
    if output_path:
        with open(output_path, 'w') as f:
            f.write(markdown_content)
        print(f"\n\n✓ Saved to: {output_path}")
    
    return markdown_content


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize self-contrast voting")
    parser.add_argument("trajectory", nargs='?', 
                       default="experiment/stability_tests/agent_fresh_echoprime_20260131_011921/trajectories/trajectory_20260131_145537.json",
                       help="Path to trajectory JSON file")
    parser.add_argument("--output", "-o", help="Output markdown file path")
    
    args = parser.parse_args()
    
    if not Path(args.trajectory).exists():
        print(f"Error: File not found: {args.trajectory}")
        sys.exit(1)
    
    create_markdown_report(args.trajectory, args.output)


if __name__ == "__main__":
    main()
