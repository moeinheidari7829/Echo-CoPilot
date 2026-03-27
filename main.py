#!/usr/bin/env python3
"""
Command-line entry point for the EchoPilot ReAct agent.

Example:
    python -m echo-agent.main --video path/to/video.mp4 --query "Assess LV systolic function."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from agents import get_intelligent_agent
from config import Config


# ANSI color codes for cross-platform compatibility
class Colors:
    """ANSI color codes for terminal output."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    # Text colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Background colors
    BG_BLUE = '\033[44m'
    BG_CYAN = '\033[46m'
    BG_GREEN = '\033[42m'
    
    # Bright colors
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'


def safe_print(text: str) -> None:
    """Safely print text, handling Unicode encoding errors."""
    try:
        print(text)
    except UnicodeEncodeError:
        # Fallback: remove or replace problematic Unicode characters
        safe_text = text.encode('ascii', errors='replace').decode('ascii', errors='replace')
        print(safe_text)


def print_banner():
    """Print a colorful banner for the CLI."""
    # Create a perfectly aligned banner with EchoPilot theme (ASCII-safe)
    banner_lines = [
        "==================================================================",
        "  EchoPilot - Intelligent Echocardiography Analysis System",
        "",
        "  AI-Powered Clinical Decision Support for Cardiac Imaging",
        "  Self-Contrast Multi-Perspective Analysis Framework",
        "=================================================================="
    ]
    
    safe_print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}")
    for line in banner_lines:
        safe_print(line)
    safe_print(f"{Colors.RESET}\n")


def initialize_agent(device: Optional[str] = None):
    """Instantiate the intelligent agent in a single helper, similar to MedRAX."""
    IntelligentAgent, _ = get_intelligent_agent()
    return IntelligentAgent(device=device or Config.DEVICE)


def safe_print(text: str) -> None:
    """Safely print text, handling Unicode encoding errors."""
    try:
        print(text)
    except UnicodeEncodeError:
        # Fallback: remove or replace problematic Unicode characters
        safe_text = text.encode('ascii', errors='replace').decode('ascii', errors='replace')
        print(safe_text)


def run_cli(video_path: str, query: str) -> None:
    """Run the EchoPilot agent and display results."""
    safe_print(f"{Colors.BRIGHT_BLUE}{Colors.BOLD}[*] Initializing EchoPilot Agent...{Colors.RESET}\n")
    
    agent = initialize_agent()
    
    safe_print(f"{Colors.BRIGHT_BLUE}{Colors.BOLD}[Video]{Colors.RESET} {Colors.DIM}{video_path}{Colors.RESET}")
    safe_print(f"{Colors.BRIGHT_BLUE}{Colors.BOLD}[Query]{Colors.RESET} {Colors.BOLD}{query}{Colors.RESET}\n")
    safe_print(f"{Colors.BRIGHT_CYAN}{'-' * 60}{Colors.RESET}")
    safe_print(f"{Colors.BRIGHT_GREEN}{Colors.BOLD}[*] Processing...{Colors.RESET}\n")
    
    response = agent.process_query(query, video_path)

    if not response.success:
        error_msg = response.execution_result.error if hasattr(response, 'execution_result') else "Unknown error"
        safe_print(f"\n{Colors.RED}{Colors.BOLD}[ERROR] Analysis Failed{Colors.RESET}")
        safe_print(f"{Colors.RED}Error: {error_msg}{Colors.RESET}\n")
        sys.exit(1)

    safe_print(f"\n{Colors.BRIGHT_GREEN}{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    safe_print(f"{Colors.BRIGHT_GREEN}{Colors.BOLD}[SUCCESS] EchoPilot Analysis Complete{Colors.RESET}")
    safe_print(f"{Colors.BRIGHT_GREEN}{Colors.BOLD}{'=' * 60}{Colors.RESET}\n")
    safe_print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}Question:{Colors.RESET} {query}\n")
    safe_print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}Answer:{Colors.RESET}")
    safe_print(f"{Colors.WHITE}{response.response_text}{Colors.RESET}\n")
    safe_print(f"{Colors.DIM}{'-' * 60}{Colors.RESET}\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments with enhanced help."""
    parser = argparse.ArgumentParser(
        description=f"{Colors.BRIGHT_CYAN}EchoPilot - Intelligent Echocardiography Analysis System{Colors.RESET}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
{Colors.BRIGHT_GREEN}Examples:{Colors.RESET}
  # Analyze a specific video with a custom query
  python main.py --video path/to/echo_video.mp4 --query "Assess LV systolic function"
  
  # Use default video path with a question
  python main.py --query "What is the severity of mitral regurgitation?"
  
  # Quick evaluation with default settings
  python main.py --video MIMICEchoQA/files/p16/p16233404/s97838604/97838604_0069.mp4

{Colors.BRIGHT_YELLOW}Note:{Colors.RESET}
  - The agent uses a self-contrast framework with multiple perspectives
  - Knowledge graph and RAG are automatically utilized when relevant
  - Results include structured JSON output for easy parsing
        """.strip()
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        metavar="PATH",
        help=f"{Colors.CYAN}Path to the echocardiography video file{Colors.RESET} (required if no default configured)"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Evaluate cardiac function.",
        metavar="QUESTION",
        help=f"{Colors.CYAN}Clinical question or query for the agent{Colors.RESET} (default: 'Evaluate cardiac function.')"
    )
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Suppress the banner display"
    )
    return parser.parse_args()


def resolve_video_path(cli_path: Optional[str]) -> str:
    if cli_path:
        video = Path(cli_path).expanduser().resolve()
        if not video.is_file():
            raise FileNotFoundError(f"Video not found: {video}")
        return str(video)

    default_path = Path(Config.get_video_path()).expanduser().resolve()
    if not default_path.is_file():
        raise FileNotFoundError(
            f"No video provided and default path not found: {default_path}. "
            "Use --video to specify a file."
        )
    return str(default_path)


def main() -> None:
    """Main entry point with banner and argument parsing."""
    args = parse_args()
    
    # Display banner unless suppressed
    if not args.no_banner:
        print_banner()
    
    try:
        video_path = resolve_video_path(args.video)
        run_cli(video_path, args.query.strip())
    except FileNotFoundError as e:
        safe_print(f"\n{Colors.RED}{Colors.BOLD}[ERROR] File Not Found{Colors.RESET}")
        safe_print(f"{Colors.RED}{str(e)}{Colors.RESET}\n")
        sys.exit(1)
    except KeyboardInterrupt:
        safe_print(f"\n\n{Colors.YELLOW}{Colors.BOLD}[!] Interrupted by user{Colors.RESET}\n")
        sys.exit(130)
    except Exception as e:
        safe_print(f"\n{Colors.RED}{Colors.BOLD}[ERROR] Unexpected Error{Colors.RESET}")
        safe_print(f"{Colors.RED}{str(e)}{Colors.RESET}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
