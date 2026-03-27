"""
Configuration settings for the EchoPilot agent.
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv  # Add this import

load_dotenv()
class Config:
    """Configuration class for EchoPilot."""

    _PROJECT_ROOT = Path(__file__).resolve().parent

    # API Configuration
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    OPENAI_MAX_TOKENS: int = int(os.getenv("OPENAI_MAX_TOKENS", "1000"))
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    # Optional reasoning controls for models that support it (e.g., gpt-5, o3-mini).
    # Effort expected values: "low", "medium", "high".
    OPENAI_REASONING_EFFORT: Optional[str] = os.getenv("OPENAI_REASONING_EFFORT")
    # Summary expected values: "off", "on", "auto" (or as supported by the backend).
    OPENAI_REASONING_SUMMARY: Optional[str] = os.getenv("OPENAI_REASONING_SUMMARY")

    # Video Configuration
    DEFAULT_VIDEO_PATH: str = os.getenv("VIDEO_PATH", str((_PROJECT_ROOT / "videos" / "val1.mp4")))
    DEFAULT_PATIENT_ID: str = os.getenv("PATIENT_ID", "PATIENT-001")

    # Analysis Configuration
    MAX_VIDEOS: int = int(os.getenv("MAX_VIDEOS", "1"))
    INCLUDE_CONFIDENCE: bool = os.getenv("INCLUDE_CONFIDENCE", "true").lower() == "true"
    SAVE_RESULTS: bool = os.getenv("SAVE_RESULTS", "true").lower() == "true"

    # Output Configuration
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", str((_PROJECT_ROOT / "outputs")))
    RESULTS_FILE: str = os.getenv("RESULTS_FILE", "echo_analysis_results.json")
    STATE_FILE: str = os.getenv("STATE_FILE", "final_analysis_state.json")

    # Model Configuration
    DEVICE: str = os.getenv("DEVICE", "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu")

    # Self-Contrast Configuration
    USE_SELF_CONTRAST: bool = os.getenv("USE_SELF_CONTRAST", "true").lower() == "true"
    CONTRAST_LLM_MODEL: str = os.getenv("CONTRAST_LLM_MODEL", "gpt-4o-mini")

    # Measurement Tool Configuration
    # Options: "echoprime", "echonet", "both"
    MEASUREMENT_TOOL: str = os.getenv("MEASUREMENT_TOOL", "echonet")

    # Segmentation Prompt Configuration
    # Optional default initial mask (first-frame) applied to segmentation when none is provided explicitly
    DEFAULT_INITIAL_MASK_PATH: Optional[str] = os.getenv(
        "ECHO_INITIAL_MASK_PATH", None
    )
    DEFAULT_INITIAL_MASK_STRUCTURE: str = os.getenv("ECHO_INITIAL_MASK_STRUCTURE", "LV")
    DEFAULT_SAM2_CONFIG_DIR: str = os.getenv(
        "SAM2_CONFIG_DIR",
        str(_PROJECT_ROOT / "tool_repos" / "MedSAM2-main" / "sam2" / "configs"),
    )

    # Annotation-based prompting configuration
    # Maps video stem (or filename) -> annotation metadata used to seed MedSAM2.
    _ANNOTATION_PROMPTS_PATH = os.getenv("ECHO_ANNOTATION_PROMPTS")
    _DEFAULT_ANNOTATIONS_FILE = _PROJECT_ROOT / "assets" / "annotation_prompts.json"

    if _ANNOTATION_PROMPTS_PATH:
        try:
            with open(_ANNOTATION_PROMPTS_PATH, "r", encoding="utf-8") as annotations_file:
                ANNOTATION_PROMPTS: Dict[str, Dict[str, Any]] = json.load(annotations_file)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"⚠️  Unable to load annotation prompts from {_ANNOTATION_PROMPTS_PATH}: {exc}")
            ANNOTATION_PROMPTS = {}
    elif _DEFAULT_ANNOTATIONS_FILE.exists():
        try:
            with open(_DEFAULT_ANNOTATIONS_FILE, "r", encoding="utf-8") as annotations_file:
                ANNOTATION_PROMPTS = json.load(annotations_file)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"⚠️  Unable to load default annotation prompts: {exc}")
            ANNOTATION_PROMPTS = {}
    else:
        ANNOTATION_PROMPTS = {}

    @classmethod
    def validate(cls) -> bool:
        """Validate configuration."""
        if not cls.OPENAI_API_KEY:
            print("❌ OPENAI_API_KEY not set. Please set it as an environment variable.")
            return False

        if not os.path.exists(cls.DEFAULT_VIDEO_PATH):
            print(f"❌ Video file not found: {cls.DEFAULT_VIDEO_PATH}")
            return False

        return True

    @classmethod
    def get_video_path(cls, video_path: Optional[str] = None) -> str:
        """Get video path, using provided path or default."""
        return video_path or cls.DEFAULT_VIDEO_PATH

    @classmethod
    def get_patient_id(cls, patient_id: Optional[str] = None) -> str:
        """Get patient ID, using provided ID or default."""
        return patient_id or cls.DEFAULT_PATIENT_ID
