from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Type
from pathlib import Path
import sys
import tempfile
import uuid
import json

import numpy as np
from huggingface_hub import hf_hub_download

try:
    import cv2  # type: ignore
except Exception as e:  # pragma: no cover
    cv2 = None  # lazy import error handled in _ensure_dependencies

import torch
from pydantic import BaseModel, Field, validator

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool

from config import Config

# Ensure the vendored MedSAM2 repo (tool_repos/MedSAM2-main) is importable so that
# `import sam2` succeeds even when the package is not installed via pip.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MEDSAM2_REPO = PROJECT_ROOT / "tool_repos" / "MedSAM2-main"
if MEDSAM2_REPO.exists():
    repo_str = str(MEDSAM2_REPO)
    if repo_str not in sys.path:
        sys.path.append(repo_str)


class EchoSegmentationInput(BaseModel):
    """Input schema for the Echo (ultrasound) segmentation tool.

    Supports MP4/AVI/GIF and single image (PNG/JPG). For DICOM cine, please
    convert to a standard video first or extend this tool to read DICOM directly.
    """

    video_path: str = Field(
        ..., description="Path to echo video (mp4/avi/gif) or single image (png/jpg)"
    )
    prompt_mode: Literal["auto", "points", "box", "mask"] = Field(
        "auto", description="Segmentation prompt mode: auto, points, box, or mask"
    )
    # Normalized coordinates in [0,1], labels: 1=foreground, 0=background
    # points: Optional[List[Tuple[float, float, int]]] = Field(
    #     None, description="List of (x,y,label) in normalized coords for the first frame"
    # )

    # Normalized box [x1,y1,x2,y2] in [0,1]
    box: Optional[List[float]] = Field(
        None,
        min_length=4,
        max_length=4,
        description="Normalized box (x1,y1,x2,y2) for the first frame",
    )
    mask_path: Optional[str] = Field(
        None, description="Path to an initial segmentation mask for the first frame (for 'mask' mode)"
    )
    mask_label: Optional[int] = Field(
        None,
        description="Palette label to extract from the provided mask when using dataset annotations",
    )
    mask_frame_index: Optional[int] = Field(
        0,
        ge=0,
        description="Frame index to pick when mask_path points to an annotation directory",
    )
    mask_label_map: Optional[Dict[int, int]] = Field(
        None,
        description="Mapping from palette pixel values to object IDs (e.g. {1:1,2:2,3:3,4:4})",
    )
    mask_palette: Optional[Dict[int, List[int]]] = Field(
        None,
        description="Mapping from object IDs to RGB colors for overlays (each value is [R,G,B])",
    )
    target_name: Optional[str] = Field(
        Config.DEFAULT_INITIAL_MASK_STRUCTURE or "LV",
        description="Optional target label used in metadata/filenames",
    )
    sample_rate: int = Field(
        1,
        description="Process every Nth frame for speed (1 = every frame)",
        ge=1,
    )
    output_fps: Optional[int] = Field(
        None, description="FPS for output video. Defaults to source FPS"
    )
    save_mask_video: bool = Field(True, description="Save binary mask-only video")
    save_overlay_video: bool = Field(True, description="Save overlay video")

    @validator("box")
    def _validate_box(cls, v):
        if v is not None and len(v) != 4:
            raise ValueError("box must be (x1,y1,x2,y2)")
        return v

    @validator("mask_palette")
    def _validate_palette(cls, v):
        if v is not None:
            for obj_id, color in v.items():
                if len(color) != 3:
                    raise ValueError("mask_palette colors must be RGB triplets")
                if any(c < 0 or c > 255 for c in color):
                    raise ValueError("mask_palette colors must be 0-255 integers")
        return v


class EchoSegmentationTool(BaseTool):
    """Segments cardiac chambers in echocardiography videos using MedSAM2 (HF) with SAM2 video predictor.

    - Downloads MedSAM2 checkpoint from Hugging Face by default (wanglab/MedSAM2) and builds a SAM2 video predictor.
    - Supports auto or prompted segmentation (points/box on first frame) with propagation.
    - Returns paths to generated videos (overlay and/or mask) and basic per-frame metrics.

    Note: You must supply a valid SAM2 model config YAML via `model_cfg` (from the SAM2 repo). The tool will
    auto-download the MedSAM2 checkpoint unless you provide a local `checkpoint` path. Pass the CONFIG NAME
    (e.g., 'sam2.1_hiera_t.yaml'), not a filesystem path.
    """

    name: str = "echo_segmentation"
    description: str = (
        "Segments echocardiography videos/images with MedSAM2 (SAM2-based). "
        "Downloads MedSAM2 weights from Hugging Face if needed. "
        "Input: video_path and optional prompt (points/box). "
        "Output: paths to generated videos and per-frame metrics."
    )
    args_schema: Type[BaseModel] = EchoSegmentationInput

    # Runtime
    device: Optional[str] = None
    temp_dir: Path = Path("temp")

    # Model config
    model_cfg: Optional[str] = None
    checkpoint: Optional[str] = None
    cache_dir: Optional[str] = None
    # Hugging Face model info (used if checkpoint not provided)
    model_id: Optional[str] = "wanglab/MedSAM2"
    model_filename: Optional[str] = "MedSAM2_US_Heart.pt"

    # Internal predictor (SAM2/MedSAM2 video predictor)
    _predictor: Any = None
    default_mask_path: Optional[str] = None
    default_target_name: Optional[str] = None
    sam2_config_dir: Optional[Path] = None

    def __init__(
        self,
        device: Optional[str] = None,
        temp_dir: Optional[str] = "temp",
        model_cfg: Optional[str] = None,
        checkpoint: Optional[str] = None,
        cache_dir: Optional[str] = None,
        model_id: Optional[str] = "wanglab/MedSAM2",
        model_filename: Optional[str] = "MedSAM2_US_Heart.pt",
        default_mask_path: Optional[str] = None,
        default_target_name: Optional[str] = None,
        sam2_config_dir: Optional[str] = None,
    ):
        super().__init__()
        # Resolve device:
        # 1) explicit arg if provided
        # 2) Config.DEVICE if set
        # 3) auto-detect: cuda -> mps -> cpu
        requested_device = device or getattr(Config, "DEVICE", None)
        chosen_device: Optional[str]
        if requested_device:
            rd = requested_device.lower()
            if rd in {"cuda", "gpu"}:
                if torch.cuda.is_available():
                    chosen_device = "cuda"
                elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    chosen_device = "mps"
                else:
                    chosen_device = "cpu"
            elif rd == "mps":
                if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    chosen_device = "mps"
                elif torch.cuda.is_available():
                    chosen_device = "cuda"
                else:
                    chosen_device = "cpu"
            elif rd == "cpu":
                chosen_device = "cpu"
            else:
                # Unknown hint; fall back to auto
                if torch.cuda.is_available():
                    chosen_device = "cuda"
                elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    chosen_device = "mps"
                else:
                    chosen_device = "cpu"
        else:
            if torch.cuda.is_available():
                chosen_device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                chosen_device = "mps"
            else:
                chosen_device = "cpu"
        self.device = chosen_device
        self.temp_dir = Path(temp_dir or tempfile.mkdtemp())
        self.temp_dir.mkdir(exist_ok=True, parents=True)
        self.model_cfg = model_cfg
        self.checkpoint = checkpoint
        self.cache_dir = cache_dir
        self.model_id = model_id
        self.model_filename = model_filename
        self.default_mask_path = (
            default_mask_path or Config.DEFAULT_INITIAL_MASK_PATH or None
        )
        self.default_target_name = (
            default_target_name or Config.DEFAULT_INITIAL_MASK_STRUCTURE or None
        )
        resolved_config_dir = sam2_config_dir or Config.DEFAULT_SAM2_CONFIG_DIR or None
        self.sam2_config_dir = None
        if resolved_config_dir:
            candidate_dir = Path(resolved_config_dir)
            if candidate_dir.exists():
                self.sam2_config_dir = candidate_dir

        # Lazy-load predictor on first run to avoid heavy startup if unused
        self._predictor = None

    # ------------- SAM2/MedSAM2 predictor helpers -------------
    def _ensure_dependencies(self):
        if cv2 is None:
            raise RuntimeError(
                "OpenCV (cv2) is required. Install with: pip install opencv-python"
            )
        # Torch is imported already; SAM2/MedSAM2 imports happen in _load_predictor

    def _candidate_config_dirs(self) -> List[Path]:
        """Return possible directories that contain SAM2 config YAML files."""
        dirs: List[Path] = []
        if self.sam2_config_dir and self.sam2_config_dir.is_dir():
            dirs.append(self.sam2_config_dir)
        try:
            import sam2  # type: ignore
            import os

            pkg_dir = Path(os.path.dirname(sam2.__file__)) / "configs"
            if pkg_dir.is_dir():
                dirs.append(pkg_dir)
        except Exception:
            pass
        return dirs

    def _get_sam2_config_dir(self) -> Path:
        """Return the first available SAM2 config directory or raise an error."""
        candidate_dirs = self._candidate_config_dirs()
        if candidate_dirs:
            return candidate_dirs[0]
        raise RuntimeError(
            "SAM2 configs directory not found. Install `sam2` or set SAM2_CONFIG_DIR."
        )

    def _resolve_default_model_cfg(self) -> Optional[str]:
        """Resolve a default SAM2 YAML CONFIG NAME if none provided.

        We rely on the configs packaged inside the installed `sam2` module.
        Returns a config NAME like 'sam2.1_hiera_t' if found, else None.
        """
        if self.model_cfg:
            return self.model_cfg

        candidates = [
            "sam2.1_hiera_t512.yaml",
            "sam2.1_hiera_t.yaml",
            "sam2_hiera_s.yaml",
        ]
        for cfg_dir in self._candidate_config_dirs():
            for name in candidates:
                cfg_path = cfg_dir / name
                if cfg_path.is_file():
                    return name[:-5] if name.endswith(".yaml") else name

        # If not found, return None and let caller raise a clear error.
        return None

    def _normalize_model_cfg_name(self, cfg: str) -> str:
        """Normalize user-provided model_cfg to a config NAME for Hydra.

        - If a filesystem path is provided, reduce to basename.
        - Fix common typos: 'sam2.1.hiera' -> 'sam2.1_hiera'.
        - Remove .yaml extension as Hydra expects just the config name.
        """
        try:
            p = Path(cfg)
            if p.exists():
                cfg = p.name
        except Exception:
            pass
        if "sam2.1.hiera" in cfg:
            cfg = cfg.replace("sam2.1.hiera", "sam2.1_hiera")

        # Remove .yaml extension - Hydra expects just the config name
        if cfg.endswith('.yaml'):
            cfg = cfg[:-5]

        return cfg

    def _load_predictor(self):
        """Load the SAM2 video predictor with MedSAM2 weights.

        If `checkpoint` is not provided, attempt to download from Hugging Face using
        `model_id` and `model_filename` (defaults target the ultrasound heart model).
        A valid SAM2 YAML config NAME is required; if not provided, we try to resolve a default.
        """
        if self._predictor is not None:
            return

        # Ensure checkpoint (local or download)
        if not self.checkpoint:
            if not self.model_id or not self.model_filename:
                raise RuntimeError(
                    "Either provide `checkpoint` or set (`model_id`, `model_filename`) to download MedSAM2."
                )
            try:
                ckpt_path = hf_hub_download(
                    repo_id=self.model_id,
                    filename=self.model_filename,
                    local_dir=self.cache_dir or str(self.temp_dir / "hf_cache"),
                    local_dir_use_symlinks=False,
                )
                self.checkpoint = ckpt_path
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download MedSAM2 checkpoint from Hugging Face ({self.model_id}/{self.model_filename}): {e}"
                )

        # Ensure a model config NAME
        if not self.model_cfg:
            self.model_cfg = self._resolve_default_model_cfg()
        if not self.model_cfg:
            raise RuntimeError(
                "Could not resolve a SAM2 config automatically. Install `sam2` and pass a config NAME, e.g., --model-cfg sam2.1_hiera_t.yaml"
            )

        cfg_name = self._normalize_model_cfg_name(self.model_cfg)

        try:
            # Build SAM2 video predictor with MedSAM2 weights
            from sam2.build_sam import build_sam2_video_predictor  # type: ignore
            from hydra.core.global_hydra import GlobalHydra
            from hydra import initialize_config_dir

            # Clear any existing Hydra configuration to avoid conflicts
            GlobalHydra.instance().clear()
            sam2_configs_dir = self._get_sam2_config_dir()

            # Initialize Hydra with SAM2 configs directory
            with initialize_config_dir(config_dir=str(sam2_configs_dir), version_base=None):
                predictor = build_sam2_video_predictor(
                    cfg_name, self.checkpoint, device=self.device
                )
        except Exception as e:
            raise RuntimeError(
                f"Failed to build predictor with MedSAM2 weights. Config: '{cfg_name}', "
                f"Checkpoint: '{self.checkpoint}'. Error: {e}"
            )

        self._predictor = predictor

    # ------------- Video IO helpers -------------
    def _read_video(self, path: str) -> Tuple[List[np.ndarray], float]:
        """Read video into list of RGB frames and return frames + fps.
        If it's an image, return single frame and default fps=25.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Video/image not found: {path}")

        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError("Failed to read image.")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return [img], 25.0

        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            raise RuntimeError("Failed to open video.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames: List[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        if not frames:
            raise RuntimeError("No frames read from video.")
        return frames, float(fps)

    def _write_video(self, frames: List[np.ndarray], fps: float, out_path: Path):
        out_path.parent.mkdir(exist_ok=True, parents=True)
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"H264")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        for fr in frames:
            bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
        writer.release()

    def _export_frames_to_jpeg_dir(self, frames: List[np.ndarray]) -> Path:
        """
        Export RGB frames to a temporary JPEG directory for SAM2, which accepts a folder of images.
        """
        out_dir = self.temp_dir / f"sam2_frames_{uuid.uuid4().hex[:8]}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, fr in enumerate(frames):
            bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
            filename = out_dir / f"{idx:06d}.jpg"
            cv2.imwrite(str(filename), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return out_dir

    # ------------- Segmentation core -------------
    def _normalized_to_abs_points(self, points: List[Tuple[float, float, int]], w: int, h: int):
        coords = np.array([[int(x * w), int(y * h)] for x, y, _ in points], dtype=np.int32)
        labels = np.array([int(lbl) for _, _, lbl in points], dtype=np.int32)
        return coords, labels

    def _normalized_to_abs_box(self, box: Sequence[float], w: int, h: int):
        x1, y1, x2, y2 = box
        return np.array([int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)], dtype=np.int32)

    def _compose_color_layer(
        self,
        masks: Dict[int, np.ndarray],
        palette: Dict[int, Tuple[int, int, int]],
        fallback: Tuple[Tuple[int, int, int], ...],
    ) -> np.ndarray:
        """Create an RGB layer where each object mask is painted with its palette color."""

        # Determine output spatial size from first mask
        sample_mask = next(iter(masks.values()))
        # Canonicalize sample_mask to 2D shape for sizing
        if sample_mask.ndim == 3:
            sample_mask = np.squeeze(sample_mask)
            if sample_mask.ndim == 3:
                sample_mask = sample_mask[..., 0]
        height, width = sample_mask.shape[:2]
        color_layer = np.zeros((height, width, 3), dtype=np.uint8)

        for obj_id, mask in masks.items():
            m = np.asarray(mask)
            # Squeeze singleton dims and enforce 2D (H, W)
            if m.ndim == 3:
                m = np.squeeze(m)
                if m.ndim == 3:
                    m = m[..., 0]
            if m.ndim != 2:
                # Fallback: best-effort reshape if total size matches
                try:
                    m = m.reshape((height, width))
                except Exception:
                    m = np.array(m, dtype=np.uint8)
            if m.shape != (height, width):
                m = cv2.resize(m.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
            mask_bool = (m.astype(np.uint8) > 0)
            color = palette.get(obj_id)
            if color is None:
                color = fallback[obj_id % len(fallback)]
            # Broadcast assignment across channel dimension
            color_layer[mask_bool] = np.array(color, dtype=np.uint8)

        return color_layer

    def _render_overlay(
        self,
        frame: np.ndarray,
        masks: Dict[int, np.ndarray],
        palette: Dict[int, Tuple[int, int, int]],
        fallback: Tuple[Tuple[int, int, int], ...],
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Alpha blend colorized masks onto the frame."""

        color_layer = self._compose_color_layer(masks, palette, fallback)
        overlay = cv2.addWeighted(frame, 1 - alpha, color_layer, alpha, 0)
        return overlay

    def _load_mask_prompt(
        self,
        mask_path: str,
        frame_shape: Tuple[int, int],
        mask_label: Optional[int] = None,
        mask_frame_index: Optional[int] = 0,
        mask_label_map: Optional[Dict[int, int]] = None,
    ) -> Dict[int, np.ndarray]:
        """Load prompt masks (object_id -> binary mask) from annotation."""

        if mask_path is None:
            raise ValueError("mask_path must be provided for mask prompts")

        candidate = Path(mask_path)
        if candidate.is_dir():
            if mask_frame_index is None:
                mask_frame_index = 0
            frame_name = f"{int(mask_frame_index):04d}.png"
            candidate = candidate / frame_name

        if not candidate.exists():
            raise FileNotFoundError(f"Mask prompt not found at {candidate}")

        mask = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask prompt: {candidate}")

        height, width = frame_shape
        if mask.shape != frame_shape:
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

        prompt_masks: Dict[int, np.ndarray] = {}

        if mask_label_map:
            for pixel_value, obj_id in mask_label_map.items():
                prompt_masks[int(obj_id)] = (mask == pixel_value).astype(np.uint8)
        elif mask_label is not None:
            prompt_masks[1] = (mask == mask_label).astype(np.uint8)
        else:
            prompt_masks[1] = (mask > 0).astype(np.uint8)

        if not prompt_masks:
            raise RuntimeError("No foreground objects extracted from mask prompt")

        return prompt_masks

    def _run(
        self,
        video_path: str,
        prompt_mode: Literal["auto", "points", "box", "mask"] = "auto",
        points: Optional[List[Tuple[float, float, int]]] = None,
        box: Optional[Sequence[float]] = None,
        mask_path: Optional[str] = None,
        mask_label: Optional[int] = None,
        mask_frame_index: Optional[int] = 0,
        mask_label_map: Optional[Dict[int, int]] = None,
        mask_palette: Optional[Dict[int, List[int]]] = None,
        target_name: Optional[str] = "LV",
        sample_rate: int = 1,
        output_fps: Optional[int] = None,
        save_mask_video: bool = True,
        save_overlay_video: bool = True,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Run MedSAM2/SAM2 video segmentation on an echo video or image.

        Returns (output, metadata), where output contains file paths;
        metadata contains additional info and basic per-frame metrics.
        """
        self._ensure_dependencies()

        # Load predictor lazily
        self._load_predictor()
        predictor = self._predictor

        # Get video info for output formatting
        frames, src_fps = self._read_video(video_path)
        fps = float(output_fps) if output_fps else src_fps
        h, w = frames[0].shape[:2]

        default_palette = {
            1: (0, 255, 0),  # LV - green
            2: (255, 0, 0),  # RV - red
            3: (255, 255, 0),  # LA - yellow
            4: (0, 0, 255),  # RA - blue
            5: (255, 0, 255),  # myocardium/other
        }
        palette_rgb: Dict[int, Tuple[int, int, int]] = dict(default_palette)
        if mask_palette:
            palette_rgb.update(
                {
                    int(obj_id): tuple(int(c) for c in color)
                    for obj_id, color in mask_palette.items()
                }
            )
        fallback_colors: Tuple[Tuple[int, int, int], ...] = (
            (0, 255, 0),
            (255, 0, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        )
        active_object_ids: List[int] = []

        # Initialize video state (SAM2 expects video path directly)
        try:
            # SAM2 wants the video file path, not processed frames
            state = predictor.init_state(video_path)
        except Exception as e:
            err_text = f"{e}".lower()
            # If decord is missing, fall back to exporting frames as JPEGs and use folder input
            if "decord" in err_text or "no module named 'decord'" in err_text:
                try:
                    frames_dir = self._export_frames_to_jpeg_dir(frames)
                    state = predictor.init_state(str(frames_dir))
                except Exception as e2:
                    raise RuntimeError(
                        "decord is not installed and JPEG folder fallback failed. "
                        "Install decord (pip install decord) or ensure SAM2 supports your video format. "
                        f"Original error: {e}; Fallback error: {e2}"
                    )
            else:
                raise RuntimeError(
                    f"Failed to initialize SAM2 state with video: {video_path}. "
                    f"SAM2 may only support MP4 videos and JPEG folders. Error: {e}"
                )

        # Resolve default prompting configuration
        resolved_mask_path = mask_path or self.default_mask_path
        effective_prompt_mode = prompt_mode
        
        # Check if mask path exists before using it
        mask_path_exists = False
        if resolved_mask_path:
            mask_candidate = Path(resolved_mask_path)
            if mask_candidate.is_dir():
                if mask_frame_index is not None:
                    frame_name = f"{int(mask_frame_index):04d}.png"
                    mask_candidate = mask_candidate / frame_name
            mask_path_exists = mask_candidate.exists() if mask_candidate else False
        
        if resolved_mask_path and mask_path_exists and effective_prompt_mode == "auto":
            effective_prompt_mode = "mask"
        elif effective_prompt_mode == "auto":
            # If mask doesn't exist and mode is auto, fall back to center point
            effective_prompt_mode = "points"

        if not target_name and self.default_target_name:
            target_name = self.default_target_name
        target_name = target_name or "LV"

        # Feed prompt to predictor on first frame
        try:
            if state is None:
                raise RuntimeError("SAM2 state initialization failed")

            if effective_prompt_mode == "mask" and resolved_mask_path and mask_path_exists:
                try:
                    prompt_masks = self._load_mask_prompt(
                        resolved_mask_path,
                        (h, w),
                        mask_label=mask_label,
                        mask_frame_index=mask_frame_index,
                        mask_label_map=mask_label_map,
                    )
                    for obj_id, obj_mask in prompt_masks.items():
                        predictor.add_new_mask(
                            state,
                            frame_idx=0,
                            obj_id=int(obj_id),
                            mask=obj_mask.astype(bool),
                        )
                        active_object_ids.append(int(obj_id))
                except (FileNotFoundError, RuntimeError) as mask_error:
                    # If mask loading fails, fall back to center point prompting
                    print(f"Warning: Could not load mask prompt ({mask_error}), falling back to center point prompting.")
                    center_x, center_y = w // 2, h // 2
                    center_points = np.array([[center_x, center_y]])
                    center_labels = np.array([1])
                    predictor.add_new_points(
                        state,
                        frame_idx=0,
                        obj_id=1,
                        points=center_points,
                        labels=center_labels,
                    )
                    active_object_ids.append(1)
            elif effective_prompt_mode == "points" and points:
                abs_points, point_labels = self._normalized_to_abs_points(points, w, h)
                predictor.add_new_points(
                    state,
                    frame_idx=0,
                    obj_id=1,
                    points=abs_points,
                    labels=point_labels,
                )
                active_object_ids.append(1)
            elif effective_prompt_mode == "box" and box:
                abs_box = self._normalized_to_abs_box(box, w, h)
                predictor.add_new_points_or_box(
                    state,
                    frame_idx=0,
                    obj_id=1,
                    box=abs_box,
                )
                active_object_ids.append(1)
            else:
                # Default: use center point as prompt
                center_x, center_y = w // 2, h // 2
                center_points = np.array([[center_x, center_y]])
                center_labels = np.array([1])
                predictor.add_new_points(
                    state,
                    frame_idx=0,
                    obj_id=1,
                    points=center_points,
                    labels=center_labels,
                )
                active_object_ids.append(1)
        except Exception as e:
            raise RuntimeError(
                f"Prompting API mismatch. Please adapt the add_new_points calls "
                f"to your installed SAM2/MedSAM2 version. Error: {e}"
            )

        # Propagate segmentation across frames
        mask_frames: List[Dict[int, np.ndarray]] = []
        overlay_frames: List[np.ndarray] = []
        per_frame_metrics: List[Dict[str, Any]] = []

        try:
            for out in predictor.propagate_in_video(state):
                if not (isinstance(out, tuple) and len(out) == 3):
                    continue

                frame_idx, obj_ids, mask_logits = out
                if len(mask_logits) == 0:
                    continue

                frame_masks: Dict[int, np.ndarray] = {}
                for idx, obj_id in enumerate(obj_ids):
                    logits = mask_logits[idx]
                    mask_np = torch.sigmoid(logits).detach().cpu().numpy()
                    # Ensure mask becomes 2D binary (H, W)
                    mask_np = np.squeeze(mask_np)
                    if mask_np.ndim == 3:
                        mask_np = mask_np[..., 0]
                    mask_bin = (mask_np > 0.5).astype(np.uint8)
                    if mask_bin.shape != (h, w):
                        mask_bin = cv2.resize(mask_bin.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                    frame_masks[int(obj_id)] = mask_bin

                if not frame_masks:
                    continue

                mask_frames.append(frame_masks)

                if save_overlay_video and frame_idx < len(frames):
                    overlay = self._render_overlay(
                        frames[frame_idx], frame_masks, palette_rgb, fallback_colors
                    )
                    overlay_frames.append(overlay)

                per_frame_metrics.append(
                    {
                        "frame_index": int(frame_idx),
                        "object_areas": {
                            int(obj_id): int(mask.sum()) for obj_id, mask in frame_masks.items()
                        },
                    }
                )
        except Exception as e:
            raise RuntimeError(f"Error during propagation: {e}")

        # Write outputs
        out_base = f"echo_seg_{target_name}_{uuid.uuid4().hex[:8]}"
        outputs: Dict[str, Any] = {}

        if save_overlay_video and overlay_frames:
            overlay_path = self.temp_dir / f"{out_base}_overlay.mp4"
            self._write_video(overlay_frames, fps, overlay_path)
            outputs["overlay_video_path"] = str(overlay_path)

        if save_mask_video and mask_frames:
            mask_rgb_frames: List[np.ndarray] = []
            for frame_masks in mask_frames:
                color_layer = self._compose_color_layer(frame_masks, palette_rgb, fallback_colors)
                mask_rgb_frames.append(color_layer)
            mask_path = self.temp_dir / f"{out_base}_mask.mp4"
            self._write_video(mask_rgb_frames, fps, mask_path)
            outputs["mask_video_path"] = str(mask_path)

        metadata: Dict[str, Any] = {
            "video_path": video_path,
            "frames_processed": len(mask_frames),
            "source_frames": len(frames),
            "sample_rate": sample_rate,
            "fps_out": fps,
            "resolution": [h, w],
            "target_name": target_name,
            "active_object_ids": sorted(set(active_object_ids) or {1}),
            "per_frame_metrics": per_frame_metrics,
            "analysis_status": "completed",
        }

        return outputs, metadata

    async def _arun(
        self,
        video_path: str,
        prompt_mode: Literal["auto", "points", "box", "mask"] = "auto",
        points: Optional[List[Tuple[float, float, int]]] = None,
        box: Optional[Sequence[float]] = None,
        mask_path: Optional[str] = None,
        mask_label: Optional[int] = None,
        mask_frame_index: Optional[int] = 0,
        mask_label_map: Optional[Dict[int, int]] = None,
        mask_palette: Optional[Dict[int, List[int]]] = None,
        target_name: Optional[str] = "LV",
        sample_rate: int = 1,
        output_fps: Optional[int] = None,
        save_mask_video: bool = True,
        save_overlay_video: bool = True,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        return self._run(
            video_path,
            prompt_mode,
            points,
            box,
            mask_path,
            mask_label,
            mask_frame_index,
            mask_label_map,
            mask_palette,
            target_name,
            sample_rate,
            output_fps,
            save_mask_video,
            save_overlay_video,
        )
