"""Video utility functions for codec conversion and temp handling."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def _probe_codec(video_path: str) -> Optional[str]:
    """Return the codec name for the first video stream using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "csv=p=0",
                video_path,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip().lower()
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return None


def convert_video_to_h264(input_path: str, output_path: Optional[str] = None) -> str:
    """Convert *input_path* to H.264 if needed and return the playable path.

    If the file already uses an H.264 video stream, the original path is
    returned. When conversion is required, a new MP4 file is written (leaving
    the original untouched). The new path is returned, or the original path if
    conversion fails.
    """

    if not input_path:
        return input_path

    input_path = str(Path(input_path))
    if not os.path.isfile(input_path):
        return input_path

    codec = _probe_codec(input_path)
    if codec and "h264" in codec:
        # Already compatible
        return input_path

    base = Path(input_path)
    if output_path is None:
        output_path = str(base.with_name(f"{base.stem}_h264.mp4"))

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-vcodec",
            "libx264",
            "-acodec",
            "aac",
            "-pix_fmt",
            "yuv420p",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0 and os.path.isfile(output_path):
            return output_path
        # If conversion failed, fall back to original path.
        return input_path
    except FileNotFoundError:
        # ffmpeg missing; nothing we can do besides returning original
        return input_path
    except Exception:
        return input_path


def copy_video_to_temp(video_path: str, temp_dir: Optional[str] = None) -> str:
    """Copy *video_path* to *temp_dir* (or a default temp) and return new path."""

    if not video_path:
        return video_path

    if temp_dir is None:
        temp_dir = Path("/tmp") / f"video_temp_{os.getpid()}"
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    src = Path(video_path)
    if not src.exists():
        return video_path

    dest = temp_dir / src.name
    try:
        shutil.copy2(src, dest)
        return str(dest)
    except Exception:
        return video_path
