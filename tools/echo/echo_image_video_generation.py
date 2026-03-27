from __future__ import annotations

from typing import Any, Dict, List, Optional, Type
from pathlib import Path
import os
import sys
import json
import shutil
import datetime
import subprocess

from pydantic import BaseModel, Field, field_validator
from langchain_core.tools import BaseTool
from langchain_core.callbacks import (
    CallbackManagerForToolRun,
    AsyncCallbackManagerForToolRun,
)


class EchoSynGenerationInput(BaseModel):
    """Generate synthetic echo videos using EchoNet-Synthetic (LIDM + Privacy + LVDM)."""

    dataset: str = Field(
        "dynamic",
        description="Target dataset flavor: one of ['dynamic', 'ped_a4c', 'ped_psax']",
    )
    num_samples: int = Field(4, ge=1, description="Number of videos to generate.")
    batch_size: int = Field(8, ge=1, description="Batch size for sampling.")

    num_steps_img: int = Field(64, ge=1, description="Sampling steps for LIDM (latent images).")
    num_steps_vid: int = Field(64, ge=1, description="Sampling steps for LVDM (videos).")
    frames: int = Field(64, ge=32, description="Number of frames per video (multiple of 32).")
    min_lvef: int = Field(20, ge=0, le=100, description="Minimum EF for conditioning.")
    max_lvef: int = Field(80, ge=0, le=100, description="Maximum EF for conditioning.")
    save_as: str = Field("mp4", description="Comma-separated formats: avi,mp4,gif,jpg,png,pt")

    use_privacy_filter: bool = Field(
        False,
        description="Apply EchoNet-Synthetic privacy filter (requires reference latents).",
    )
    reference_latents_dir: Optional[str] = Field(
        None,
        description="Path to real latent dataset (with FileList.csv and Latents/) if privacy filter is used.",
    )

    outdir: Optional[str] = Field(
        None, description="Output directory; default is temp/echosyn_run_<UTC_TIMESTAMP>"
    )
    repo_root: Optional[str] = Field(
        None,
        description="Path to EchoNet-Synthetic repo root. Defaults to tool_repos/EchoNet-Synthetic",
    )
    models_dir: Optional[str] = Field(
        None,
        description="Path to EchoNet-Synthetic models dir. Defaults to <repo_root>/models",
    )

    @field_validator("dataset")
    @classmethod
    def _validate_dataset(cls, v: str) -> str:
        allowed = {"dynamic", "ped_a4c", "ped_psax"}
        if v not in allowed:
            raise ValueError(f"dataset must be one of {sorted(allowed)}")
        return v

    @field_validator("max_lvef")
    @classmethod
    def _validate_ef_range(cls, v: int, info) -> int:
        min_v = info.data.get("min_lvef", 0)
        if v < min_v:
            raise ValueError("max_lvef must be >= min_lvef")
        return v


class EchoImageVideoGenerationTool(BaseTool):
    """EchoNet-Synthetic-backed image/video generation tool."""

    name: str = "echo_image_video_generation"
    description: str = (
        "Generate synthetic echocardiography videos using EchoNet-Synthetic. "
        "Pipeline: LIDM latent image sampling -> (optional) Privacy filter -> LVDM video sampling. "
        "Requires EchoNet-Synthetic weights in the repo models directory."
    )
    args_schema: Type[BaseModel] = EchoSynGenerationInput

    def _default_repo_root(self) -> Path:
        # <repo_root>/tools/echo/... -> project root -> tool_repos/EchoNet-Synthetic
        project_root = Path(__file__).resolve().parents[2]
        return project_root / "tool_repos" / "EchoNet-Synthetic"

    def _default_models_dir(self, repo_root: Path) -> Path:
        return repo_root / "models"

    def _python_executable(self) -> str:
        return sys.executable or "python3"

    def _ensure_paths(self, params: EchoSynGenerationInput) -> Dict[str, str]:
        repo_root = Path(params.repo_root) if params.repo_root else self._default_repo_root()
        models_dir = Path(params.models_dir) if params.models_dir else self._default_models_dir(repo_root)

        # Dataset-specific components
        lidm_config_map = {
            "dynamic": "echosyn/lidm/configs/dynamic.yaml",
            "ped_a4c": "echosyn/lidm/configs/ped_a4c.yaml",
            "ped_psax": "echosyn/lidm/configs/ped_psax.yaml",
        }
        reid_model_map = {
            "dynamic": "reidentification_dynamic",
            "ped_a4c": "reidentification_ped_a4c",
            "ped_psax": "reidentification_ped_psax",
        }
        lidm_model_map = {
            "dynamic": "lidm_dynamic",
            "ped_a4c": "lidm_ped_a4c",
            "ped_psax": "lidm_ped_psax",
        }

        lidm_config = repo_root / lidm_config_map[params.dataset]
        lvdm_config = repo_root / "echosyn" / "lvdm" / "configs" / "default.yaml"
        vae_dir = models_dir / "vae"
        lidm_dir = models_dir / lidm_model_map[params.dataset]
        lvdm_dir = models_dir / "lvdm"
        reid_dir = models_dir / reid_model_map[params.dataset]

        missing: List[str] = []
        for p in [repo_root, lidm_config, lvdm_config, vae_dir, lidm_dir, lvdm_dir]:
            if not p.exists():
                missing.append(str(p))
        if params.use_privacy_filter and not reid_dir.exists():
            missing.append(str(reid_dir))

        if missing:
            raise ValueError(
                "EchoNet-Synthetic integration missing required files/directories:\n"
                + "\n".join(f"- {m}" for m in missing)
            )

        return {
            "repo_root": str(repo_root),
            "models_dir": str(models_dir),
            "lidm_config": str(lidm_config),
            "lvdm_config": str(lvdm_config),
            "vae_dir": str(vae_dir),
            "lidm_dir": str(lidm_dir),
            "lvdm_dir": str(lvdm_dir),
            "reid_dir": str(reid_dir),
        }

    def _run_cmd(self, cmd: List[str], cwd: str) -> Dict[str, Any]:
        env = os.environ.copy()
        # Ensure the repo root is importable as a package for echosyn.*
        env["PYTHONPATH"] = cwd + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
        return {"stdout": proc.stdout, "stderr": proc.stderr}

    def _prepare_outdirs(self, base_outdir: Optional[str]) -> Dict[str, Path]:
        stamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        root = Path(base_outdir) if base_outdir else Path("temp") / f"echosyn_run_{stamp}"
        lidm_out = root / "lidm_samples"
        privacy_out = root / "privatised_latents"
        lvdm_out = root / "lvdm_videos"
        for p in [root, lidm_out, lvdm_out]:
            p.mkdir(parents=True, exist_ok=True)
        return {"root": root, "lidm_out": lidm_out, "privacy_out": privacy_out, "lvdm_out": lvdm_out}

    def _frames_multiple_of_32(self, frames: int) -> int:
        if frames % 32 == 0:
            return frames
        return ((frames // 32) + 1) * 32

    def _run(  # type: ignore[override]
        self,
        dataset: str,
        num_samples: int = 4,
        batch_size: int = 8,
        num_steps_img: int = 64,
        num_steps_vid: int = 64,
        frames: int = 64,
        min_lvef: int = 20,
        max_lvef: int = 80,
        save_as: str = "mp4",
        use_privacy_filter: bool = False,
        reference_latents_dir: Optional[str] = None,
        outdir: Optional[str] = None,
        repo_root: Optional[str] = None,
        models_dir: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        params = EchoSynGenerationInput(
            dataset=dataset,
            num_samples=num_samples,
            batch_size=batch_size,
            num_steps_img=num_steps_img,
            num_steps_vid=num_steps_vid,
            frames=frames,
            min_lvef=min_lvef,
            max_lvef=max_lvef,
            save_as=save_as,
            use_privacy_filter=use_privacy_filter,
            reference_latents_dir=reference_latents_dir,
            outdir=outdir,
            repo_root=repo_root,
            models_dir=models_dir,
        )

        paths = self._ensure_paths(params)
        outs = self._prepare_outdirs(params.outdir)
        frames = self._frames_multiple_of_32(params.frames)

        py = self._python_executable()
        repo_root = paths["repo_root"]

        # 1) LIDM: generate latent images (+ latents)
        lidm_cmd: List[str] = [
            py, "echosyn/lidm/sample.py",
            "--config", paths["lidm_config"],
            "--unet", paths["lidm_dir"],
            "--vae", paths["vae_dir"],
            "--output", str(outs["lidm_out"]),
            "--num_samples", str(params.num_samples),
            "--batch_size", str(params.batch_size),
            "--num_steps", str(params.num_steps_img),
            "--save_latent",
        ]
        lidm_logs = self._run_cmd(lidm_cmd, cwd=repo_root)

        # Determine conditioning folder (apply privacy if requested and possible)
        conditioning_dir = outs["lidm_out"] / "latents"
        privacy_logs: Optional[Dict[str, Any]] = None
        if params.use_privacy_filter:
            if not params.reference_latents_dir:
                raise ValueError("use_privacy_filter=True requires reference_latents_dir to be provided.")
            privacy_cmd: List[str] = [
                py, "echosyn/privacy/apply.py",
                "--model", paths["reid_dir"],
                "--synthetic", str(conditioning_dir),
                "--reference", str(Path(params.reference_latents_dir).resolve()),
                "--output", str(outs["privacy_out"]),
            ]
            privacy_logs = self._run_cmd(privacy_cmd, cwd=repo_root)
            conditioning_dir = outs["privacy_out"]

        # 2) LVDM: generate videos
        lvdm_cmd: List[str] = [
            py, "echosyn/lvdm/sample.py",
            "--config", paths["lvdm_config"],
            "--unet", paths["lvdm_dir"],
            "--vae", paths["vae_dir"],
            "--conditioning", str(conditioning_dir),
            "--output", str(outs["lvdm_out"]),
            "--num_samples", str(params.num_samples),
            "--batch_size", str(params.batch_size),
            "--num_steps", str(params.num_steps_vid),
            "--min_lvef", str(params.min_lvef),
            "--max_lvef", str(params.max_lvef),
            "--save_as", params.save_as,
            "--frames", str(frames),
        ]
        lvdm_logs = self._run_cmd(lvdm_cmd, cwd=repo_root)

        # Collect generated files (prefer mp4 if available)
        outputs: Dict[str, Any] = {
            "outdir": str(outs["lvdm_out"].resolve()),
            "artifacts": {},
            "logs": {
                "lidm": lidm_logs,
                "privacy": privacy_logs,
                "lvdm": lvdm_logs,
            },
            "meta": {
                "dataset": params.dataset,
                "num_samples": params.num_samples,
                "frames": frames,
                "save_as": params.save_as,
                "use_privacy_filter": params.use_privacy_filter,
            },
        }

        for ext in params.save_as.split(","):
            ext = ext.strip().lower()
            folder = outs["lvdm_out"] / ext
            if folder.exists():
                outputs["artifacts"][ext] = sorted(str(p) for p in folder.glob(f"*.{ext}"))

        return outputs

    async def _arun(  # pragma: no cover
        self,
        dataset: str,
        num_samples: int = 4,
        batch_size: int = 8,
        num_steps_img: int = 64,
        num_steps_vid: int = 64,
        frames: int = 64,
        min_lvef: int = 20,
        max_lvef: int = 80,
        save_as: str = "mp4",
        use_privacy_filter: bool = False,
        reference_latents_dir: Optional[str] = None,
        outdir: Optional[str] = None,
        repo_root: Optional[str] = None,
        models_dir: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        return self._run(
            dataset=dataset,
            num_samples=num_samples,
            batch_size=batch_size,
            num_steps_img=num_steps_img,
            num_steps_vid=num_steps_vid,
            frames=frames,
            min_lvef=min_lvef,
            max_lvef=max_lvef,
            save_as=save_as,
            use_privacy_filter=use_privacy_filter,
            reference_latents_dir=reference_latents_dir,
            outdir=outdir,
            repo_root=repo_root,
            models_dir=models_dir,
        )


# Quick example (manual):
# tool = EchoImageVideoGenerationTool()
# res = tool.run({
#     "dataset": "dynamic",
#     "num_samples": 2,
#     "frames": 64,
#     "save_as": "mp4",
#     "use_privacy_filter": False
# })
# print(res["outdir"], res["artifacts"].get("mp4"))
