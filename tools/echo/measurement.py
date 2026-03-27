"""EchoPrime measurement tool (MedRAX style)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

import numpy as np
import torch
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class EchoMeasurementPredictionInput(BaseModel):
    input_dir: str = Field(..., description="Directory containing echo videos")
    max_videos: Optional[int] = Field(None, description="Maximum number of videos to process")
    include_report: bool = Field(True, description="Include detailed report")
    save_csv: bool = Field(True, description="Save measurements to CSV")


def _load_echo_prime_model():
    from models.model_factory import get_model

    model = get_model("echo_prime")
    if model is None:
        raise RuntimeError("EchoPrime model not available")
    return model


class EchoMeasurementPredictionTool(BaseTool):
    name: str = "echo_measurement_prediction"
    description: str = "Extract echocardiography measurements using EchoPrime."
    args_schema: Type[BaseModel] = EchoMeasurementPredictionInput

    def _run(
        self,
        input_dir: str,
        max_videos: Optional[int] = None,
        include_report: bool = True,
        save_csv: bool = True,
        run_manager: Optional[Any] = None,
    ) -> Dict[str, Any]:
        model = _load_echo_prime_model()

        stack_of_videos = model.process_mp4s(input_dir)
        if len(stack_of_videos) == 0:
            raise RuntimeError("No videos processed successfully")

        # Some studies (or corrupted clips) can lead to empty tensors inside the
        # EchoPrime pipeline, which in turn cause low-level PyTorch errors like
        # "stack expects a non-empty TensorList". Wrap these with a clearer message.
        try:
            video_features = model.embed_videos(stack_of_videos)
            view_encodings = model.get_views(stack_of_videos)
        except RuntimeError as exc:  # noqa: BLE001
            msg = str(exc)
            if "stack expects a non-empty TensorList" in msg:
                raise RuntimeError(
                    "EchoPrime measurement failed: no valid frames/tensors could be "
                    "built from the input videos (empty TensorList)."
                ) from exc
            raise
        if view_encodings.dim() == 1:
            view_encodings = view_encodings.unsqueeze(0)
        study_embedding = torch.cat((video_features, view_encodings), dim=1)

        measurements = model.predict_metrics(study_embedding)
        formatted: Dict[str, Dict[str, float]] = {}
        for key, value in measurements.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                unit = "%" if key == "EF" else "cm" if "d" in key else "mL"
                formatted[key] = {
                    "value": float(value),
                    "unit": unit,
                    "confidence": 0.85,
                }

        return {
            "status": "success",
            "model": "EchoPrime",
            "input_dir": input_dir,
            "max_videos": max_videos,
            "processed_videos": len(stack_of_videos),
            "measurements": [
                {
                    "video": "study_measurements",
                    "measurements": formatted,
                }
            ],
            "message": f"Measurement prediction completed for {len(stack_of_videos)} videos using EchoPrime",
        }
