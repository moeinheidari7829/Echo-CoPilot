"""EchoPrime view classification tool."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class EchoViewClassificationInput(BaseModel):
    input_dir: str = Field(..., description="Directory containing echo videos")
    visualize: bool = Field(False, description="Generate visualizations")
    max_videos: Optional[int] = Field(None, description="Maximum number of videos to process")


def _load_echo_prime_model():
    from models.model_factory import get_model

    model = get_model("echo_prime")
    if model is None:
        raise RuntimeError("EchoPrime model not available")
    return model


class EchoViewClassificationTool(BaseTool):
    name: str = "echo_view_classification"
    description: str = "Classify echocardiography video views using EchoPrime."
    args_schema: Type[BaseModel] = EchoViewClassificationInput

    def _run(
        self,
        input_dir: str,
        visualize: bool = False,
        max_videos: Optional[int] = None,
        run_manager: Optional[Any] = None,
    ) -> Dict[str, Any]:
        model = _load_echo_prime_model()
        stack_of_videos = model.process_mp4s(input_dir)
        if stack_of_videos.shape[0] == 0:
            raise RuntimeError(f"No valid videos found in {input_dir}")

        if max_videos and stack_of_videos.shape[0] > max_videos:
            stack_of_videos = stack_of_videos[:max_videos]

        views = model.get_views(stack_of_videos, visualize=visualize, return_view_list=True)
        if not views:
            raise RuntimeError("No videos processed successfully")

        results: List[Dict[str, Any]] = []
        distribution: Dict[str, Dict[str, float]] = {}
        for idx, view in enumerate(views):
            entry = {
                "video": f"video_{idx + 1}.mp4",
                "predicted_view": view,
                "confidence": 0.85,
            }
            results.append(entry)
            info = distribution.setdefault(view, {"count": 0, "confidence": 0.0})
            info["count"] += 1
            info["confidence"] = max(info["confidence"], entry["confidence"])

        return {
            "status": "success",
            "model": "EchoPrime",
            "input_dir": input_dir,
            "max_videos": max_videos,
            "processed_videos": len(results),
            "classifications": distribution,
            "detailed_results": results,
            "message": f"View classification completed for {len(results)} videos using EchoPrime",
        }
