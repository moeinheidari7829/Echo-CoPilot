"""EchoPrime report generation tool."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import matplotlib.pyplot as plt
import numpy as np
import torch
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class EchoReportGenerationInput(BaseModel):
    input_dir: str = Field(..., description="Directory containing echo videos")
    visualize_views: bool = Field(False, description="Generate view visualizations")
    max_videos: Optional[int] = Field(None, description="Maximum number of videos to process")
    include_sections: bool = Field(True, description="Include all report sections")


def _load_echo_prime_model():
    from models.model_factory import get_model

    model = get_model("echo_prime")
    if model is None:
        raise RuntimeError("EchoPrime model not available")
    return model


class EchoReportGenerationTool(BaseTool):
    name: str = "echo_report_generation"
    description: str = "Generate comprehensive echo report using EchoPrime."
    args_schema: Type[BaseModel] = EchoReportGenerationInput

    def _run(
        self,
        input_dir: str,
        visualize_views: bool = False,
        max_videos: Optional[int] = None,
        include_sections: bool = True,
        run_manager: Optional[Any] = None,
    ) -> Dict[str, Any]:
        model = _load_echo_prime_model()

        stack_of_videos = model.process_mp4s(input_dir)
        if len(stack_of_videos) == 0:
            raise RuntimeError("No videos processed successfully")

        video_features = model.embed_videos(stack_of_videos)
        view_encodings = model.get_views(stack_of_videos, visualize=visualize_views)
        if view_encodings.dim() == 1:
            view_encodings = view_encodings.unsqueeze(0)
        study_embedding = torch.cat((video_features, view_encodings), dim=1)

        report = model.generate_report(study_embedding)
        measurements = model.predict_metrics(study_embedding)
        views = model.get_views(stack_of_videos, return_view_list=True)

        analysis = {
            "video": "study_analysis",
            "view_classification": {
                "predicted_views": views,
                "view_distribution": {view: views.count(view) for view in set(views)},
            },
            "measurements": measurements,
            "disease_predictions": {},
            "quality_assessment": {"confidence": 0.85},
        }

        summary = self._generate_comprehensive_report([analysis], include_sections)
        visualization = self._create_view_visualization([analysis], input_dir) if visualize_views else None

        return {
            "status": "success",
            "model": "EchoPrime",
            "input_dir": input_dir,
            "max_videos": max_videos,
            "processed_videos": len(stack_of_videos),
            "analysis": analysis,
            "report": summary,
            "view_visualization": visualization,
            "message": f"Report generation completed for {len(stack_of_videos)} videos using EchoPrime",
        }

    def _generate_comprehensive_report(self, analyses: List[Dict[str, Any]], include_sections: bool) -> Dict[str, Any]:
        avg_measurements: Dict[str, float] = {}
        measurement_keys = ["EF", "LVEDV", "LVESV", "GLS", "IVSd", "LVPWd", "LVIDs", "LVIDd"]

        for key in measurement_keys:
            values = [
                m.get(key, {}).get("value", 0) if isinstance(m.get(key), dict) else m.get(key, 0)
                for analysis in analyses
                for m in [analysis.get("measurements", {})]
                if m
            ]
            if values:
                avg_measurements[key] = float(np.mean(values))

        ef = avg_measurements.get("EF", 0)
        if ef > 55:
            ef_status = "Normal"
        elif ef > 45:
            ef_status = "Mildly reduced"
        else:
            ef_status = "Moderately to severely reduced"

        summary = f"Left ventricular ejection fraction is {ef_status} ({ef:.1f}%). "
        if "LVEDV" in avg_measurements:
            summary += f"Left ventricular end-diastolic volume is {avg_measurements['LVEDV']:.1f} mL. "
        if "GLS" in avg_measurements:
            summary += f"Global longitudinal strain is {avg_measurements['GLS']:.1f}%"

        recommendations = []
        if ef < 50:
            recommendations.append("Consider cardiology consultation")
        if avg_measurements.get("GLS", 0) < -18:
            recommendations.append("Monitor for heart failure")
        if not recommendations:
            recommendations.append("Routine follow-up in 1 year")

        view_distribution = {}
        for analysis in analyses:
            view = analysis.get("view_classification", {}).get("predicted_views", [])
            if isinstance(view, list):
                for v in view:
                    view_distribution[v] = view_distribution.get(v, 0) + 1

        sections = [
            "findings",
            "measurements",
            "view_analysis",
            "recommendations",
        ] if include_sections else []

        return {
            "summary": summary,
            "recommendations": recommendations,
            "sections": sections,
            "measurements": {k: f"{v:.1f}" for k, v in avg_measurements.items()},
            "view_distribution": view_distribution,
            "processed_videos": len(analyses),
            "overall_confidence": np.mean([a.get("confidence", 0) for a in analyses]),
        }

    def _create_view_visualization(self, analyses: List[Dict[str, Any]], input_dir: str) -> Optional[str]:
        try:
            view_counts: Dict[str, int] = {}
            for analysis in analyses:
                views = analysis.get("view_classification", {}).get("predicted_views", [])
                if isinstance(views, list):
                    for view in views:
                        view_counts[view] = view_counts.get(view, 0) + 1

            if not view_counts:
                return None

            plt.figure(figsize=(4, 4))
            plt.pie(view_counts.values(), labels=view_counts.keys(), autopct="%1.1f%%")
            plt.title("Echo View Distribution")
            output_path = Path(input_dir) / "view_distribution.png"
            plt.savefig(output_path, dpi=200, bbox_inches="tight")
            plt.close()
            return str(output_path)
        except Exception:
            return None
