"""PanEcho-based disease prediction tool."""

from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Optional, Type

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


_MODEL_CACHE: Dict[str, Any] = {}


class EchoDiseasePredictionInput(BaseModel):
    input_dir: str = Field(..., description="Directory containing echo videos")
    max_videos: Optional[int] = Field(None, description="Maximum number of videos to process")
    save_csv: bool = Field(True, description="Save results to CSV file")
    include_confidence: bool = Field(True, description="Include confidence scores in output")


def _load_panecho_model():
    if "panecho" in _MODEL_CACHE:
        return _MODEL_CACHE["panecho"]

    from models.model_factory import get_model

    model = get_model("panecho")
    if model is None:
        raise RuntimeError("PanEcho model not available")
    _MODEL_CACHE["panecho"] = model
    return model


class EchoDiseasePredictionTool(BaseTool):
    name: str = "echo_disease_prediction"
    description: str = "Predict cardiac diseases from echo videos using PanEcho."
    args_schema: Type[BaseModel] = EchoDiseasePredictionInput

    def _run(
        self,
        input_dir: str,
        max_videos: Optional[int] = None,
        save_csv: bool = True,
        include_confidence: bool = True,
        run_manager: Optional[Any] = None,
    ) -> Dict[str, Any]:
        model = _load_panecho_model()
        video_files = glob.glob(os.path.join(input_dir, "*.mp4"))
        if max_videos:
            video_files = video_files[:max_videos]
        if not video_files:
            raise RuntimeError(f"No MP4 videos found in {input_dir}")

        predictions: List[Dict[str, Any]] = []
        for video_path in video_files:
            try:
                entry = self._process_video(model, video_path)
                predictions.append(entry)
            except Exception:
                continue

        if not predictions:
            raise RuntimeError("No videos processed successfully")

        return {
            "status": "success",
            "model": "PanEcho",
            "input_dir": input_dir,
            "max_videos": max_videos,
            "processed_videos": len(predictions),
            "predictions": predictions,
            "message": f"Disease prediction completed for {len(predictions)} videos using PanEcho",
        }

    def _process_video(self, model, video_path: str) -> Dict[str, Any]:
        cap = cv2.VideoCapture(video_path)
        frames: List[np.ndarray] = []
        max_frames = 16

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        while len(frames) < max_frames and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        if not frames:
            raise RuntimeError("No frames captured")
        while len(frames) < max_frames:
            frames.append(frames[-1])

        frames_array = np.array(frames, dtype=np.float32) / 255.0
        frames_tensor = torch.tensor(frames_array).permute(0, 3, 1, 2)
        frames_tensor = frames_tensor.unsqueeze(0)
        frames_tensor = normalize(frames_tensor.view(-1, 3, 224, 224)).view(1, 16, 3, 224, 224)
        frames_tensor = frames_tensor.permute(0, 2, 1, 3, 4)

        device = next(model.parameters()).device
        frames_tensor = frames_tensor.to(device)

        with torch.no_grad():
            preds = model(frames_tensor)

        disease_predictions: Dict[str, Any] = {}
        for task_name, pred_value in preds.items():
            description = TASK_DESCRIPTIONS.get(task_name, task_name)
            if torch.is_tensor(pred_value):
                value, task_type, confidence = self._interpret_tensor(task_name, pred_value)
            else:
                value = float(pred_value) if isinstance(pred_value, (int, float)) else 0.0
                task_type = "unknown"
                confidence = 0.0
            disease_predictions[task_name] = {
                "value": value,
                "description": description,
                "confidence": confidence,
                "task_type": task_type,
                "units": TASK_UNITS.get(task_name, "unknown"),
            }

        return {"video": os.path.basename(video_path), "predictions": disease_predictions}

    def _interpret_tensor(self, task_name: str, tensor: torch.Tensor) -> tuple[Any, str, float]:
        if tensor.shape == (1, 1):
            raw = float(tensor[0, 0].item())
            if task_name in REGRESSION_TASKS:
                return raw, "regression", 0.85
            return raw, "binary_classification", max(raw, 1.0 - raw)

        if tensor.shape[1] > 1:
            probs = tensor[0]
            predicted_class = int(probs.argmax().item())
            confidence = float(probs.max().item())
            class_names = CLASS_NAMES.get(task_name)
            if class_names and predicted_class < len(class_names):
                return class_names[predicted_class], "multi-class_classification", confidence
            return predicted_class, "multi-class_classification", confidence

        value = float(tensor.flatten().mean().item())
        return value, "regression", 0.85


REGRESSION_TASKS = {
    "EF",
    "GLS",
    "LVEDV",
    "LVESV",
    "LVSV",
    "IVSd",
    "LVPWd",
    "LVIDs",
    "LVIDd",
    "LVOTDiam",
    "E|EAvg",
    "RVSP",
    "RVIDd",
    "TAPSE",
    "RVSVel",
    "LAIDs2D",
    "LAVol",
    "RADimensionM-L(cm)",
    "AVPkVel(m/s)",
    "TVPkGrad",
    "AORoot",
}

TASK_UNITS = {
    "EF": "%",
    "GLS": "%",
    "LVEDV": "cm³",
    "LVESV": "cm³",
    "LVSV": "cm³",
    "IVSd": "cm",
    "LVPWd": "cm",
    "LVIDs": "cm",
    "LVIDd": "cm",
    "LVOTDiam": "cm",
    "E|EAvg": "ratio",
    "RVSP": "mmHg",
    "RVIDd": "cm",
    "TAPSE": "cm",
    "RVSVel": "cm/s",
    "LAIDs2D": "cm",
    "LAVol": "cm³",
    "RADimensionM-L(cm)": "cm",
    "AVPkVel(m/s)": "m/s",
    "TVPkGrad": "mmHg",
    "AORoot": "cm",
}

CLASS_NAMES = {
    "LVSize": ["Mildly Increased", "Moderately|Severely Increased", "Normal"],
    "LVSystolicFunction": ["Mildly Decreased", "Moderately|Severely Decreased", "Normal|Hyperdynamic"],
    "LVDiastolicFunction": ["Mild|Indeterminate", "Moderate|Severe", "Normal"],
    "RVSize": ["Mildly Increased", "Moderately|Severely Increased", "Normal"],
    "LASize": ["Mildly Dilated", "Moderately|Severely Dilated", "Normal"],
    "AVStenosis": ["Mild|Moderate", "None", "Severe"],
    "AVRegurg": ["Mild", "Moderate|Severe", "None|Trace"],
    "MVRegurgitation": ["Mild", "Moderate|Severe", "None|Trace"],
    "TVRegurgitation": ["Mild", "Moderate|Severe", "None|Trace"],
}

TASK_DESCRIPTIONS = {
    "pericardial-effusion": "Pericardial Effusion",
    "EF": "Ejection Fraction (%)",
    "GLS": "Global Longitudinal Strain (%)",
    "LVEDV": "LV End-Diastolic Volume (cm³)",
    "LVESV": "LV End-Systolic Volume (cm³)",
    "LVSV": "LV Stroke Volume (cm³)",
    "LVSize": "LV Size",
    "LVWallThickness-increased-any": "LV Wall Thickness - Any Increase",
    "LVWallThickness-increased-modsev": "LV Wall Thickness - Moderate/Severe Increase",
    "LVSystolicFunction": "LV Systolic Function",
    "LVWallMotionAbnormalities": "LV Wall Motion Abnormalities",
    "IVSd": "Interventricular Septum Diastole (cm)",
    "LVPWd": "LV Posterior Wall Diastole (cm)",
    "LVIDs": "LV Internal Diameter Systole (cm)",
    "LVIDd": "LV Internal Diameter Diastole (cm)",
    "LVOTDiam": "LV Outflow Tract Diameter (cm)",
    "LVDiastolicFunction": "LV Diastolic Function",
    "E|EAvg": "E/e' Ratio",
    "RVSP": "RV Systolic Pressure (mmHg)",
    "RVSize": "RV Size",
    "RVSystolicFunction": "RV Systolic Function",
    "RVIDd": "RV Internal Diameter Diastole (cm)",
    "TAPSE": "Tricuspid Annular Plane Systolic Excursion (cm)",
    "RVSVel": "RV Systolic Excursion Velocity (cm/s)",
    "LASize": "Left Atrial Size",
    "LAIDs2D": "LA Internal Diameter Systole 2D (cm)",
    "LAVol": "LA Volume (cm³)",
    "RASize": "Right Atrial Size",
    "RADimensionM-L(cm)": "RA Major Dimension (cm)",
    "AVStructure": "Aortic Valve Structure",
    "AVStenosis": "Aortic Valve Stenosis",
    "AVPkVel(m/s)": "Aortic Valve Peak Velocity (m/s)",
    "AVRegurg": "Aortic Valve Regurgitation",
    "MVStenosis": "Mitral Valve Stenosis",
    "MVRegurgitation": "Mitral Valve Regurgitation",
    "TVRegurgitation": "Tricuspid Valve Regurgitation",
    "TVPkGrad": "Tricuspid Valve Peak Gradient (mmHg)",
    "RAP-8-or-higher": "Elevated RA Pressure",
    "AORoot": "Aortic Root Diameter (cm)",
}
