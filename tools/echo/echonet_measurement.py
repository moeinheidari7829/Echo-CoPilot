"""EchoNet measurement tool for automatic annotation of echocardiography."""

from __future__ import annotations

import sys
import os
import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional, Type

import torch
import numpy as np
import cv2
import pydicom
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

# Add echonet-measurements to path
ECHONET_PATH = Path(__file__).parent.parent.parent / "tool_repos" / "echonet-measurements"
if str(ECHONET_PATH) not in sys.path:
    sys.path.insert(0, str(ECHONET_PATH))

# Import from echonet-measurements utils (avoid conflict with project utils)
# Need to change directory temporarily for utils.py to find its .npy file
import importlib.util
original_cwd = os.getcwd()
try:
    os.chdir(str(ECHONET_PATH))
    echonet_utils_path = ECHONET_PATH / "utils.py"
    spec = importlib.util.spec_from_file_location("echonet_utils", echonet_utils_path)
    echonet_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(echonet_utils)

    segmentation_to_coordinates = echonet_utils.segmentation_to_coordinates
    get_coordinates_from_dicom = echonet_utils.get_coordinates_from_dicom
    ybr_to_rgb = echonet_utils.ybr_to_rgb
finally:
    os.chdir(original_cwd)


class EchoNetMeasurementInput(BaseModel):
    """Input for EchoNet measurement tool."""

    file_path: str = Field(..., description="Path to echo video file (.mp4, .avi, or .dcm)")
    measurement_type: str = Field(
        ...,
        description="Type of measurement to perform. For 2D: ivs, lvid, lvpw, aorta, aortic_root, la, rv_base, pa, ivc. For Doppler: avvmax, trvmax, mrvmax, lvotvmax, latevel, medevel"
    )
    output_dir: Optional[str] = Field(None, description="Directory to save output files (optional)")


def _load_2d_model(measurement_type: str, device: str):
    """Load 2D measurement model."""
    from torchvision.models.segmentation import deeplabv3_resnet50

    weights_path = ECHONET_PATH / "weights" / "2D_models" / f"{measurement_type}_weights.ckpt"
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    weights = torch.load(weights_path, map_location=device)
    backbone = deeplabv3_resnet50(num_classes=2)
    weights = {k.replace("m.", ""): v for k, v in weights.items()}
    backbone.load_state_dict(weights)
    backbone = backbone.to(device)
    backbone.eval()
    return backbone


def _load_doppler_model(measurement_type: str, device: str):
    """Load Doppler measurement model."""
    from torchvision.models.segmentation import deeplabv3_resnet50

    weights_path = ECHONET_PATH / "weights" / "Doppler_models" / f"{measurement_type}_weights.ckpt"
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    weights = torch.load(weights_path, map_location=device)
    backbone = deeplabv3_resnet50(num_classes=1)
    weights = {k.replace("m.", ""): v for k, v in weights.items()}
    backbone.load_state_dict(weights)
    backbone = backbone.to(device)
    backbone.eval()
    return backbone


def _process_2d_measurement(file_path: str, measurement_type: str, device: str) -> Dict[str, Any]:
    """Process 2D linear measurement."""
    backbone = _load_2d_model(measurement_type, device)

    # Load video
    frames = []
    if file_path.endswith((".avi", ".mp4", ".mov")):
        video = cv2.VideoCapture(file_path)
        while True:
            ret, frame = video.read()
            if not ret:
                break
            frames.append(frame)
        video.release()
        input_type = "video"
    elif file_path.endswith(".dcm"):
        ds = pydicom.dcmread(file_path)
        input_dicom = ds.pixel_array
        for frame in input_dicom:
            if ds.PhotometricInterpretation == "YBR_FULL_422":
                frame = ybr_to_rgb(frame)
            resized_frame = cv2.resize(frame, (640, 480))
            frames.append(resized_frame)
        input_type = "dcm"
    else:
        raise ValueError(f"File must be video (.avi, .mp4, .mov) or DICOM (.dcm), got: {file_path}")

    frames = np.array(frames)
    input_tensor = torch.tensor(frames).float() / 255.0
    input_tensor = input_tensor.to(device).permute(0, 3, 1, 2)

    # Run inference
    predictions = []
    with torch.no_grad():
        for i in range(input_tensor.shape[0]):
            batch = input_tensor[i].unsqueeze(0)
            logits = backbone(batch)["out"]
            logits = torch.sigmoid(logits)
            pred = segmentation_to_coordinates(logits, normalize=False, order="XY")
            predictions.append(pred)

    predictions = torch.cat(predictions, dim=0).cpu().numpy()

    # Calculate measurements
    avg_point1 = predictions[:, 0, :].mean(axis=0)
    avg_point2 = predictions[:, 1, :].mean(axis=0)
    pixel_distance = np.linalg.norm(avg_point2 - avg_point1)

    result = {
        "measurement_type": measurement_type,
        "num_frames": len(frames),
        "point1": avg_point1.tolist(),
        "point2": avg_point2.tolist(),
        "pixel_distance": float(pixel_distance),
    }

    # Get physical distance if DICOM (video files don't have calibration metadata)
    if input_type == "dcm":
        ds = pydicom.dcmread(file_path)
        doppler_region = get_coordinates_from_dicom(ds)[0]
        REGION_PHYSICAL_DELTA_X_SUBTAG = (0x0018, 0x602C)
        REGION_PHYSICAL_DELTA_Y_SUBTAG = (0x0018, 0x602E)

        if REGION_PHYSICAL_DELTA_X_SUBTAG in doppler_region:
            conversion_factor_X = abs(doppler_region[REGION_PHYSICAL_DELTA_X_SUBTAG].value)
            height, width = input_dicom.shape[1], input_dicom.shape[2]
            ratio_height = height / 480
            physical_distance = pixel_distance * ratio_height * conversion_factor_X * 10
            result["physical_distance_mm"] = float(physical_distance)
            result["physical_distance_cm"] = float(physical_distance / 10)

    return result


def _process_doppler_measurement(file_path: str, measurement_type: str, device: str) -> Dict[str, Any]:
    """Process Doppler velocity measurement."""
    if not file_path.endswith(".dcm"):
        raise ValueError("Doppler measurement requires DICOM file")

    backbone = _load_doppler_model(measurement_type, device)

    # Load DICOM
    ds = pydicom.dcmread(file_path)
    input_image = ds.pixel_array

    if ds.PhotometricInterpretation == 'MONOCHROME2':
        input_image = np.stack((input_image,) * 3, axis=-1)
    elif ds.PhotometricInterpretation == "YBR_FULL_422" and len(input_image.shape) == 3:
        from pydicom.pixel_data_handlers.util import convert_color_space
        input_image = convert_color_space(arr=input_image, current="YBR_FULL_422", desired="RGB")
        ecg_mask = np.logical_and(input_image[:, :, 1] > 200, input_image[:, :, 0] < 100)
        input_image[ecg_mask, :] = 0

    # Get Doppler region
    doppler_region = get_coordinates_from_dicom(ds)[0]
    REGION_PHYSICAL_DELTA_Y_SUBTAG = (0x0018, 0x602E)
    REGION_Y0_SUBTAG = (0x0018, 0x601A)
    REFERENCE_LINE_TAG = (0x0018, 0x6022)

    conversion_factor = abs(doppler_region[REGION_PHYSICAL_DELTA_Y_SUBTAG].value)
    y0 = doppler_region[REGION_Y0_SUBTAG].value
    horizontal_y = doppler_region[REFERENCE_LINE_TAG].value

    # Process Doppler area
    input_dicom_doppler_area = input_image[342:, :, :]
    doppler_area_tensor = torch.tensor(input_dicom_doppler_area)
    doppler_area_tensor = doppler_area_tensor.permute(2, 0, 1).unsqueeze(0).float() / 255.0
    doppler_area_tensor = doppler_area_tensor.to(device)

    # Run inference
    with torch.no_grad():
        logit = backbone(doppler_area_tensor)["out"]
        logits_normalized = torch.sigmoid(logit)
        logits_normalized = (logits_normalized - logits_normalized.min()) / (logits_normalized.max() - logits_normalized.min())
        logits_normalized = logits_normalized.squeeze().cpu().numpy()
        max_coords = np.unravel_index(np.argmax(logits_normalized), logits_normalized.shape)

        predicted_x = int(max_coords[1])
        predicted_y = int(max_coords[0] + y0)
        peak_velocity = conversion_factor * (predicted_y - (y0 + horizontal_y))
        peak_velocity = round(peak_velocity, 2)

    return {
        "measurement_type": measurement_type,
        "peak_velocity_cm_s": float(peak_velocity),
        "predicted_point": [predicted_x, predicted_y],
    }


class EchoNetMeasurementTool(BaseTool):
    """Tool for automatic annotation of echocardiography measurements using EchoNet."""

    name: str = "echonet_measurement"
    description: str = """
    Perform automatic measurements on echocardiography videos/images.

    Supports 2D linear measurements (IVS, LVID, LVPW, Aorta, LA, RV, PA, IVC)
    and Doppler velocity measurements (TRVMAX, AVVMAX, MRVMAX, LVOTVMAX, etc.).

    Input can be video (.mp4, .avi, .mov) or DICOM (.dcm) format.
    DICOM is required for physical measurements (mm/cm) and Doppler velocity measurements.
    Video formats will return pixel distances only.
    """
    args_schema: Type[BaseModel] = EchoNetMeasurementInput

    def _run(
        self,
        file_path: str,
        measurement_type: str,
        output_dir: Optional[str] = None,
        run_manager: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Execute the measurement."""
        # Validate inputs
        measurement_type = measurement_type.lower()

        TWO_D_MEASUREMENTS = ["ivs", "lvid", "lvpw", "aorta", "aortic_root", "la", "rv_base", "pa", "ivc"]
        DOPPLER_MEASUREMENTS = ["avvmax", "trvmax", "mrvmax", "lvotvmax", "latevel", "medevel"]

        if measurement_type not in TWO_D_MEASUREMENTS + DOPPLER_MEASUREMENTS:
            raise ValueError(f"Unknown measurement type: {measurement_type}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            if measurement_type in TWO_D_MEASUREMENTS:
                result = _process_2d_measurement(file_path, measurement_type, device)
                category = "2D Linear Measurement"
            else:
                result = _process_doppler_measurement(file_path, measurement_type, device)
                category = "Doppler Velocity Measurement"

            return {
                "status": "success",
                "model": "EchoNet Measurements",
                "category": category,
                "file_path": file_path,
                "results": result,
                "message": f"Successfully measured {measurement_type.upper()}",
            }

        except Exception as e:
            return {
                "status": "error",
                "model": "EchoNet Measurements",
                "file_path": file_path,
                "measurement_type": measurement_type,
                "error": str(e),
                "message": f"Failed to measure {measurement_type.upper()}: {str(e)}",
            }

