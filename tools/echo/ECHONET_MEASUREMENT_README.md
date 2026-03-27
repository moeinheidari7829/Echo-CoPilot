# EchoNet Measurement Tool

Automatic annotation tool for echocardiography measurements using EchoNet models.

## Overview

This tool provides automatic measurements for:
- **2D Linear Measurements**: IVS, LVID, LVPW, Aorta, Aortic Root, LA, RV Base, PA, IVC
- **Doppler Velocity Measurements**: TRVMAX, AVVMAX, MRVMAX, LVOTVMAX, Lateral E', Medial E'

## Usage

The tool is integrated into the EchoPilot agent and can be called like any other tool:

```python
from tools import EchoNetMeasurementTool

tool = EchoNetMeasurementTool()

# For 2D measurements
result = tool._run(
    file_path="/path/to/video.avi",  # or .dcm
    measurement_type="lvid"
)

# For Doppler measurements (requires DICOM)
result = tool._run(
    file_path="/path/to/doppler.dcm",
    measurement_type="trvmax"
)
```

## Supported Measurement Types

### 2D Linear Measurements
- `ivs` - Interventricular Septum
- `lvid` - Left Ventricular Internal Diameter
- `lvpw` - Left Ventricular Posterior Wall
- `aorta` - Aorta
- `aortic_root` - Aortic Root
- `la` - Left Atrium
- `rv_base` - Right Ventricle Base
- `pa` - Pulmonary Artery
- `ivc` - Inferior Vena Cava

### Doppler Velocity Measurements
- `trvmax` - Tricuspid Regurgitation Maximum Velocity
- `avvmax` - Aortic Valve Maximum Velocity
- `mrvmax` - Mitral Regurgitation Maximum Velocity
- `lvotvmax` - Left Ventricular Outflow Tract Maximum Velocity
- `latevel` - Lateral E' Velocity
- `medevel` - Medial (Septal) E' Velocity

## Input Requirements

- **2D Measurements**: AVI or DICOM format (480x640 resolution)
- **Doppler Measurements**: DICOM format only (requires DICOM tags for velocity calculation)

## Output

Returns a dictionary with:
- `status`: "success" or "error"
- `model`: "EchoNet Measurements"
- `category`: "2D Linear Measurement" or "Doppler Velocity Measurement"
- `results`: Measurement results including coordinates, distances, or velocities

## Model Weights

Model weights are automatically downloaded from the EchoNet GitHub releases when first cloning the repository. If weights are missing, run:

```bash
cd tool_repos/echonet-measurements
./download_weights.sh
```

## Citation

If using this tool, please cite:

Sahashi Y, Ieki H, Yuan V, Christensen M, Vukadinovic M, Binder-Rodriguez C, Rhee J, Zou JY, He B, Cheng P, Ouyang D. Artificial Intelligence Automation of Echocardiographic Measurements. J Am Coll Cardiol. 2025 Sep 30;86(13):964-978. doi: 10.1016/j.jacc.2025.07.053.

## Repository

Original repository: https://github.com/echonet/measurements

