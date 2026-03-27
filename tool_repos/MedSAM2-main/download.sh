#!/usr/bin/env bash
# Script to download MedSAM2 model checkpoints
# Create checkpoints directory if it doesn't exist
mkdir -p checkpoints
# Use either wget or curl to download the checkpoints
if command -v wget > /dev/null 2>&1; then
    CMD="wget -P checkpoints"
elif command -v curl > /dev/null 2>&1; then
    CMD="curl -L -o"
    CURL=1
else
    echo "Please install wget or curl to download the checkpoints."
    exit 1
fi
# Define the base URL for MedSAM2 models on Hugging Face
HF_BASE_URL="https://huggingface.co/wanglab/MedSAM2/resolve/main"
# Define the model checkpoint files (as separate variables instead of an array)
MODEL1="MedSAM2_2411.pt"
MODEL2="MedSAM2_US_Heart.pt"
MODEL3="MedSAM2_MRI_LiverLesion.pt"
MODEL4="MedSAM2_CTLesion.pt"
MODEL5="MedSAM2_latest.pt"

# Download each checkpoint
for model in $MODEL1 $MODEL2 $MODEL3 $MODEL4 $MODEL5; do
    echo "Downloading ${model}..."
    model_url="${HF_BASE_URL}/${model}"
    
    if [ -n "$CURL" ]; then
        $CMD "checkpoints/${model}" "$model_url" || { echo "Failed to download checkpoint from $model_url"; exit 1; }
    else
        $CMD "$model_url" || { echo "Failed to download checkpoint from $model_url"; exit 1; }
    fi
done
echo "All MedSAM2 model checkpoints have been downloaded successfully to the 'checkpoints' directory."

# Download the Efficient Track Anything checkpoint
ETA_BASE_URL="https://huggingface.co/yunyangx/efficient-track-anything/resolve/main"
ETA_MODELS=("efficienttam_s_512x512.pt" "efficienttam_ti_512x512.pt")

for ETA_MODEL in "${ETA_MODELS[@]}"; do
    echo "Downloading ${ETA_MODEL}..."
    eta_model_url="${ETA_BASE_URL}/${ETA_MODEL}"

    if [ -n "$CURL" ]; then
        $CMD "checkpoints/${ETA_MODEL}" "$eta_model_url" || { echo "Failed to download checkpoint from $eta_model_url"; exit 1; }
    else
        $CMD "$eta_model_url" || { echo "Failed to download checkpoint from $eta_model_url"; exit 1; }
    fi
done
echo "Efficient Track Anything checkpoints have been downloaded successfully to the 'checkpoints' directory."

# download SAM2 model
SAM2_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"
SAM2_MODEL="sam2.1_hiera_tiny.pt"
echo "Downloading ${SAM2_MODEL}..."
sam2_model_url="${SAM2_BASE_URL}/${SAM2_MODEL}"
if [ -n "$CURL" ]; then
    $CMD "checkpoints/${SAM2_MODEL}" "$sam2_model_url" || { echo "Failed to download checkpoint from $sam2_model_url"; exit 1; }
else
    $CMD "$sam2_model_url" || { echo "Failed to download checkpoint from $sam2_model_url"; exit 1; }
fi
echo "SAM2 model checkpoint has been downloaded successfully to the 'checkpoints' directory."


