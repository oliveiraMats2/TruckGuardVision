#!/bin/bash

# get sudo permission
sudo echo "sudo permission acquired"

# Define variables
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <weights_path> [workspace_ram]"
  echo "  weights_path: Path to the PyTorch weights file"
  echo "  workspace_ram: Amount of workspace memory for TensorRT (default: 4GB)"
  exit 1
fi

WEIGHTS_PATH="$1"
WORKSPACE_RAM="${2:-4GB}"
ENGINE_NAME="yolov7-fp16-1x8x8.engine"
TIMING_CACHE="timing.cache"
WEIGHTS_DIR=$(dirname "${WEIGHTS_PATH}")

# add ./ to the beginning of the path if it is relative
if [[ ! "${WEIGHTS_PATH}" = /* ]]; then
  WEIGHTS_PATH="./${WEIGHTS_PATH}"
  WEIGHTS_DIR=$(dirname "${WEIGHTS_PATH}")
fi

# Check if weights file exists
if [ ! -f "${WEIGHTS_PATH}" ]; then
  echo "Weights file not found. Please ensure the path is correct."
  exit 1
fi
echo "Weights file found at ${WEIGHTS_PATH}"

echo "Reparameterizing model..."
python -m Src.app.yolo_deploy.reparameterization ${WEIGHTS_PATH}

WEIGHTS_PATH="${WEIGHTS_PATH%.*}_reparam.pt"
if [ ! -f "${WEIGHTS_PATH}" ]; then
  echo "Reparameterization failed. Exiting."
  exit 1
fi

# Step 1: Export PyTorch model to ONNX with grid, EfficientNMS plugin and dynamic batch size
echo "Exporting PyTorch model to ONNX..."
python -m Src.app.yolo_deploy.export \
  --weights ${WEIGHTS_PATH} \
  --grid \
  --end2end \
  --dynamic-batch \
  --simplify \
  --topk-all 100 \
  --iou-thres 0.65 \
  --conf-thres 0.35 \
  --img-size 640 640

WEIGHTS_PATH="${WEIGHTS_PATH%.*}.onnx"

# ONNX_NAME is the name of the ONNX file, for example yolov7.onnx
ONNX_NAME="${WEIGHTS_PATH##*/}"

# Check if ONNX export was successful
if [ ! -f "${WEIGHTS_PATH}" ]; then
  echo "Failed to export ONNX model. Exiting."
  exit 1
fi

echo "ONNX model exported successfully to ${WEIGHTS_PATH}"

# Step 2: Convert ONNX to TensorRT engine using Docker with volume mounts
echo "Converting ONNX to TensorRT engine..."

WORKSPACE_DIR="/workspace/weights"
sudo docker run -it --rm --gpus=all \
  -v "${WEIGHTS_DIR}:${WORKSPACE_DIR}" \
  nvcr.io/nvidia/tensorrt:22.06-py3 \
  /bin/bash -c "cd /workspace && \
  ./tensorrt/bin/trtexec \
    --onnx=${WORKSPACE_DIR}/${ONNX_NAME} \
    --minShapes=images:1x3x640x640 \
    --optShapes=images:8x3x640x640 \
    --maxShapes=images:8x3x640x640 \
    --fp16 \
    --workspace=${WORKSPACE_RAM} \
    --saveEngine=${WORKSPACE_DIR}/${ENGINE_NAME} \
    --timingCacheFile=${WORKSPACE_DIR}/${TIMING_CACHE} && \
  echo 'TensorRT engine created successfully' && \
  ./tensorrt/bin/trtexec --loadEngine=${WORKSPACE_DIR}/${ENGINE_NAME}"

ENGINE_PATH="${WEIGHTS_DIR}/${ENGINE_NAME}"

echo "Copying TensorRT engine back to ${ENGINE_PATH}..."

# Check if engine file was successfully copied back. 
if [ -f "${ENGINE_PATH}" ]; then
  echo "Conversion complete. TensorRT engine saved as ${ENGINE_PATH}"
else
  echo "Failed to create or copy TensorRT engine."
  exit 1
fi

# copy the engine file to Src/app/yolo_deploy/deploy/triton-deploy/models/yolov7/1/ and rename it to model.plan
MODEL_DIR="Src/app/yolo_deploy/triton-deploy/models/yolov7/1"
mkdir -p ${MODEL_DIR}
mv ${ENGINE_PATH} ${MODEL_DIR}/model.plan


echo "TensorRT engine saved as ${MODEL_DIR}/model.plan"