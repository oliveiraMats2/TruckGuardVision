from fastapi import FastAPI, HTTPException

from PIL import Image, ImageOps
from pydantic import BaseModel
from typing import List, Dict, Any
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException
import base64
import cv2
import numpy as np
import enum
from io import BytesIO
import os

# Get configuration from environment variables
TRITON_URL = os.environ.get("TRITON_URL", "localhost:8001")
MODEL_WIDTH = int(os.environ.get("MODEL_WIDTH", "640"))
MODEL_HEIGHT = int(os.environ.get("MODEL_HEIGHT", "640"))
MODEL_NAME = os.environ.get("MODEL_NAME", "yolov7")

print(f"Using Triton server at: {TRITON_URL}")
print(f"Model dimensions: {MODEL_WIDTH}x{MODEL_HEIGHT}")
print(f"Model name: {MODEL_NAME}")

# Define PragasLabels enum
class PragasLabels(enum.Enum):
    CHYSODEIXIS = 0
    HELICOVERPA = 1
    SPODOPTERA = 2

# Pydantic models for request and response
class ImageRequest(BaseModel):
    image_id: str
    base64: str

class InputModel(BaseModel):
    images: List[ImageRequest]

class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float

class Prediction(BaseModel):
    class_id: int
    confidence: float
    class_name: str
    bbox: BoundingBox

class ImageResponse(BaseModel):
    image_id: str
    predictions: List[Prediction] = []

class OutputModel(BaseModel):
    images: List[ImageResponse]

# Create the FastAPI app
app = FastAPI(title="Detection API", description="API for pest detection using Triton Inference Server")

# Constants from the original script
INPUT_NAMES = ["images"]
OUTPUT_NAMES = ["num_dets", "det_boxes", "det_scores", "det_classes"]

# Helper functions adapted from the original script
def preprocess(image, input_shape):
    """Preprocess an image for Triton inference."""
    height, width = input_shape
    input_img = cv2.resize(image, (width, height))
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    input_img = input_img.transpose(2, 0, 1).astype(np.float32)
    input_img /= 255.0
    return input_img

class Box:
    def __init__(self, x1, y1, x2, y2, confidence=None, classID=None):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.confidence = confidence
        self.classID = classID

    def width(self):
        return self.x2 - self.x1

    def height(self):
        return self.y2 - self.y1

    def box(self):
        return (self.x1, self.y1, self.x2, self.y2)

def postprocess(num_dets, det_boxes, det_scores, det_classes, orig_width, orig_height, input_shape):
    """Postprocess detection results."""
    boxes = []
    model_width, model_height = input_shape
    
    for i in range(int(num_dets[0])):
        x1, y1, x2, y2 = det_boxes[0][i]
        
        # Adjust to original image dimensions
        x1 = float(x1) / model_width * orig_width
        y1 = float(y1) / model_height * orig_height
        x2 = float(x2) / model_width * orig_width
        y2 = float(y2) / model_height * orig_height
        
        confidence = det_scores[0][i]
        class_id = int(det_classes[0][i])
        
        boxes.append(Box(x1, y1, x2, y2, confidence, class_id))
    
    return boxes

# FastAPI endpoint
@app.post("/detect", response_model=OutputModel)
async def detect_pests(input_data: InputModel):
    # Create Triton client
    try:
        triton_client = grpcclient.InferenceServerClient(url=TRITON_URL)
        
        # Check if server and model are ready
        if not triton_client.is_server_live():
            raise HTTPException(status_code=503, detail="Triton server is not live")
        
        if not triton_client.is_server_ready():
            raise HTTPException(status_code=503, detail="Triton server is not ready")
        
        if not triton_client.is_model_ready(MODEL_NAME):
            raise HTTPException(status_code=503, detail=f"Model {MODEL_NAME} is not ready")
        
        # Prepare response
        response_images = []
        
        # Process each image
        for image_data in input_data.images:
            # Decode base64 image using PIL and adjust orientation based on EXIF tag
            try:
                img_bytes = base64.b64decode(image_data.base64)
                image = Image.open(BytesIO(img_bytes))
                image = ImageOps.exif_transpose(image)
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                if image is None:
                    raise ValueError("Failed to decode image")
                
                # Prepare for inference
                input_image_buffer = preprocess(image, [MODEL_WIDTH, MODEL_HEIGHT])
                input_image_buffer = np.expand_dims(input_image_buffer, axis=0)
                
                # Set up inputs and outputs
                inputs = []
                outputs = []
                inputs.append(grpcclient.InferInput(INPUT_NAMES[0], [1, 3, MODEL_WIDTH, MODEL_HEIGHT], "FP32"))
                inputs[0].set_data_from_numpy(input_image_buffer)
                
                for output_name in OUTPUT_NAMES:
                    outputs.append(grpcclient.InferRequestedOutput(output_name))
                
                # Perform inference
                results = triton_client.infer(
                    model_name=MODEL_NAME,
                    inputs=inputs,
                    outputs=outputs
                )
                
                # Process inference results
                num_dets = results.as_numpy(OUTPUT_NAMES[0])
                det_boxes = results.as_numpy(OUTPUT_NAMES[1])
                det_scores = results.as_numpy(OUTPUT_NAMES[2])
                det_classes = results.as_numpy(OUTPUT_NAMES[3])
                
                # Get original image dimensions
                height, width = image.shape[:2]
                
                # Postprocess results
                detected_objects = postprocess(
                    num_dets, det_boxes, det_scores, det_classes, 
                    width, height, [MODEL_WIDTH, MODEL_HEIGHT]
                )
                
                # Convert to response format
                predictions = []
                for obj in detected_objects:
                    # Convert box coordinates to x, y, width, height format
                    x = obj.x1 / width
                    y = obj.y1 / height
                    w = obj.width() / width
                    h = obj.height() / height
                    
                    # Get class name from enum
                    class_name = PragasLabels(obj.classID).name
                    
                    predictions.append(Prediction(
                        class_id=obj.classID,
                        confidence=float(obj.confidence),
                        class_name=class_name,
                        bbox=BoundingBox(x=x, y=y, width=w, height=h)
                    ))
                
                # Add to response
                response_images.append(ImageResponse(
                    image_id=image_data.image_id,
                    predictions=predictions
                ))
                
            except Exception as e:
                # If processing fails for a specific image, add empty predictions
                response_images.append(ImageResponse(
                    image_id=image_data.image_id,
                    predictions=[]
                ))
        
        return OutputModel(images=response_images)
    
    except InferenceServerException as e:
        raise HTTPException(status_code=500, detail=f"Inference server error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    try:
        triton_client = grpcclient.InferenceServerClient(url=TRITON_URL)
        server_live = triton_client.is_server_live()
        server_ready = triton_client.is_server_ready()
        model_ready = triton_client.is_model_ready(MODEL_NAME)
        
        return {
            "status": "healthy" if (server_live and server_ready and model_ready) else "unhealthy",
            "triton_server_live": server_live,
            "triton_server_ready": server_ready,
            "model_ready": model_ready,
            "model_name": MODEL_NAME,
            "triton_url": TRITON_URL
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

# Run with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)