from dataclasses import dataclass
import glob
import json
import os
import sys
from typing import Any, Dict

import requests
from Src.core.scheduler import Scheduler

@dataclass
class PredictionImage:
    image_id: str
    image_base64: str

def detect_pests(api_url: str, prediction_images: list[PredictionImage]) -> Dict[str, Any]:
    """Send image to API for pest detection."""
    print(f"Sending {len(prediction_images)} images to API for pest detection")
    try:
        payload = {
            "images": [
                {
                    "image_id": prediction_image.image_id,
                    "base64": prediction_image.image_base64
                }
            for prediction_image in prediction_images]
        }
        
        # Make the request
        print(f"Making POST request to {api_url}/detect")
        response = requests.post(f"{api_url}/detect", json=payload)
        
        if response.status_code != 200:
            print(f"API request failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            sys.exit(1)
        
        print(f"API request successful with status code: {response.status_code}")
        return response.json()
    except Exception as e:
        print(f"Error calling API: {str(e)}")
        sys.exit(1)

def execute_pest_detection() -> None:
    """Execute pest detection."""
    print("Starting pest detection execution")
    api_url = "http://fastapi-app:8000"  # Replace with your API URL
    input_dir = "input/pragas"
    output_dir = "output"
    
    print(f"Looking for JSON files in {input_dir}")
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    print(f"Found {len(json_files)} JSON files")
    
    prediction_images = []
    for json_file in json_files:
        print(f"Processing file: {json_file}")
        with open(json_file, "r") as f:
            json_list = json.load(f)
        
        print(f"File contains {len(json_list)} entries")
        for json_data in json_list:
            image_id = json_data['id']
            type_img = json_data['metadata']['type_img']
            
            print(f"Processing image ID: {image_id}, type: {type_img}")
            image_base64 = json_data['metadata']['image']
            
            image_base64 = image_base64[image_base64.find('base64,') + 7:]
            prediction_images.append(PredictionImage(image_id, image_base64))

        print(f"Calling pest detection API with {len(prediction_images)} images")
        results = detect_pests(api_url, prediction_images)
        
        # Save the results to a file
        output_file = os.path.join(output_dir, f"pragas_output.json")
        print(f"Saving results to {output_file}")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
        print("Pest detection completed successfully")

if __name__ == "__main__":
    print("Starting scheduler")
    scheduler = Scheduler(task_function=execute_pest_detection, 
                        schedule_time="03:00", 
                        run_on_start=True)
    print(f"Scheduler initialized with run time 03:00, starting now")
    scheduler.start()