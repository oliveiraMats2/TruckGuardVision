#!/usr/bin/env python3
import requests
import argparse
import base64
import cv2
import numpy as np
import json
import sys
import random
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import io

# Generate random colors for each class
colors = {
    'CHYSODEIXIS': (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
    'HELICOVERPA': (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
    'SPODOPTERA': (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
}

def check_health(api_url: str) -> bool:
    """Check if the API and Triton server are healthy."""
    try:
        response = requests.get(f"{api_url}/health")
        if response.status_code != 200:
            print(f"Health check failed with status code: {response.status_code}")
            return False
        
        health_data = response.json()
        if health_data['status'] != 'healthy':
            print(f"API reports unhealthy status: {health_data}")
            return False
        
        print("API and Triton server are healthy")
        return True
    except Exception as e:
        print(f"Error checking API health: {str(e)}")
        return False

def encode_image(image_path: str) -> str:
    """Encode an image file to base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {str(e)}")
        sys.exit(1)

def detect_pests(api_url: str, image_path: str) -> Dict[str, Any]:
    """Send image to API for pest detection."""
    try:
        # Prepare the request
        base64_image = encode_image(image_path)
        image_id = image_path.split('/')[-1]
        
        payload = {
            "images": [
                {
                    "image_id": image_id,
                    "base64": base64_image
                }
            ]
        }
        
        # Make the request
        response = requests.post(f"{api_url}/detect", json=payload)
        
        if response.status_code != 200:
            print(f"API request failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            sys.exit(1)
        
        return response.json()
    except Exception as e:
        print(f"Error calling API: {str(e)}")
        sys.exit(1)

def visualize_results(image_path: str, results: Dict[str, Any], output_path: str = None,
                    font_size: int = 12, line_thickness: int = 2):
    """Visualize detection results on the image."""
    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            sys.exit(1)
        
        # Get image dimensions
        height, width, _ = image.shape
        
        # Convert to RGB for matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process each image in the results
        for img_result in results['images']:
            print(f"Processing results for image: {img_result['image_id']}")
            print(f"Found {len(img_result['predictions'])} predictions")
            
            # Draw each prediction
            for pred in img_result['predictions']:
                # Get class and confidence
                class_name = pred['class_name']
                confidence = pred['confidence']
                
                # Get bounding box coordinates
                bbox = pred['bbox']
                x1 = int(bbox['x'] * width)
                y1 = int(bbox['y'] * height)
                w = int(bbox['width'] * width)
                h = int(bbox['height'] * height)
                
                # Draw the box
                color = colors.get(class_name, (0, 255, 0))  # Default to green if class not in colors
                
                # OpenCV accepts colors as BGR
                color_bgr = (color[2], color[1], color[0])
                
                # Draw rectangle
                cv2.rectangle(image, (x1, y1), (x1 + w, y1 + h), color_bgr, line_thickness)
                
                # Prepare label text
                label = f"{class_name}: {confidence:.2f}"
                
                # Get text size
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                # Draw label background
                cv2.rectangle(image, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color_bgr, -1)
                
                # Draw text
                cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # add rectangle as background for text
                cv2.rectangle(image, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color_bgr, 2)
                print(f"  - {class_name}: confidence={confidence:.2f}, bbox={bbox}")
        
        # Display or save the image
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"Results saved to {output_path}")
        else:
            cv2.imshow('Pest Detection Results', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error visualizing results: {str(e)}")
        sys.exit(1)

def main():
    # Check if API and Triton server are healthy
    if not check_health(args.api_url):
        print("API health check failed. Exiting.")
        sys.exit(1)
    
    # Send the image for detection
    print(f"Sending image {args.image} to API for pest detection...")
    results = detect_pests(args.api_url, args.image)
    
    # Visualize the results
        
    visualize_results(
        args.image, 
        results, 
        args.output, 
        args.font_size,
        args.line_thickness
    )

if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser(description='Test the Pest Detection API')
    parser.add_argument('image', type=str, help='Path to the image file')
    parser.add_argument('--api_url', type=str, default='http://localhost:8080', help='Base URL of the API')
    parser.add_argument('--output', type=str, help='Output image path (if not provided, will display instead)')
    parser.add_argument('--font_size', type=int, default=32, help='Font size for labels')
    parser.add_argument('--line_thickness', type=int, default=5, help='Thickness of bounding box lines')
    args = parser.parse_args()

    main()