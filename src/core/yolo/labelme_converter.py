import json
import numpy as np
import supervision as sv
from pathlib import Path
import argparse
import cv2
from typing import List, Optional, Tuple, Dict
import os
import random
from collections import defaultdict
from prettytable import PrettyTable

def read_labelme_json(json_path: str) -> dict:
    with open(json_path, 'r') as f:
        return json.load(f)

def extract_polygons_and_labels(shapes: List[dict], filter_labels: List[str]) -> Tuple[List[np.ndarray], List[str]]:
    polygons, labels = [], []
    for shape in shapes:
        if shape['shape_type'] == 'polygon' and len(shape['points']) >= 3 and shape['label'].lower() not in filter_labels:
            polygons.append(np.array(shape['points']))
            labels.append(shape['label'])
    return polygons, labels

def modify_image_path(original_path: str, new_image_dir: Optional[str]) -> str:
    if not new_image_dir:
        return original_path
    return os.path.join(new_image_dir, os.path.basename(original_path))

def process_single_json(
    json_path: str, 
    classes: List[str], 
    new_image_dir: Optional[str] = None,
    filter_labels: List[str] = ["Indefinido"]
) -> Tuple[Optional[str], Optional[sv.Detections]]:
    data = read_labelme_json(json_path)
    if 'shapes' not in data:
        return None, None
    
    polygons, labels = extract_polygons_and_labels(data['shapes'], filter_labels)
    if not polygons:
        return None, None
    
    image_height, image_width = data['imageHeight'], data['imageWidth']
    class_id_map = {label: idx for idx, label in enumerate(classes)}
    valid_indices = [i for i, label in enumerate(labels) if label in class_id_map]
    
    if not valid_indices:
        return None, None
    
    polygons = [polygons[i] for i in valid_indices]
    labels = [labels[i] for i in valid_indices]
    masks = []
    for polygon in polygons:
        mask = np.zeros((image_height, image_width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
        masks.append(mask)
    masks = np.stack(masks) if masks else np.zeros((0, image_height, image_width))
    xyxy = np.array([sv.polygon_to_xyxy(p) for p in polygons])
    class_ids = np.array([class_id_map[label] for label in labels])
    
    return (
        modify_image_path(data['imagePath'], new_image_dir),
        sv.Detections(xyxy=xyxy, mask=masks, class_id=class_ids)
    )

def get_all_labels(json_dir: str, filter_labels: List[str]) -> List[str]:
    json_files = list(Path(json_dir).glob('*.json'))
    all_labels = set()
    for json_file in json_files:
        data = read_labelme_json(str(json_file))
        if 'shapes' in data:
            for shape in data['shapes']:
                if shape['shape_type'] == 'polygon' and shape['label'] not in filter_labels:
                    all_labels.add(shape['label'])
    return sorted(all_labels)

def collect_class_statistics(json_files: List[Path], classes: List[str], filter_labels: List[str]) -> Dict[str, Dict[str, int]]:
    """
    Collects statistics for each class across provided JSON files.
    Returns a dictionary mapping class labels to counts of images and instances.
    """
    stats = {cls: {'images': 0, 'instances': 0} for cls in classes}
    images_with_class = {cls: set() for cls in classes}
    
    for json_file in json_files:
        data = read_labelme_json(str(json_file))
        if 'shapes' not in data:
            continue
            
        # Get all valid labels in this file
        file_labels = []
        for shape in data['shapes']:
            if shape['shape_type'] == 'polygon' and shape['label'] in classes and shape['label'] not in filter_labels:
                file_labels.append(shape['label'])
                
        # Count unique classes and instances
        for label in set(file_labels):
            images_with_class[label].add(str(json_file))
        
        for label in file_labels:
            stats[label]['instances'] += 1
    
    # Set the image count for each class
    for cls in classes:
        stats[cls]['images'] = len(images_with_class[cls])
    
    # Get total unique images across all classes
    unique_images = set()
    for cls in classes:
        unique_images.update(images_with_class[cls])
    
    # Add total count
    stats['total'] = {
        'images': len(unique_images),
        'instances': sum(stats[cls]['instances'] for cls in classes)
    }
            
    return stats

def create_dataset_batch(
    json_files: List[Path],
    classes: List[str],
    new_image_dir: Optional[str],
    filter_labels: List[str] = ["Indefinido"]
) -> sv.DetectionDataset:
    """Process a batch of JSON files and create a dataset"""
    annotations = {}
    image_paths = []
    for json_file in json_files:
        image_path, detections = process_single_json(str(json_file), classes, new_image_dir, filter_labels)
        if image_path and detections:
            annotations[image_path] = detections
            image_paths.append(image_path)
    return sv.DetectionDataset(classes=classes, images=image_paths, annotations=annotations)

def validate_splits(splits: Optional[List[float]]) -> None:
    if splits and (len(splits) not in [2, 3] or abs(sum(splits) - 1.0) > 1e-6 or any(s <= 0 for s in splits)):
        raise ValueError("Invalid split ratios")

def write_image_paths(ds: sv.DetectionDataset, output_images: str, phase: str) -> None:
    with open(Path(output_images).parent / f'{phase}.txt', 'a+') as f:
        for path, _, _ in ds:
            path_str = os.path.join(Path(output_images).stem, f'{phase}', Path(path).name)
            f.write('./' + path_str + '\n')

def print_split_statistics_table(all_stats: Dict[str, Dict[str, Dict[str, int]]], classes: List[str]) -> None:
    """
    Prints a detailed statistics table for each split and total.
    """
    # Create a table for images count
    img_table = PrettyTable()
    img_table.field_names = ["Class", "Train Images", "Val Images", "Test Images", "Total Images"]
    
    # Create a table for instances count
    inst_table = PrettyTable()
    inst_table.field_names = ["Class", "Train Instances", "Val Instances", "Test Instances", "Total Instances"]
    
    # Compute total counts across all splits
    total_stats = {
        cls: {
            'images': sum(all_stats[split][cls]['images'] for split in all_stats if cls in all_stats[split]),
            'instances': sum(all_stats[split][cls]['instances'] for split in all_stats if cls in all_stats[split])
        } for cls in classes
    }
    
    # Add row for each class
    for cls in sorted(classes):
        img_row = [cls]
        inst_row = [cls]
        
        for split in ['train', 'val', 'test']:
            if split in all_stats and cls in all_stats[split]:
                img_row.append(all_stats[split][cls]['images'])
                inst_row.append(all_stats[split][cls]['instances'])
            else:
                img_row.append(0)
                inst_row.append(0)
                
        # Add total column
        img_row.append(total_stats[cls]['images'])
        inst_row.append(total_stats[cls]['instances'])
        
        img_table.add_row(img_row)
        inst_table.add_row(inst_row)
    
    # Add total row
    img_total_row = ["TOTAL"]
    inst_total_row = ["TOTAL"]
    
    for split in ['train', 'val', 'test']:
        if split in all_stats and 'total' in all_stats[split]:
            img_total_row.append(all_stats[split]['total']['images'])
            inst_total_row.append(all_stats[split]['total']['instances'])
        else:
            img_total_row.append(0)
            inst_total_row.append(0)
    
    # Calculate grand totals
    grand_total_images = sum(all_stats[split]['total']['images'] for split in all_stats if 'total' in all_stats[split])
    grand_total_instances = sum(all_stats[split]['total']['instances'] for split in all_stats if 'total' in all_stats[split])
    
    img_total_row.append(grand_total_images)
    inst_total_row.append(grand_total_instances)
    
    img_table.add_row(img_total_row)
    inst_table.add_row(inst_total_row)
    
    print("\n=== Dataset Statistics by Split ===")
    print("\nImage Counts:")
    print(img_table)
    print("\nInstance Counts:")
    print(inst_table)
    print("\nNote: The total image count for each split may not equal the sum of individual class image counts")
    print("because an image can contain multiple classes.")

def export_dataset_in_batches(
    json_files: List[Path], 
    split_name: str,
    classes: List[str],
    batch_size: int,
    output_format: str,
    image_dir: Optional[str],
    output_images: Optional[str],
    output_annotations: str,
    data_yaml_path: Optional[str],
    min_area: float,
    max_area: float,
    approx: float,
    filter_labels: List[str]
):
    """Process and export dataset in batches"""
    if not json_files:
        return
    
    # Initialize paths based on the split name
    images_dir = os.path.join(output_images, split_name) if output_images else None
    
    if output_format == 'coco':
        # For COCO, we need to accumulate batches and export once at the end
        all_annotations = {}
        all_image_paths = []
        
        for i in range(0, len(json_files), batch_size):
            print(f"Processing {split_name} batch {i//batch_size + 1}/{(len(json_files) + batch_size - 1)//batch_size}")
            batch_files = json_files[i:i+batch_size]
            batch_dataset = create_dataset_batch(batch_files, classes, image_dir, filter_labels)
            
            # Accumulate annotations and image paths
            for image_path, annotation, _ in batch_dataset:
                all_annotations[image_path] = annotation
                all_image_paths.append(image_path)
        
        # Create a complete dataset and export once
        full_dataset = sv.DetectionDataset(classes=classes, images=all_image_paths, annotations=all_annotations)
        annotations_path = output_annotations.replace('.json', f'_{split_name}.json')
        
        print(f"Exporting {split_name} dataset in COCO format")
        full_dataset.as_coco(
            images_directory_path=images_dir,
            annotations_path=annotations_path,
            min_image_area_percentage=min_area,
            max_image_area_percentage=max_area,
            approximation_percentage=approx
        )
        
    else:  # YOLO format
        # For YOLO, we can process and export each batch directly
        annotations_dir = os.path.join(output_annotations, split_name)
        # Ensure directory exists
        os.makedirs(annotations_dir, exist_ok=True)
        
        for i in range(0, len(json_files), batch_size):
            print(f"Processing {split_name} batch {i//batch_size + 1}/{(len(json_files) + batch_size - 1)//batch_size}")
            batch_files = json_files[i:i+batch_size]
            batch_dataset = create_dataset_batch(batch_files, classes, image_dir, filter_labels)
            
            # Export the batch
            should_export_yaml = (split_name == 'train' and i == 0)  # Only export YAML with first train batch
            
            batch_dataset.as_yolo(
                images_directory_path=images_dir,
                annotations_directory_path=annotations_dir,
                data_yaml_path=data_yaml_path if should_export_yaml else None,
                min_image_area_percentage=min_area,
                max_image_area_percentage=max_area,
                approximation_percentage=approx
            )
            
            # Write image paths for this batch
            write_image_paths(batch_dataset, output_images, split_name)

def main():
    parser = argparse.ArgumentParser(description='Optimized dataset conversion with batching')
    parser.add_argument('json_dir', type=str, help='Directory containing LabelMe JSON files')
    parser.add_argument('--output-images', type=str, help='Directory to save images')
    parser.add_argument('--output-annotations', type=str, required=True, help='Annotations output path')
    parser.add_argument('--min-area', type=float, default=0.0, help='Minimum detection area percentage')
    parser.add_argument('--max-area', type=float, default=1.0, help='Maximum detection area percentage')
    parser.add_argument('--approx', type=float, default=0.75, help='Polygon approximation percentage')
    parser.add_argument('--image-dir', type=str, help='New directory path for images')
    parser.add_argument('--format', choices=['coco', 'yolo'], default='coco', help='Output format')
    parser.add_argument('--split', type=float, nargs='+', help='Train/val[/test] split ratios')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed')
    parser.add_argument('--data-yaml', type=str, help='Path for data.yaml')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for processing')
    parser.add_argument('--filter-labels', type=str, nargs='+', default=["Indefinido"], 
                        help='List of labels to filter out (e.g., "undefined", "indefinido")')
    args = parser.parse_args()

    validate_splits(args.split)
    
    # Get all classes
    print("Collecting label information...")
    classes = get_all_labels(args.json_dir, args.filter_labels)
    print(f"Found {len(classes)} classes: {', '.join(classes)}")
    
    # Get all JSON files
    json_files = list(Path(args.json_dir).glob('*.json'))
    print(f"Found {len(json_files)} JSON files")
    
    # Shuffle files if needed
    random.seed(args.random_seed)
    random.shuffle(json_files)

    # Dictionary to store statistics for each split
    all_split_stats = {}

    # Process train/val/test split
    if args.split:
        splits = args.split
        train_ratio, val_ratio = splits[0], splits[1] if len(splits) >= 2 else 0
        train_end = int(len(json_files) * train_ratio)
        val_end = train_end + int(len(json_files) * val_ratio) if len(splits) == 3 else len(json_files)
        
        # Split files
        train_files = json_files[:train_end]
        all_split_stats['train'] = collect_class_statistics(train_files, classes, args.filter_labels)
        print(f"Train split: {len(train_files)} files")
        
        if val_end > train_end:
            val_files = json_files[train_end:val_end]
            all_split_stats['val'] = collect_class_statistics(val_files, classes, args.filter_labels)
            print(f"Validation split: {len(val_files)} files")
        else:
            val_files = []
            
        if val_end < len(json_files):
            test_files = json_files[val_end:]
            all_split_stats['test'] = collect_class_statistics(test_files, classes, args.filter_labels)
            print(f"Test split: {len(test_files)} files")
        else:
            test_files = []
    else:
        train_files, val_files, test_files = json_files, [], []
        all_split_stats['train'] = collect_class_statistics(train_files, classes, args.filter_labels)
        print(f"No split specified. Using all {len(train_files)} files for training.")

    # Create output directories if needed
    if args.output_images:
        for split in ['train', 'val', 'test']:
            if (split == 'train' and train_files) or \
               (split == 'val' and val_files) or \
               (split == 'test' and test_files):
                os.makedirs(os.path.join(args.output_images, split), exist_ok=True)
    
    if args.format == 'yolo' and args.output_annotations:
        for split in ['train', 'val', 'test']:
            if (split == 'train' and train_files) or \
               (split == 'val' and val_files) or \
               (split == 'test' and test_files):
                os.makedirs(os.path.join(args.output_annotations, split), exist_ok=True)

    # Process and export each split
    print("\nExporting datasets in batches...")
    
    if train_files:
        export_dataset_in_batches(
            train_files, 'train', classes, args.batch_size, args.format,
            args.image_dir, args.output_images, args.output_annotations,
            args.data_yaml, args.min_area, args.max_area, args.approx, args.filter_labels
        )
    
    if val_files:
        export_dataset_in_batches(
            val_files, 'val', classes, args.batch_size, args.format,
            args.image_dir, args.output_images, args.output_annotations,
            None, args.min_area, args.max_area, args.approx, args.filter_labels
        )
    
    if test_files:
        export_dataset_in_batches(
            test_files, 'test', classes, args.batch_size, args.format,
            args.image_dir, args.output_images, args.output_annotations,
            None, args.min_area, args.max_area, args.approx, args.filter_labels
        )
    
    # Print detailed statistics table with splits
    print_split_statistics_table(all_split_stats, classes)

if __name__ == "__main__":
    main()