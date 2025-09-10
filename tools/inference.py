#!/usr/bin/env python3
"""
MMDetection Batch Inference Script

This script performs inference on multiple images using MMDetection's DetInferencer.
It supports two input modes:
1. Annotation JSON file with COCO-style image metadata
2. Directory containing image files

Usage:
    python batch_inference.py --model rtmdet_tiny_8xb32-300e_coco --input /path/to/images --output /path/to/results
    python batch_inference.py --model rtmdet_tiny_8xb32-300e_coco --annotation /path/to/annotations.json --input /path/to/images --output /path/to/results
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Union
import torch
from tqdm import tqdm

try:
    from mmdet.apis import DetInferencer
except ImportError:
    print("ERROR: MMDetection not installed. Please install with: pip install mmdet")
    sys.exit(1)


class BatchInferencer:
    """Handles batch inference for MMDetection models."""

    def __init__(self, model_name: str, weights: str = None, text_prompt=None, device: str = 'cuda', batch_size: int = 8):
        """
        Initialize the batch inferencer.
        
        Args:
            model_name: Name of the MMDetection model
            device: Device to run inference on ('cuda', 'cpu', or specific GPU like 'cuda:0')
            batch_size: Number of images to process in each batch
        """
        self.model_name = model_name
        self.weights = weights
        self.device = device
        self.batch_size = batch_size
        self.text_prompt = text_prompt
        
        print(f"Initializing DetInferencer with model: {model_name}")
        print(f"Device: {device}")
        
        try:
            self.inferencer = DetInferencer(model_name, weights=weights, device=device)
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"ERROR: Failed to load model {model_name}: {e}")
            sys.exit(1)
    
    def get_images_from_annotation(self, annotation_path: str, image_dir: str) -> List[str]:
        """
        Extract image paths from annotation JSON file.
        
        Args:
            annotation_path: Path to the annotation JSON file
            image_dir: Directory containing the images
            
        Returns:
            List of image file paths
        """
        print(f"Loading annotations from: {annotation_path}")
        
        try:
            with open(annotation_path, 'r') as f:
                annotations = json.load(f)
        except Exception as e:
            print(f"ERROR: Failed to load annotation file: {e}")
            sys.exit(1)
        
        if 'images' not in annotations:
            print("ERROR: 'images' key not found in annotation file")
            sys.exit(1)
        
        image_paths = []
        missing_images = []
        
        for img_info in annotations['images']:
            if 'file_name' not in img_info:
                print(f"WARNING: Missing 'file_name' in image info: {img_info}")
                continue
                
            img_path = os.path.join(image_dir, img_info['file_name'])
            
            if os.path.exists(img_path):
                image_paths.append(img_path)
            else:
                missing_images.append(img_path)
        
        if missing_images:
            print(f"WARNING: {len(missing_images)} images not found:")
            for missing in missing_images[:10]:  # Show first 10
                print(f"  - {missing}")
            if len(missing_images) > 10:
                print(f"  ... and {len(missing_images) - 10} more")
        
        print(f"✓ Found {len(image_paths)} valid images from annotations")
        return image_paths
    
    def get_images_from_directory(self, image_dir: str) -> List[str]:
        """
        Get all image files from a directory.
        
        Args:
            image_dir: Directory containing images
            
        Returns:
            List of image file paths
        """
        print(f"Scanning directory: {image_dir}")
        
        # Common image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        
        image_paths = []
        
        for ext in image_extensions:
            # Case insensitive search
            image_paths.extend(Path(image_dir).glob(f"*{ext}"))
            image_paths.extend(Path(image_dir).glob(f"*{ext.upper()}"))
        
        # Convert to strings and sort
        image_paths = sorted([str(p) for p in image_paths])
        
        print(f"✓ Found {len(image_paths)} images in directory")
        return image_paths
    
    def run_inference(self, image_paths: List[str], output_dir: str, 
                     save_pred: bool = True, save_vis: bool = True):
        """
        Run inference on all images.
        
        Args:
            image_paths: List of image file paths
            output_dir: Output directory for results
            save_pred: Whether to save prediction results
            save_vis: Whether to save visualizations
        """
        if not image_paths:
            print("No images to process")
            return
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Processing {len(image_paths)} images...")
        print(f"Output directory: {output_dir}")
        print(f"Batch size: {self.batch_size}")
        
        # Process images in batches
        total_batches = (len(image_paths) + self.batch_size - 1) // self.batch_size
        
        with tqdm(total=len(image_paths), desc="Processing images") as pbar:
            for i in range(0, len(image_paths), self.batch_size):
                batch_paths = image_paths[i:i + self.batch_size]
                batch_num = i // self.batch_size + 1
                
                # try:
                # Run inference on batch
                # import pdb; pdb.set_trace()
                self.inferencer(
                    batch_paths,
                    with_indices=True,
                    out_dir=output_dir,
                    texts=self.text_prompt,
                    no_save_pred=not save_pred,
                    no_save_vis=not save_vis
                )
                pbar.update(len(batch_paths))
                    
                # except Exception as e:
                #     print(f"\nERROR processing batch {batch_num}: {e}")
                #     # Continue with next batch
                #     pbar.update(len(batch_paths))
                #     continue
        
        print(f"✓ Inference completed. Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Batch inference with MMDetection")
    
    parser.add_argument('--model', '-m', type=str, required=True,
                       help='MMDetection model name (e.g., rtmdet_tiny_8xb32-300e_coco)')

    parser.add_argument('--weights', '-w', type=str, required=True,
                       help='Path to the model weights file (e.g., model.pth)')

    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input directory containing images')
    
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output directory for results')
    
    parser.add_argument('--annotation', '-a', type=str, default=None,
                       help='Optional: JSON annotation file with image metadata')
    parser.add_argument('--text-prompt', '-t', type=str, default="data.",)
    
    parser.add_argument('--device', '-d', type=str, default='cuda',
                       help='Device for inference (default: cuda)')
    
    parser.add_argument('--batch-size', '-b', type=int, default=8,
                       help='Batch size for inference (default: 8)')
    
    parser.add_argument('--no-vis', action='store_true',
                       help='Skip saving visualizations (only save predictions)')
    
    parser.add_argument('--no-pred', action='store_true',
                       help='Skip saving predictions (only save visualizations)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.input):
        # import pdb; pdb.set_trace()
        print(f"ERROR: Input directory does not exist: {args.input}")
        sys.exit(1)
    
    elif args.annotation and not os.path.exists(args.annotation):
        print(f"ERROR: Annotation file does not exist: {args.annotation}")
        sys.exit(1)
    
    # Check GPU availability
    if 'cuda' in args.device and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = 'cpu'
    elif 'cuda' in args.device:
        print(f"Available GPUs: {torch.cuda.device_count()}")
    
    # Initialize inferencer
    inferencer = BatchInferencer(
        model_name=args.model,
        weights=args.weights,
        device=args.device,
        batch_size=args.batch_size,
        text_prompt=args.text_prompt,
    )
    
    # Get image paths
    # import pdb; pdb.set_trace()
    if args.annotation:
        image_paths = inferencer.get_images_from_annotation(args.annotation, args.input)
    else:
        image_paths = inferencer.get_images_from_directory(args.input)
    
    if not image_paths:
        print("No images found to process")
        sys.exit(1)
    
    # Run inference
    inferencer.run_inference(
        image_paths=image_paths,
        output_dir=args.output,
        save_pred=not args.no_pred,
        save_vis=not args.no_vis
    )


if __name__ == '__main__':
    main()