import json
import jsonlines
from datetime import datetime

def odvg2coco(args):
    """
    Convert ODVG format to COCO format
    
    Args:
        args: Arguments object with:
            - input: path to ODVG jsonlines file
            - output: path for output COCO json file (optional)
            - dataset: dataset type ('coco', 'o365v1', 'o365v2', 'v3det')
    """
    
    # Set output path
    if args.output is None:
        if args.input.endswith('_od.json'):
            out_path = args.input[:-8] + '.json'
        else:
            out_path = args.input.rsplit('.', 1)[0] + '_coco.json'
    else:
        out_path = args.output
    
    # Initialize COCO structure
    coco_data = {
        "info": {
            "description": f"Converted from ODVG format",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "odvg2coco converter",
            "date_created": datetime.now().isoformat()
        },
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": []
    }
    
    # Create categories from the label mappings
    # for i, (odvg_label, coco_label) in enumerate(label_mapping.items()):
    #     # You might need to get actual category names - this is a placeholder
    #     category_name = f"category_{coco_label}"  # Replace with actual category names if available
    coco_data["categories"].append({
        "id": 1,
        "name": "data",
        "supercategory": ""
    })
    
    # Read ODVG data
    with jsonlines.open(args.input, 'r') as reader:
        odvg_data = list(reader)
    
    ann_id = 1
    
    for img_id, meta in enumerate(odvg_data, 1):
        # Add image info
        filename = meta['filename']
        
        # Handle dataset-specific filename formatting
        # if args.dataset == 'o365v2':
        #     if not (filename.startswith('images/v1/') or filename.startswith('images/v2/')):
        #         # Determine which subdirectory to use - you might need logic here
        #         # For now, defaulting to v2
        #         filename = f'images/v2/{filename}'
        
        image_info = {
            "id": img_id,
            "file_name": filename,
            "width": meta['width'],
            "height": meta['height']
        }
        coco_data["images"].append(image_info)
        
        # Add annotations
        for instance in meta['grounding']['regions']:
            bbox_xyxy = instance['bbox']
            x1, y1, x2, y2 = bbox_xyxy
            
            # Convert to COCO bbox format [x, y, width, height]
            bbox_xywh = [x1, y1, x2 - x1, y2 - y1]
            
            # Get category ID from label mapping
            # odvg_label = instance['label']
            # if odvg_label not in label_mapping:
            #     print(f"Warning: Unknown label {odvg_label}, skipping annotation")
            #     continue
                
            # category_id = label_mapping[odvg_label]
            
            # Calculate area
            area = (x2 - x1) * (y2 - y1)
            
            annotation = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": bbox_xywh,
                "area": area,
                "iscrowd": 0,
                "ignore": 0
            }
            
            coco_data["annotations"].append(annotation)
            ann_id += 1
    
    # Write COCO format file
    with open(out_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f'Saved COCO format to {out_path}')
    print(f'Images: {len(coco_data["images"])}')
    print(f'Annotations: {len(coco_data["annotations"])}')
    print(f'Categories: {len(coco_data["categories"])}')

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser('ODVG to COCO format converter.', add_help=True)
    parser.add_argument('input', type=str, help='input ODVG jsonlines file name')
    parser.add_argument('--output', '-o', type=str, help='output COCO json file name')
    
    args = parser.parse_args()
    odvg2coco(args)