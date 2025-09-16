
import torch
import numpy as np


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        box1 (torch.Tensor or np.ndarray): Bounding box [x1, y1, x2, y2]
        box2 (torch.Tensor or np.ndarray): Bounding box [x1, y1, x2, y2]
    
    Returns:
        float: IoU value between 0 and 1
    """
    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Check if there's an intersection
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    # Calculate intersection area
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    # Calculate IoU
    iou = intersection / union if union > 0 else 0.0
    return iou


# def calculate_label_overlap(box1, box2, label_height=15):
#     """Calculate overlap between label positions above bounding boxes.
    
#     Args:
#         box1 (torch.Tensor or np.ndarray): First bounding box [x1, y1, x2, y2]
#         box2 (torch.Tensor or np.ndarray): Second bounding box [x1, y1, x2, y2]
#         label_height (int): Height of the label text area
    
#     Returns:
#         float: Overlap ratio (0 to 1)
#     """
#     # Label positions are above the top-left corner of bboxes
#     label1_x1, label1_y1 = box1[0], box1[1] - label_height - 10
#     label1_x2, label1_y2 = box1[0] + 30, box1[1] - 10  # Assume label width ~30px
    
#     label2_x1, label2_y1 = box2[0], box2[1] - label_height - 10
#     label2_x2, label2_y2 = box2[0] + 30, box2[1] - 10
    
#     # Calculate intersection
#     x1 = max(label1_x1, label2_x1)
#     y1 = max(label1_y1, label2_y1)
#     x2 = min(label1_x2, label2_x2)
#     y2 = min(label1_y2, label2_y2)
    
#     # Check if there's an intersection
#     if x2 <= x1 or y2 <= y1:
#         return 0.0
    
#     # Calculate intersection area
#     intersection = (x2 - x1) * (y2 - y1)
    
#     # Calculate areas
#     area1 = (label1_x2 - label1_x1) * (label1_y2 - label1_y1)
#     area2 = (label2_x2 - label2_x1) * (label2_y2 - label2_y1)
    
#     # Calculate overlap ratio as intersection over minimum area
#     min_area = min(area1, area2)
#     overlap_ratio = intersection / min_area if min_area > 0 else 0.0
    
#     return overlap_ratio


def filter_overlapping_bboxes(bboxes, bbox_threshold=0.5, label_threshold=0.5, 
                                keep_highest_score=True):
    """Filter out overlapping bounding boxes and labels from instances.
    
    Args:
        instances: Instance data containing bboxes, labels, scores, etc.
        bbox_threshold (float): IoU threshold for bbox overlap (default: 0.5)
        label_threshold (float): Overlap threshold for label positions (default: 0.5)
        keep_highest_score (bool): If True, keep bbox with highest score when overlapping
    
    Returns:
        Filtered instances with non-overlapping bboxes and labels
    """
    n_boxes = len(bboxes)
    
    if n_boxes <= 1:
        return bboxes

    # Get scores for prioritization (if available)
    scores = torch.ones(n_boxes)  # Default scores if not available
    
    # Track which boxes to keep
    keep_indices = []
    removed_indices = set()
    
    # Sort by score (highest first) if keeping highest score
    if keep_highest_score:
        sorted_indices = torch.argsort(scores, descending=True).tolist()
    else:
        sorted_indices = list(range(n_boxes))
    
    for i in sorted_indices:
        if i in removed_indices:
            continue
            
        # Check if this box overlaps with any already kept boxes
        should_keep = True
        
        for j in keep_indices:
            # Check bbox overlap
            bbox_iou = calculate_iou(bboxes[i], bboxes[j])
            
            if bbox_iou > bbox_threshold:
                should_keep = False
                break
        if should_keep:
            keep_indices.append(i)
        else:
            removed_indices.add(i)
    
    # Sort keep_indices to maintain original order
    keep_indices.sort()
    # print(f"Filtered {n_boxes - len(keep_indices)} overlapping boxes, kept {len(keep_indices)} boxes.")
    
    # Simply index the instances with keep_indices
    return bboxes[keep_indices]


def calculate_intersection_area(box1, box2):
    """Calculate intersection area of two bounding boxes.
    
    Args:
        box1: Bounding box [x1, y1, x2, y2]
        box2: Bounding box [x1, y1, x2, y2]
    
    Returns:
        float: Intersection area
    """
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Check if there's an intersection
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    # Calculate intersection area
    return (x2 - x1) * (y2 - y1)


def calculate_box_area(box):
    """Calculate area of a bounding box.
    
    Args:
        box: Bounding box [x1, y1, x2, y2]
    
    Returns:
        float: Box area
    """
    return (box[2] - box[0]) * (box[3] - box[1])



# Alternative version that only checks if smaller box is contained in larger box
def filter_smaller_contained_bboxes(bboxes, intersection_threshold=0.7, scores=None):
    """Filter out smaller bounding boxes that are contained within larger ones.
    
    This version only removes the smaller box when it's contained within a larger one,
    which is often the desired behavior for object detection post-processing.
    
    Args:
        bboxes: Tensor/array of bounding boxes with shape (N, 4) in format [x1, y1, x2, y2]
        intersection_threshold (float): Threshold for intersection/smaller_box_area ratio
        scores: Optional tensor/array of confidence scores for each box
    
    Returns:
        Filtered bounding boxes with smaller contained boxes removed
    """
    import torch
    assert scores is None
    
    n_boxes = len(bboxes)
    
    if n_boxes <= 1:
        return bboxes
    
    # Calculate areas for all boxes
    areas = torch.tensor([calculate_box_area(box) for box in bboxes])
    
    # Track which boxes to remove
    remove_indices = set()
    
    for i in range(n_boxes):
        if i in remove_indices:
            continue
            
        for j in range(i + 1, n_boxes):
            if j in remove_indices:
                continue
                
            # Calculate intersection
            intersection_area = calculate_intersection_area(bboxes[i], bboxes[j])
            
            if intersection_area == 0:
                continue
            
            # Determine which box is smaller
            if areas[i] < areas[j]:
                smaller_idx, larger_idx = i, j
            else:
                smaller_idx, larger_idx = j, i
            
            # Check if smaller box is contained in larger box
            containment_ratio = intersection_area / areas[smaller_idx]
            
            if containment_ratio > intersection_threshold:
                # Remove the smaller box (or the one with lower score if available)
                if scores is not None and scores[smaller_idx] > scores[larger_idx]:
                    remove_indices.add(larger_idx)
                else:
                    remove_indices.add(smaller_idx)
    
    # Get indices to keep
    keep_indices = [i for i in range(n_boxes) if i not in remove_indices]
    
    print(f"Filtered {len(remove_indices)} smaller contained boxes, kept {len(keep_indices)} boxes.")
    
    return bboxes[keep_indices]