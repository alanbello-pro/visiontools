import cv2
import numpy as np

def check_bbox_in_masks(bbox, masks):

    x1, y1, x2, y2 = map(float, bbox)
    
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    center = (center_x, center_y)

    for mask_name, mask_coords in masks.items():
        mask_np = np.array(mask_coords, np.int32)
        mask_np = mask_np.reshape((-1, 1, 2))
        if cv2.pointPolygonTest(mask_np, center, False) > 0:
            return mask_name
    return None

def calcular_iou(box_a, box_b) -> float:

    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    union_area = box_a_area + box_b_area - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area

def scale_bounding_box(box: np.ndarray, scale_factor: float) -> np.ndarray:

    if scale_factor == 1.0:
        return box
    
    scaled = box.copy()
    scaled[[0, 2]] /= scale_factor
    scaled[[1, 3]] /= scale_factor
    return scaled