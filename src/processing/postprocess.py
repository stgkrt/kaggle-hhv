import cv2
import numpy as np


def remove_small_objects(mask, min_size, threshold):
    mask = ((mask > threshold) * 255).astype(np.uint8)
    # find all connected components (labels)
    num_label, label, stats, centroid = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    # create a mask where small objects are removed
    processed = np.zeros_like(mask)
    for label_idx in range(1, num_label):
        if stats[label_idx, cv2.CC_STAT_AREA] >= min_size:
            processed[label == label_idx] = 1
    return processed
