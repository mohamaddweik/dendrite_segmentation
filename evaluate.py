import os
import cv2
import numpy as np
from pathlib import Path

def calculate_metrics(gt_folder, pred_folder):
    gt_paths = [p for p in Path(gt_folder).iterdir() if p.suffix.lower() in ['.png', '.jpg', '.tif']]
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for gt_path in gt_paths:
        # Load Ground Truth
        gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        
        pred_path = os.path.join(pred_folder, f"{gt_path.stem}_mask.png")
        
        if not os.path.exists(pred_path):
            # Fallback just in case it's named exactly the same
            pred_path = os.path.join(pred_folder, gt_path.name)
            if not os.path.exists(pred_path):
                print(f"Skipping {gt_path.name} - no matching prediction found.")
                continue
                
        pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize prediction to match Ground truth just in case
        if gt_mask.shape != pred_mask.shape:
            pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Binarize masks (0 or 1)
        gt_bin = (gt_mask > 127).astype(np.uint8)
        pred_bin = (pred_mask > 127).astype(np.uint8)
        
        # Calculate True Positives, False Positives, False Negatives
        tp = np.logical_and(pred_bin == 1, gt_bin == 1).sum()
        fp = np.logical_and(pred_bin == 1, gt_bin == 0).sum()
        fn = np.logical_and(pred_bin == 0, gt_bin == 1).sum()
        
        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Calculate final metrics
    epsilon = 1e-7 # Prevent division by zero
    precision = total_tp / (total_tp + total_fp + epsilon)
    recall = total_tp / (total_tp + total_fn + epsilon)
    iou = total_tp / (total_tp + total_fp + total_fn + epsilon)
    
    return iou * 100, precision * 100, recall * 100

if __name__ == "__main__":
    GT_DIR = r"dendrite_dataset\labels_masks\val"
    YOLO_PRED_DIR = r"runs\segment\pred_tiled_out_2\masks"
    CV_PRED_DIR = r"runs\segment\cv_out\masks" 
    
    print("--- YOLO Model Results ---")
    iou_y, prec_y, rec_y = calculate_metrics(GT_DIR, YOLO_PRED_DIR)
    print(f"IoU: {iou_y:.2f}% | Precision: {prec_y:.2f}% | Recall: {rec_y:.2f}%\n")
    
    print("--- Classic CV Results ---")
    iou_c, prec_c, rec_c = calculate_metrics(GT_DIR, CV_PRED_DIR)
    print(f"IoU: {iou_c:.2f}% | Precision: {prec_c:.2f}% | Recall: {rec_c:.2f}%")