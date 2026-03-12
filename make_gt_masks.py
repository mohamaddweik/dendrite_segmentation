import os
import cv2
import numpy as np
from pathlib import Path

# This script converts YOLO-format text labels into binary mask images.
# Each polygon defined in the YOLO labels is drawn as a filled white shape on a black background,
# creating a mask that can be used as ground truth for segmentation tasks.

def convert_yolo_to_masks(images_folder, labels_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    images = [p for p in Path(images_folder).iterdir() if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif']]
    print(f"Found {len(images)} images. Drawing Ground Truth masks...")
    
    for img_path in images:
        # Load image just to get its exact height and width
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        
        # Create a blank black mask
        mask = np.zeros((H, W), dtype=np.uint8)
        
        # Find the matching text label file
        label_path = os.path.join(labels_folder, img_path.stem + ".txt")
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                    
                # Skip the class ID (parts[0]), take the rest as coordinates
                coords = np.array([float(x) for x in parts[1:]])
                
                # Reshape into pairs of (x, y)
                pts = coords.reshape(-1, 2)
                
                # Un-normalize (multiply x by Width, y by Height)
                pts[:, 0] *= W
                pts[:, 1] *= H
                
                # Convert to integers so OpenCV can draw them
                pts = pts.astype(np.int32)
                
                # Draw the filled polygon in white (255)
                cv2.fillPoly(mask, [pts], 255)
        
        # Save the mask
        out_path = os.path.join(output_folder, img_path.stem + ".png")
        cv2.imwrite(out_path, mask)
        
    print(f"Done! Ground Truth masks saved to: {output_folder}")

if __name__ == "__main__":
    # --- UPDATE THESE PATHS IF NEEDED ---
    IMAGES_DIR = r"dendrite_dataset\images\test"
    LABELS_DIR = r"dendrite_dataset\labels\test"
    OUTPUT_DIR = r"dendrite_dataset\labels_masks\test"
    
    convert_yolo_to_masks(IMAGES_DIR, LABELS_DIR, OUTPUT_DIR)