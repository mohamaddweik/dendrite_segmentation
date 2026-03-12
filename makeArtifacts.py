import os
import cv2
import numpy as np
from pathlib import Path

def create_comparison_artifacts(img_dir, cv_overlay_dir, yolo_overlay_dir, skel_overlay_dir, out_dir, num_samples=5):
    os.makedirs(out_dir, exist_ok=True)
    
    # Get all validation images
    img_paths = list(Path(img_dir).glob("*.*"))
    sample_paths = img_paths[:num_samples]
    
    for path in sample_paths:
        base_name = path.stem
        
        # Load the 4 images (Original, CV Overlay, YOLO Overlay, Skeleton Overlay)
        img = cv2.imread(str(path))
        
        # NOTE: Adjust the filenames here if your overlay images have different endings!
        cv_overlay = cv2.imread(os.path.join(cv_overlay_dir, f"{base_name}_overlay.png"))
        
        # Assuming YOLO overlays end in .jpg based on your uploaded images earlier
        yolo_overlay = cv2.imread(os.path.join(yolo_overlay_dir, f"{base_name}_overlay.jpg")) 
        if yolo_overlay is None: # fallback to png just in case
            yolo_overlay = cv2.imread(os.path.join(yolo_overlay_dir, f"{base_name}_overlay.png"))
            
        skel_overlay = cv2.imread(os.path.join(skel_overlay_dir, f"{base_name}_skeleton.png"))
        
        # Fallback if an image is missing
        if any(x is None for x in [img, cv_overlay, yolo_overlay, skel_overlay]):
            print(f"Skipping {base_name}: Missing one of the component images. Check your folder paths!")
            continue
            
        target_h, target_w = img.shape[:2]
        
        def process_panel(panel, title):
            if panel.shape[:2] != (target_h, target_w):
                panel = cv2.resize(panel, (target_w, target_h))
            if len(panel.shape) == 2:
                panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
            
            # Add a black bar at the top for the title
            header_h = 50
            header = np.zeros((header_h, target_w, 3), dtype=np.uint8)
            cv2.putText(header, title, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            
            return np.vstack((header, panel))

        # Prepare panels
        p1 = process_panel(img, "1. Original")
        p2 = process_panel(cv_overlay, "2. Classic CV Overlay")
        p3 = process_panel(yolo_overlay, "3. YOLO Overlay")
        p4 = process_panel(skel_overlay, "4. Skeleton Overlay")
        
        # Stitch them side-by-side
        final_artifact = np.hstack((p1, p2, p3, p4))
        
        # Save
        out_path = os.path.join(out_dir, f"Artifact_{base_name}.jpg")
        cv2.imwrite(out_path, final_artifact)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    # --- UPDATE THESE TO YOUR EXACT FOLDERS ---
    IMG_DIR = r"dendrite_dataset\images\test"
    CV_OVERLAY_DIR = r"runs\segment\cv_out\overlays"
    YOLO_OVERLAY_DIR = r"runs\segment\pred_tiled_out_final\overlays"
    SKEL_OVERLAY_DIR = r"runs\segment\cv_out\skeletons"
    OUT_DIR = r"Artifacts"
    
    create_comparison_artifacts(IMG_DIR, CV_OVERLAY_DIR, YOLO_OVERLAY_DIR, SKEL_OVERLAY_DIR, OUT_DIR)