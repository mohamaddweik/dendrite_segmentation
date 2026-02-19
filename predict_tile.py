import os
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO


def list_images(folder: str):
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    return [p for p in Path(folder).iterdir() if p.suffix.lower() in exts]

def should_tile(w, h, threshold=1000):
    return max(w, h) >= threshold

def mask_from_xy(r, H, W):
    mask = np.zeros((H, W), dtype=np.uint8)
    if r.masks is None:
        return mask
    for poly in r.masks.xy:  # list of (N,2) arrays in original pixel coords
        pts = poly.astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def tile_coords(W, H, tile=768, overlap=0.2):
    # stride is tile minus overlap
    stride = int(tile * (1 - overlap))
    stride = max(1, stride)

    xs = list(range(0, max(1, W - tile + 1), stride))
    ys = list(range(0, max(1, H - tile + 1), stride))

    # ensure last tile covers the end
    if xs[-1] != max(0, W - tile):
        xs.append(max(0, W - tile))
    if ys[-1] != max(0, H - tile):
        ys.append(max(0, H - tile))

    coords = []
    for y in ys:
        for x in xs:
            x2 = min(W, x + tile)
            y2 = min(H, y + tile)
            coords.append((x, y, x2, y2))
    return coords


def run_tiled_segmentation(
    model: YOLO,
    img_bgr: np.ndarray,
    tile=768,
    overlap=0.2,
    imgsz=896,
    conf=0.10,
    iou=0.7,
    device=0,
    max_det=120,
):
    H, W = img_bgr.shape[:2]
    coords = tile_coords(W, H, tile=tile, overlap=overlap)

    # Accumulate a "vote" mask: sum of predicted masks across tiles
    acc = np.zeros((H, W), dtype=np.float32)
    cnt = np.zeros((H, W), dtype=np.float32)

    for (x1, y1, x2, y2) in coords:
        tile_img = img_bgr[y1:y2, x1:x2]

        # Predict on this tile
        results = model.predict(
            source=tile_img,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            max_det=max_det,
            verbose=False,
        )
        r = results[0]

        # Make a binary mask for this tile (merge all instances into 1)
        tile_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)

        if r.masks is not None:
            # r.masks.data -> (n, h, w) torch tensor at the tile resolution
            m = r.masks.data.detach().float().cpu().numpy()  # (n, h, w)
            m = (m > 0.5).astype(np.uint8)
            tile_mask = np.clip(m.sum(axis=0), 0, 1).astype(np.uint8)

            # ---- FIX: resize mask to match the original tile size ----
            th, tw = (y2 - y1), (x2 - x1)
            if tile_mask.shape[0] != th or tile_mask.shape[1] != tw:
                tile_mask = cv2.resize(tile_mask, (tw, th), interpolation=cv2.INTER_NEAREST)


        # Accumulate (vote)
        acc[y1:y2, x1:x2] += tile_mask.astype(np.float32)
        cnt[y1:y2, x1:x2] += 1.0

    # Average votes and threshold
    avg = acc / np.maximum(cnt, 1e-6)
    # If a pixel is predicted in >= 30% of overlapping tiles that cover it, keep it
    full_mask = (avg >= 0.50).astype(np.uint8)

    return full_mask


def overlay_mask(img_bgr: np.ndarray, mask: np.ndarray, alpha=0.35):
    # Create a colored overlay (cyan)
    overlay = img_bgr.copy()
    color = np.array([255, 255, 0], dtype=np.uint8)  # BGR: yellow-ish/cyan-ish depending
    overlay[mask == 1] = (overlay[mask == 1] * (1 - alpha) + color * alpha).astype(np.uint8)
    return overlay


def main():
    # ---- EDIT THESE PATHS ----
    weights = r"C:\Users\moham\OneDrive\Desktop\jce\Image_processing\dendrite_segmentation\runs\segment\yolo26_dendrite_tiled_v1\weights\best.pt"
    source_folder = r"C:\Users\moham\OneDrive\Desktop\jce\Image_processing\dendrite_segmentation\dendrite_dataset\images\val"
    out_dir = r"C:\Users\moham\OneDrive\Desktop\jce\Image_processing\dendrite_segmentation\runs\segment\pred_tiled_out_2"
    # --------------------------

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "overlays"), exist_ok=True)

    model = YOLO(weights)

    images = list_images(source_folder)
    if not images:
        print("No images found in:", source_folder)
        return

    for p in images:
        name = p.stem
        print(f"\nProcessing: {p.name}")

        # Read image with OpenCV (keeps it as BGR)
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            # fallback for weird formats
            pil = Image.open(str(p)).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        H, W = img.shape[:2]

        if should_tile(W, H, threshold=1000):
            mask = run_tiled_segmentation(
                model,
                img,
                tile=768,
                overlap=0.20,
                imgsz=896,
                conf=0.15,
                iou=0.80,
                device="cpu",
                max_det=120,
            )
        else:
          results = model.predict(
              source=img,
              imgsz=768,
              conf=0.15,
              iou=0.80,
              device="cpu",
              max_det=120,
              retina_masks=True,
              verbose=False,
          )
          r = results[0]

          mask = np.zeros((H, W), dtype=np.uint8)
          if r.masks is not None:
              m = r.masks.data.detach().float().cpu().numpy()  # (n,h,w)
              m = (m > 0.7).astype(np.uint8)
              mask = np.clip(m.sum(axis=0), 0, 1).astype(np.uint8)

              if mask.shape[:2] != (H, W):
                  mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)


        # Save binary mask (0/255)
        mask_path = os.path.join(out_dir, "masks", f"{name}_mask.png")
        cv2.imwrite(mask_path, (mask * 255))

        # Save overlay for report
        overlay = overlay_mask(img, mask, alpha=0.35)
        overlay_path = os.path.join(out_dir, "overlays", f"{name}_overlay.png")
        cv2.imwrite(overlay_path, overlay)

        print("Saved:", overlay_path)

    print("\nDone. Outputs in:", out_dir)


if __name__ == "__main__":
    main()
