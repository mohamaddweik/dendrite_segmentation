import os
import random
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import yaml


# -----------------------------
# Geometry: polygon clip to rectangle (Sutherland–Hodgman)
# -----------------------------
def _clip_poly_to_edge(poly: np.ndarray, edge_fn):
    """Clip polygon against a single half-plane defined by edge_fn(p)->(inside:bool, intersection(p1,p2))."""
    if poly is None or len(poly) == 0:
        return np.empty((0, 2), dtype=np.float32)

    output = []
    prev = poly[-1]
    prev_in = edge_fn(prev)[0]

    for curr in poly:
        curr_in = edge_fn(curr)[0]
        if curr_in:
            if not prev_in:
                # entering: add intersection
                inter = edge_fn(prev, curr)[1]
                if inter is not None:
                    output.append(inter)
            output.append(curr)
        else:
            if prev_in:
                # leaving: add intersection
                inter = edge_fn(prev, curr)[1]
                if inter is not None:
                    output.append(inter)
        prev, prev_in = curr, curr_in

    if len(output) == 0:
        return np.empty((0, 2), dtype=np.float32)
    return np.array(output, dtype=np.float32)


def clip_polygon_to_rect(poly: np.ndarray, x_min: float, y_min: float, x_max: float, y_max: float) -> np.ndarray:
    """
    Clip polygon (Nx2) to axis-aligned rectangle.
    Returns clipped polygon (Mx2). M may be 0.
    """

    def intersect(p1, p2, x=None, y=None):
        # Line segment p1->p2 intersect with vertical x=const or horizontal y=const
        x1, y1 = p1
        x2, y2 = p2
        if x is not None:
            if x2 == x1:
                return None
            t = (x - x1) / (x2 - x1)
            if 0 <= t <= 1:
                return np.array([x, y1 + t * (y2 - y1)], dtype=np.float32)
            return None
        if y is not None:
            if y2 == y1:
                return None
            t = (y - y1) / (y2 - y1)
            if 0 <= t <= 1:
                return np.array([x1 + t * (x2 - x1), y], dtype=np.float32)
            return None
        return None

    # left: x >= x_min
    def left_edge(p1, p2=None):
        p = p1
        inside = p[0] >= x_min
        inter = None
        if p2 is not None:
            inter = intersect(p1, p2, x=x_min)
        return inside, inter

    # right: x <= x_max
    def right_edge(p1, p2=None):
        p = p1
        inside = p[0] <= x_max
        inter = None
        if p2 is not None:
            inter = intersect(p1, p2, x=x_max)
        return inside, inter

    # top: y >= y_min
    def top_edge(p1, p2=None):
        p = p1
        inside = p[1] >= y_min
        inter = None
        if p2 is not None:
            inter = intersect(p1, p2, y=y_min)
        return inside, inter

    # bottom: y <= y_max
    def bottom_edge(p1, p2=None):
        p = p1
        inside = p[1] <= y_max
        inter = None
        if p2 is not None:
            inter = intersect(p1, p2, y=y_max)
        return inside, inter

    out = poly.astype(np.float32)
    out = _clip_poly_to_edge(out, left_edge)
    out = _clip_poly_to_edge(out, right_edge)
    out = _clip_poly_to_edge(out, top_edge)
    out = _clip_poly_to_edge(out, bottom_edge)

    return out


def polygon_area(poly: np.ndarray) -> float:
    if poly is None or len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


# -----------------------------
# YOLO-seg label parsing/writing
# -----------------------------
def read_yolo_seg_labels(label_path: Path, W: int, H: int) -> List[Tuple[int, np.ndarray]]:
    """
    Each line: cls x1 y1 x2 y2 ... (normalized 0-1)
    Returns list of (cls, poly_pixels Nx2)
    """
    if not label_path.exists():
        return []

    items = []
    for line in label_path.read_text().strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        cls = int(float(parts[0]))
        coords = list(map(float, parts[1:]))

        if len(coords) < 6 or (len(coords) % 2 != 0):
            # not a valid polygon
            continue

        pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] *= W
        pts[:, 1] *= H
        items.append((cls, pts))
    return items


def write_yolo_seg_labels(label_path: Path, polys: List[Tuple[int, np.ndarray]], tile_w: int, tile_h: int):
    lines = []
    for cls, poly in polys:
        # normalize
        p = poly.copy().astype(np.float32)
        p[:, 0] = np.clip(p[:, 0] / tile_w, 0.0, 1.0)
        p[:, 1] = np.clip(p[:, 1] / tile_h, 0.0, 1.0)
        flat = " ".join([f"{v:.6f}" for v in p.reshape(-1)])
        lines.append(f"{cls} {flat}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))


# -----------------------------
# Tiling
# -----------------------------
def tile_coords(W: int, H: int, tile: int, overlap: float):
    stride = max(1, int(tile * (1 - overlap)))
    xs = list(range(0, max(1, W - tile + 1), stride))
    ys = list(range(0, max(1, H - tile + 1), stride))

    if xs[-1] != max(0, W - tile):
        xs.append(max(0, W - tile))
    if ys[-1] != max(0, H - tile):
        ys.append(max(0, H - tile))

    for y in ys:
        for x in xs:
            x2 = min(W, x + tile)
            y2 = min(H, y + tile)
            yield x, y, x2, y2


def process_split(
    src_images_dir: Path,
    src_labels_dir: Path,
    out_images_dir: Path,
    out_labels_dir: Path,
    tile: int,
    overlap: float,
    only_tile_big: bool = True,
    big_threshold: int = 1000,
    keep_empty_prob: float = 0.25,
    min_poly_area_px: float = 15.0,
):
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    images = [p for p in src_images_dir.iterdir() if p.suffix.lower() in exts]

    for img_path in images:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print("Skip unreadable:", img_path)
            continue

        H, W = img.shape[:2]
        label_path = src_labels_dir / (img_path.stem + ".txt")
        labels = read_yolo_seg_labels(label_path, W, H)

        do_tile = True
        if only_tile_big and max(W, H) < big_threshold:
            do_tile = False

        if not do_tile:
            # copy image + label as-is
            out_img = out_images_dir / (img_path.stem + ".png")
            cv2.imwrite(str(out_img), img)
            out_lbl = out_labels_dir / (img_path.stem + ".txt")
            if labels:
                # write back normalized to new png size (same W,H)
                write_yolo_seg_labels(out_lbl, labels, W, H)
            else:
                out_lbl.write_text("")
            continue

        # tile
        for x1, y1, x2, y2 in tile_coords(W, H, tile=tile, overlap=overlap):
            tile_img = img[y1:y2, x1:x2].copy()
            th, tw = tile_img.shape[:2]

            kept_polys = []
            for cls, poly in labels:
                clipped = clip_polygon_to_rect(poly, x1, y1, x2 - 1, y2 - 1)
                if len(clipped) < 3:
                    continue
                if polygon_area(clipped) < min_poly_area_px:
                    continue
                # shift into tile coords
                clipped[:, 0] -= x1
                clipped[:, 1] -= y1
                kept_polys.append((cls, clipped))

            # optionally keep some empty tiles
            if not kept_polys and random.random() > keep_empty_prob:
                continue

            tile_name = f"{img_path.stem}_x{x1}_y{y1}_w{tw}_h{th}"
            out_img = out_images_dir / (tile_name + ".png")
            out_lbl = out_labels_dir / (tile_name + ".txt")

            cv2.imwrite(str(out_img), tile_img)
            if kept_polys:
                write_yolo_seg_labels(out_lbl, kept_polys, tw, th)
            else:
                out_lbl.write_text("")


def make_tiled_dataset(
    src_root: str,
    out_root: str,
    tile: int = 768,
    overlap: float = 0.20,
    only_tile_big: bool = True,
    big_threshold: int = 1000,
    keep_empty_prob: float = 0.25,
):
    src_root = Path(src_root)
    out_root = Path(out_root)

    # read original data.yaml to copy class names
    data_yaml = src_root / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing {data_yaml}")

    cfg = yaml.safe_load(data_yaml.read_text())

    # output dirs
    for split in ["train", "val"]:
        process_split(
            src_images_dir=src_root / "images" / split,
            src_labels_dir=src_root / "labels" / split,
            out_images_dir=out_root / "images" / split,
            out_labels_dir=out_root / "labels" / split,
            tile=tile,
            overlap=overlap,
            only_tile_big=only_tile_big,
            big_threshold=big_threshold,
            keep_empty_prob=keep_empty_prob,
        )

    # write new data.yaml
    new_cfg = dict(cfg)
    new_cfg["path"] = str(out_root.resolve())
    new_cfg["train"] = "images/train"
    new_cfg["val"] = "images/val"

    (out_root / "data.yaml").write_text(yaml.safe_dump(new_cfg, sort_keys=False))
    print("Tiled dataset created at:", out_root)


if __name__ == "__main__":
    random.seed(0)

    make_tiled_dataset(
        src_root="dendrite_dataset",
        out_root="dendrite_dataset_tiled",
        tile=768,
        overlap=0.20,
        only_tile_big=True,
        big_threshold=1000,      # tile only big images
        keep_empty_prob=0.25,    # keep 25% of empty tiles
    )
