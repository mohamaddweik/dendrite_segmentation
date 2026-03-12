import math
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os
import cv2
from skimage.morphology import skeletonize
from skimage.filters import threshold_sauvola

class SEMDendriteSegmenter:
    def __init__(self, bottom_overlay_ratio: float = 0.18):
        """
        crop the bottom of the image to remove the overlay,
        which contains the scale bar and other information.
        """
        self.bottom_overlay_ratio = bottom_overlay_ratio
    
    def load_image(self, path: str) -> np.ndarray:
        """
        Loads image in grayscale.
        """
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError(f"Could not load image from {path}")

        return img
    
    def estimate_overlay_cut(self,img_gray: np.ndarray, search_bottom_frac: float = 0.35) -> int:
        """
        Returns how many pixels to cut from the bottom.
        only search within the bottom part of the image (e.g., bottom 35%).
        """

        height, width = img_gray.shape
        start = int(height * (1 - search_bottom_frac))
        start = max(0, min(start, height - 2))

        # Row means in the bottom region
        row_mean = img_gray[start:, :].mean(axis=1)

        # Find a strong "drop" point: where mean goes down fast
        diff = np.diff(row_mean)
        idx = int(np.argmin(diff))  # strongest negative jump

        # Convert to full-image coordinates
        overlay_start_row = start + idx

        cut_pixels = height - overlay_start_row
        return max(0, min(cut_pixels, height - 1))

    def enhance_contrast_clahe(self, img_gray: np.ndarray,
                           clip_limit: float = 2.0,
                           tile_grid_size: int = 8) -> np.ndarray:
        """
        Histogram normalization using CLAHE.
        CLAHE (Contrast Limited Adaptive Histogram Equalization) is an advanced method that divides the image into small tiles,
        applies histogram equalization to each tile, and then combines the results.
        """
        if img_gray.dtype != np.uint8:
            img_gray = np.clip(img_gray, 0, 255).astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
        out = clahe.apply(img_gray)
        return out

    def denoise_bilateral(self, img_gray: np.ndarray,
                      d: int = 7,
                      sigma_color: float = 35,
                      sigma_space: float = 7) -> np.ndarray:
        """
        Edge-preserving denoising using bilateral filter.

        Why we use it here:
        - Your SEM background has grain/texture that adaptive threshold turns into foreground.
        - Bilateral reduces that texture BUT keeps sharp dendrite edges.

        Parameters:
        - d: neighborhood diameter (bigger = stronger smoothing, slower). Start 7.
        - sigma_color: how different intensities can still be smoothed together.
                      Bigger = more smoothing (but can wash small details). Start 35.
        - sigma_space: how far (in pixels) smoothing reaches spatially.
                      Bigger = smoother background. Start 7.
        """
        if img_gray.dtype != np.uint8:
            img_gray = np.clip(img_gray, 0, 255).astype(np.uint8)

        return cv2.bilateralFilter(img_gray, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    def threshold_sauvola_mask(self, img_gray: np.ndarray, window_size=75, k=0.25) -> np.ndarray:
            # Normalize to [0, 1]
            img = img_gray.astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

            # TRICK: Invert the image so it behaves like dark text on a bright page
            img_inv = 1.0 - img

            # Run Sauvola on the inverted image
            thr = threshold_sauvola(img_inv, window_size=window_size, k=k)

            # Grab the "dark" pixels from the inverted image (these are your bright dendrites)
            mask = (img_inv < thr).astype(np.uint8) * 255

            return mask

    def reconstruct_by_dilation(self, marker: np.ndarray, mask: np.ndarray,
                            kernel_size: int = 3,
                            max_iters: int = 500) -> np.ndarray:
        """
        Binary morphological reconstruction by dilation.

        marker: starting image (usually eroded mask)
        mask: constraint image (original mask). Reconstruction can only grow inside mask.

        kernel_size: dilation kernel size (3 is standard)
        max_iters: safety cap so it can't loop forever

        Returns: reconstructed binary mask (0/255 uint8)
        """
        # ensure binary 0/255
        marker = (marker > 0).astype(np.uint8) * 255
        mask = (mask > 0).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        prev = marker
        for _ in range(max_iters):
            dil = cv2.dilate(prev, kernel, iterations=1)
            # constrain inside mask
            new = cv2.bitwise_and(dil, mask)

            if np.array_equal(new, prev):
                break
            prev = new

        return prev

    def filter_connected_components(self, mask: np.ndarray, min_area: int = 50) -> np.ndarray:
        """
        Remove small connected components (noise blobs) from a binary mask.

        min_area:
        - any component with area < min_area pixels is removed.
        """
        mask_bin = (mask > 0).astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
        out = np.zeros_like(mask_bin)

        for lbl in range(1, num_labels):  # skip background (0)
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area >= min_area:
                out[labels == lbl] = 1

        return (out * 255).astype(np.uint8)

    def filter_by_strong_overlap(self, relaxed_mask: np.ndarray,
                             strict_mask: np.ndarray) -> np.ndarray:
        """
        Keep only components in relaxed_mask that overlap with strict_mask.
        """
        r = (relaxed_mask > 0).astype(np.uint8)
        s = (strict_mask > 0).astype(np.uint8)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(r, connectivity=8)
        out = np.zeros_like(r)

        for lbl in range(1, num):
            component = (labels == lbl).astype(np.uint8)

            # if any overlap with strict mask -> keep entire component
            if (component & s).any():
                out[component == 1] = 1

        return (out * 255).astype(np.uint8)

    def morph_close(self, mask: np.ndarray, k: int = 3, iters: int = 1) -> np.ndarray:
        """
        Light morphological closing: fills tiny gaps and small holes.

        Use small kernel/iters so we don't merge dendrites with artifacts.
        """
        m = (mask > 0).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=iters)

    def skeletonize_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Convert a binary dendrite mask into a 1-pixel-wide skeleton.

        Why we use it:
        - Makes it easy to measure length, branches, endpoints.
        - Removes thickness, keeps topology (tree structure).

        Input:  0/255 uint8 mask
        Output: 0/255 uint8 skeleton
        """

        m = (mask > 0)  # boolean
        skel = skeletonize(m)  # boolean
        return (skel.astype(np.uint8) * 255)

    def remove_wide_horizontal_bands_by_row_projection(
            self,
            mask: np.ndarray,
            min_row_frac: float = 0.20,
            blocks: int = 16,
            min_active_blocks_frac: float = 0.75,
            max_band_height: int = 8,
            band_pad: int = 1
        ) -> np.ndarray:
            """
            Remove thin horizontal bands anywhere in the image using row projection.

            A row is considered part of a band if:
            - enough of the row is white (min_row_frac)
            - white pixels are spread across the width (min_active_blocks_frac)

            Then consecutive candidate rows are grouped into bands, and only
            thin bands (<= max_band_height) are removed.
            """
            m = (mask > 0).astype(np.uint8)
            H, W = m.shape
            out = m.copy()

            block_w = max(1, W // blocks)
            candidate = np.zeros(H, dtype=bool)

            for y in range(H):
                row = m[y, :]
                frac = row.mean()

                if frac < min_row_frac:
                    continue

                active_blocks = 0
                for b in range(blocks):
                    x0 = b * block_w
                    x1 = W if b == blocks - 1 else (b + 1) * block_w
                    if row[x0:x1].any():
                        active_blocks += 1

                if (active_blocks / blocks) >= min_active_blocks_frac:
                    candidate[y] = True

            # group consecutive candidate rows into bands
            y = 0
            while y < H:
                if not candidate[y]:
                    y += 1
                    continue

                y0 = y
                while y < H and candidate[y]:
                    y += 1
                y1 = y - 1

                band_height = y1 - y0 + 1

                if band_height <= max_band_height:
                    r0 = max(0, y0 - band_pad)
                    r1 = min(H, y1 + band_pad + 1)
                    out[r0:r1, :] = 0

            return (out * 255).astype(np.uint8)

    def filter_skeleton_by_mask_thickness(self,
                                      mask: np.ndarray,
                                      skel: np.ndarray,
                                      min_dist: float = 1.4,
                                      dilate_iters: int = 200) -> np.ndarray:
        """
        Remove skeleton fragments that come from very thin/noisy mask regions.

        Steps:
        1) distance transform on the binary mask -> thickness estimate
        2) seed = skeleton pixels where thickness >= min_dist
        3) geodesic dilation: grow seed only along the skeleton (keeps whole fragments
          that contain at least one 'thick' pixel)

        Works well when noise is made of tiny blobs but dendrites have some thicker cores.
        """
        m = (mask > 0).astype(np.uint8)
        s = (skel > 0).astype(np.uint8)

        # distance to nearest background pixel (only meaningful inside mask)
        dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)

        # seeds: "trusted" skeleton pixels that lie in thicker parts of the mask
        seed = ((s == 1) & (dist >= float(min_dist))).astype(np.uint8)

        # geodesic dilation constrained to skeleton:
        # keep growing seed, but only where skeleton exists
        kernel = np.ones((3, 3), np.uint8)
        prev = np.zeros_like(seed)

        it = 0
        while it < dilate_iters and not np.array_equal(seed, prev):
            prev = seed
            seed = cv2.dilate(seed, kernel, iterations=1)
            seed = (seed & s).astype(np.uint8)
            it += 1

        return (seed * 255).astype(np.uint8)

    def wipe_noisy_borders(self, mask: np.ndarray, border_frac: float = 0.20, blocks: int = 10, min_active_blocks: int = 8) -> np.ndarray:
        """
        Scans top and bottom edges. If a row has white pixels spread across 
        almost the entire width (e.g., 8 out of 10 blocks), it's treated as an artifact band and wiped.
        """
        out = mask.copy()
        H, W = out.shape
        margin = int(H * border_frac)
        block_w = max(1, W // blocks)

        def is_noisy_row(y):
            row = out[y, :]
            active = 0
            for b in range(blocks):
                x0 = b * block_w
                x1 = W if b == blocks - 1 else (b + 1) * block_w
                # If there is any white pixel in this block, count it as active
                if row[x0:x1].any():
                    active += 1
            return active >= min_active_blocks

        # 1. Clean the Top: Scan from the margin upwards
        for y in range(margin, -1, -1): 
            if is_noisy_row(y):
                out[:y+1, :] = 0  # Black out the noisy band and everything above it
                break

        # 2. Clean the Bottom: Scan from the margin downwards
        for y in range(H - margin, H): 
            if is_noisy_row(y):
                out[y:, :] = 0    # Black out the noisy band and everything below it
                break

        return out

    def overlay_mask(self, img_gray: np.ndarray, mask: np.ndarray, color=(0,0,255), alpha=0.5):
        """
        Overlay binary mask on grayscale image.
        color is BGR (OpenCV format).
        """
        img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

        overlay = img_color.copy()
        overlay[mask > 0] = color

        blended = cv2.addWeighted(overlay, alpha, img_color, 1 - alpha, 0)
        return blended

    def overlay_skeleton(self, img_gray: np.ndarray, skel: np.ndarray, color=(0,255,255)):
        img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        img_color[skel > 0] = color
        return img_color

    def run_stages(self, img: np.ndarray):
        stages = {}

        # 1) Cut the image into two physical pieces
        cut = self.estimate_overlay_cut(img)
        if cut > 0:
            base = img[:-cut, :]
            bottom_strip = img[-cut:, :] # Save the chopped text/scale bar for later!
        else:
            base = img.copy()
            bottom_strip = None
        #stages["1_overlay_removed"] = base

        # 2) Histogram normalization
        norm = cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        #stages["2_normalized"] = norm

        # 3) CLAHE
        clahe = self.enhance_contrast_clahe(norm, clip_limit=1.0, tile_grid_size=12)
        #stages["3_clahe"] = clahe

        # 4) Bilateral denoising
        den = self.denoise_bilateral(clahe, d=9, sigma_color=80, sigma_space=9)
        den = cv2.medianBlur(den, 3)
        #stages["4_denoised"] = den

        # 5) Dual Sauvola masks
        relaxed = self.threshold_sauvola_mask(den, window_size=21, k=0.04)
        relaxed = self.filter_connected_components(relaxed, min_area=4)
        #stages["5a_relaxed"] = relaxed

        strict = self.threshold_sauvola_mask(den, window_size=21, k=0.15)
        strict = self.filter_connected_components(strict, min_area=6)
        #stages["5b_strict"] = strict

        mask = self.filter_by_strong_overlap(relaxed, strict)
        #stages["5c_supported"] = mask

        # 6) Morphological reconstruction
        marker = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
        recon = self.reconstruct_by_dilation(marker, mask, kernel_size=3)
        #stages["6_reconstruction"] = recon

        # 7) Light closing
        #closed = self.morph_close(recon, k=3, iters=1)
        #stages["7_closed"] = closed

        # 8) Wipe noisy borders
        wiped = self.wipe_noisy_borders(recon, border_frac=0.08, blocks=16, min_active_blocks=15)
        #stages["8_borders_wiped"] = wiped

        # 9) Connected components cleanup
        clean = self.filter_connected_components(wiped, min_area=12)
        #stages["9_cc_filtered"] = clean

        # 10) Edge lines removed
        clean = self.remove_wide_horizontal_bands_by_row_projection(
            clean,
            min_row_frac=0.6,
            blocks=16,
            min_active_blocks_frac=0.7,
            max_band_height=80,
            band_pad=1
        )
        #stages["11_edge_lines_removed"] = clean

        # 12) Final cleanup mask (cropped)
        clean = self.filter_connected_components(clean, min_area=16)
        #stages["12_final_mask_cropped"] = clean

        # 13) Skeleton (cropped)
        skel = self.skeletonize_mask(clean)
        skel = self.filter_skeleton_by_mask_thickness(
            clean, skel, min_dist=1.2, dilate_iters=200
        )
        #stages["13_skeleton_cropped"] = skel

        # 14) Overlays (cropped)
        colored_base = self.overlay_mask(base, clean, color=(0, 0, 255), alpha=0.4)
        #stages["14_overlay_cropped"] = colored_base
        
        colored_base_skel = self.overlay_skeleton(base, skel, color=(0, 255, 0))
        #stages["15_skeleton_overlay_cropped"] = colored_base_skel

        # --- 16) GLUE EVERYTHING BACK TOGETHER ---
        if bottom_strip is not None:
            # Convert the saved bottom strip to BGR so the colors match
            bottom_bgr = cv2.cvtColor(bottom_strip, cv2.COLOR_GRAY2BGR)
            
            # Glue them vertically!
            final_visual = np.vstack((colored_base, bottom_bgr))
            final_visual_skel = np.vstack((colored_base_skel, bottom_bgr))
            
            # And for the evaluation script, pad the math mask with black
            clean_full = cv2.copyMakeBorder(clean, 0, cut, 0, 0, cv2.BORDER_CONSTANT, value=0)
            skel_full = cv2.copyMakeBorder(skel, 0, cut, 0, 0, cv2.BORDER_CONSTANT, value=0)
        else:
            final_visual = colored_base
            final_visual_skel = colored_base_skel
            clean_full = clean
            skel_full = skel

        # Save the FINAL full-size images 
        stages["12_final_mask"] = clean_full
        #stages["13_skeleton"] = skel_full
        stages["14_overlay_mask"] = final_visual
        stages["15_overlay_skeleton"] = final_visual_skel

        return stages

    def show_stages(self, stages: dict, save: bool = False, filename: str = "stages.png"):
            """
            Display stage images in a larger readable grid.
            - Bigger figure
            - Larger titles
            - Better spacing
            - Uses grayscale for 2D images and RGB/BGR-aware display for color overlays
            """

            keys = list(stages.keys())
            n = len(keys)

            cols = 2
            rows = int(np.ceil(n / cols))

            # much larger figure so each image is readable
            fig, axes = plt.subplots(rows, cols, figsize=(10 * cols, 7 * rows))
            axes = np.array(axes).reshape(-1)

            for i, k in enumerate(keys):
                ax = axes[i]
                img = stages[k]

                if img.ndim == 2:
                    ax.imshow(img, cmap="gray")
                elif img.ndim == 3 and img.shape[2] == 3:
                    # OpenCV images are usually BGR, matplotlib expects RGB
                    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                else:
                    ax.imshow(img, cmap="gray")

                ax.set_title(k, fontsize=10, pad=10)
                ax.axis("off")

            # hide unused subplots
            for j in range(n, len(axes)):
                axes[j].axis("off")

            plt.subplots_adjust(wspace=0.08, hspace=0.18)

            if save:
                os.makedirs("debug_outputs", exist_ok=True)
                save_path = os.path.join("debug_outputs", filename)
                plt.savefig(save_path, dpi=250, bbox_inches="tight")
                print(f"[DEBUG] saved comparison to {save_path}")

            plt.show()
    
    def plot_all_overlays(self, segmenter, folder_path="dendrite_dataset/images/train/"):
            # Find all images in the folder (supports png, jpg, tif)
            folder = Path(folder_path)
            image_paths = [p for p in folder.iterdir() if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif']]
            
            n_images = len(image_paths)
            if n_images == 0:
                print("No images found in that folder!")
                return

            # Set up a grid (4 columns wide)
            cols = 4
            rows = math.ceil(n_images / cols)
            
            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
            
            # Flatten axes array so it's easy to loop through, even if it's just 1 row
            if n_images > 1:
                axes = axes.flatten()
            else:
                axes = [axes]

            print(f"Processing {n_images} images...")

            for i, path in enumerate(image_paths):
                # 1. Load and process
                img = segmenter.load_image(str(path))
                stages = segmenter.run_stages(img)
                overlay = stages["14_overlay_mask"]
                
                # 2. Convert OpenCV's BGR format to Matplotlib's RGB format
                overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                
                # 3. Plot
                ax = axes[i]
                ax.imshow(overlay_rgb)
                ax.set_title(path.name, fontsize=10)
                ax.axis("off")

            # Hide any empty subplots if the grid isn't completely full
            for j in range(n_images, len(axes)):
                axes[j].axis("off")

            plt.tight_layout()
            plt.show()

    def save_all_results(self, input_folder="dendrite_dataset/images/val", output_folder="runs/cv_out"):
        """
        Runs the CV pipeline on a folder of images and saves the masks and overlays.
        """
        # Create the output folders if they don't exist
        os.makedirs(os.path.join(output_folder, "masks"), exist_ok=True)
        os.makedirs(os.path.join(output_folder, "overlays"), exist_ok=True)
        os.makedirs(os.path.join(output_folder, "skeletons"), exist_ok=True)

        folder = Path(input_folder)
        image_paths = [p for p in folder.iterdir() if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif']]
        
        print(f"Processing and saving {len(image_paths)} images...")
        
        for path in image_paths:
            # 1. Load and process
            img = self.load_image(str(path))
            stages = self.run_stages(img)
            
            # 2. Grab the final mask and overlay
            # Make sure these keys match what you have in run_stages!
            mask = stages["12_final_mask"] 
            overlay = stages["14_overlay_mask"]
            skeleton = stages["15_overlay_skeleton"]
            
            # 3. Save them with the same name format YOLO uses
            mask_path = os.path.join(output_folder, "masks", f"{path.stem}_mask.png")
            cv2.imwrite(mask_path, mask)
            
            overlay_path = os.path.join(output_folder, "overlays", f"{path.stem}_overlay.png")
            cv2.imwrite(overlay_path, overlay)
            skeleton_path = os.path.join(output_folder, "skeletons", f"{path.stem}_skeleton.png")
            cv2.imwrite(skeleton_path, skeleton)

segmenter = SEMDendriteSegmenter()

#img = segmenter.load_image("dendrite_dataset/images/train/70nm_R_50nm_pitch_ETD_012.png")

#stages = segmenter.run_stages(img)
#segmenter.show_stages(stages, save=False, filename="debug_stages.png")

#segmenter.plot_all_overlays(segmenter, folder_path="dendrite_dataset/images/val/")  

segmenter.save_all_results(
    input_folder=r"dendrite_dataset\images\test", 
    output_folder=r"runs\segment\cv_out"
)


