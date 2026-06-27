import cv2
import numpy as np
from pathlib import Path
from utils import (
    rgb_to_ycgcr, is_skin_color,
    safe_read_image, safe_write_image,
    save_comparison_grid
)
from detection import (
    detect_face_and_eye_region,
    compute_redness, binarize_redness, remove_skin,
    compute_adaptive_kernel_size, apply_closing,
    label_components, shape_filter, region_growing
)
from correction import (
    detect_iris, calculate_pupil_size,
    inpaint_exemplar, paint_pupil_and_highlight, smooth_boundaries
)

DATA_DIR   = Path("data")
OUTPUT_DIR = Path("output")
IMAGES     = ["teste1.png", "teste2.jpg"]


def normalize_channel(channel):
    c_min, c_max = channel.min(), channel.max()
    if c_max - c_min > 0:
        return ((channel - c_min) / (c_max - c_min) * 255.0).astype(np.uint8)
    return np.zeros_like(channel, dtype=np.uint8)


def process_image(img_bgr, base_name, face_cascade):
    img_corrected = img_bgr.copy()
    img_vis       = img_bgr.copy()

    # --- Skin detection (full image, for reference) ---
    _, Cg_prime, Cr_prime = rgb_to_ycgcr(img_bgr)
    skin_mask = is_skin_color(Cg_prime, Cr_prime)
    safe_write_image(OUTPUT_DIR / f"{base_name}_skin_mask.png",
                     (skin_mask * 255).astype(np.uint8))

    # --- Face + eye region detection ---
    faces_info = detect_face_and_eye_region(img_bgr, face_cascade)
    print(f"  Faces detected: {len(faces_info)}")

    for idx, info in enumerate(faces_info):
        x, y, w, h       = info['x'], info['y'], info['w'], info['h']
        face_x, face_y, face_w, face_h = info['face']

        cv2.rectangle(img_vis, (face_x, face_y),
                      (face_x + face_w, face_y + face_h), (0, 255, 0), 2)
        cv2.rectangle(img_vis, (x, y), (x + w, y + h), (0, 0, 255), 2)

        rf_crop = img_bgr[y:y+h, x:x+w]

        # --- Detection pipeline ---
        redness          = compute_redness(rf_crop)
        _, cg_crop, cr_crop = rgb_to_ycgcr(rf_crop)
        skin_mask_crop   = is_skin_color(cg_crop, cr_crop)
        redness_mask     = binarize_redness(redness)
        skin_removed     = remove_skin(redness_mask, skin_mask_crop)
        kernel_size      = compute_adaptive_kernel_size(face_w)
        closed_mask      = apply_closing(skin_removed, kernel_size=kernel_size)
        num_labels, label_matrix, stats, centroids = label_components(closed_mask)
        approved_labels  = shape_filter(stats, num_labels, info['face'])

        print(f"  Face #{idx} — approved components: {approved_labels}")

        # Save intermediate steps grid
        save_comparison_grid({
            "Original":     rf_crop,
            "Redness":      redness,
            "Skin Removed": skin_removed,
            "Closing":      closed_mask,
            "Labels":       label_matrix,
            "Shape Filter": np.isin(label_matrix, approved_labels),
        }, OUTPUT_DIR / f"{base_name}_face_{idx}_steps.png")

        if not approved_labels:
            continue

        # --- Region growing (once per label, result reused) ---
        expanded_masks = {}
        combined_mask  = np.zeros(rf_crop.shape[:2], dtype=np.uint8)
        iris_info      = {}

        for label in approved_labels:
            exp_mask = region_growing(
                [label], label_matrix, stats, centroids,
                redness, rf_crop, info['face']
            )
            expanded_masks[label] = exp_mask
            combined_mask = cv2.bitwise_or(combined_mask, exp_mask)

            d_iris, center = detect_iris(exp_mask, rf_crop)
            d_pupil        = calculate_pupil_size(d_iris)
            iris_info[label] = (d_iris, center, d_pupil)

            print(f"    Label #{label} — d_iris: {d_iris:.1f}px  "
                  f"d_pupil: {d_pupil:.1f}px  center: {center}")

            cx_full = center[0] + x
            cy_full = center[1] + y
            cv2.circle(img_vis, (cx_full, cy_full),
                       int(round(d_iris / 2)), (255, 0, 0), 2)

        # --- Correction pipeline ---
        corrected_crop = inpaint_exemplar(rf_crop, combined_mask)

        for label in approved_labels:
            d_iris, center, d_pupil = iris_info[label]
            corrected_crop = paint_pupil_and_highlight(
                corrected_crop, center, d_pupil)
            corrected_crop = smooth_boundaries(
                corrected_crop, center, d_iris, d_pupil)

        safe_write_image(
            OUTPUT_DIR / f"{base_name}_face_{idx}_corrected_crop.png",
            corrected_crop)
        img_corrected[y:y+h, x:x+w] = corrected_crop

    safe_write_image(OUTPUT_DIR / f"{base_name}_vis.png",      img_vis)
    safe_write_image(OUTPUT_DIR / f"{base_name}_corrected.png", img_corrected)


def run_pipeline():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(str(cascade_path))
    if face_cascade.empty():
        raise IOError(f"Could not load classifier from: {cascade_path}")

    for filename in IMAGES:
        filepath = DATA_DIR / filename
        print(f"\nProcessing: {filename}")
        try:
            img_bgr = safe_read_image(filepath)
        except Exception as e:
            print(f"  Error: {e}")
            continue

        process_image(img_bgr, filepath.stem, face_cascade)

    print("\nDone.")


if __name__ == "__main__":
    run_pipeline()