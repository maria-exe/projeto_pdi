from pathlib import Path
import cv2
import numpy as np
from utils import rgb_to_ycgcr, is_skin_color
from detection_with_boundary import (
    detect_face_and_eye_region,
    compute_redness,
    binarize_redness,
    remove_skin,
    compute_adaptive_kernel_size,
    apply_closing,
    label_components,
    shape_filter,
    region_growing as region_growing_with_boundary
)
from detection_no_boundary import region_growing as region_growing_no_boundary


def run_pipeline():
    data_dir = Path("data")
    output_dir = Path("output")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    images = ["red_eyes2.jpg", "redeye3.png"]
    
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(str(cascade_path))
    if face_cascade.empty():
        raise IOError(f"Could not load classifier from: {cascade_path}")
        
    print("=" * 60)
    print("RUNNING INTEGRATED TESTS FOR RED EYE DETECTION PIPELINE")
    print("=" * 60)
    
    for filename in images:
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"Error: File not found: {filepath}")
            continue
            
        print(f"\nProcessing image: {filename}")
        img_bgr = cv2.imread(str(filepath))
        if img_bgr is None:
            print(f"  Error: Could not read image.")
            continue
            
        base_name = filepath.stem
        
        Y, Cg_prime, Cr_prime = rgb_to_ycgcr(img_bgr)
        skin_mask = is_skin_color(Cg_prime, Cr_prime)
        
        total_pixels = skin_mask.size
        skin_pixels = np.sum(skin_mask)
        skin_percentage = (skin_pixels / total_pixels) * 100.0
        
        print(f"  [Skin Detection] Skin pixels: {skin_pixels}/{total_pixels} ({skin_percentage:.2f}%)")
        
        def normalize_channel(channel):
            c_min, c_max = channel.min(), channel.max()
            if c_max - c_min > 0:
                return ((channel - c_min) / (c_max - c_min) * 255.0).astype(np.uint8)
            return np.zeros_like(channel, dtype=np.uint8)
            
        cv2.imwrite(str(output_dir / f"{base_name}_Y.png"), normalize_channel(Y))
        cv2.imwrite(str(output_dir / f"{base_name}_Cg_prime.png"), normalize_channel(Cg_prime))
        cv2.imwrite(str(output_dir / f"{base_name}_Cr_prime.png"), normalize_channel(Cr_prime))
        cv2.imwrite(str(output_dir / f"{base_name}_skin_mask.png"), (skin_mask * 255).astype(np.uint8))
        
        faces_info = detect_face_and_eye_region(img_bgr, face_cascade)
        print(f"  [Face Detection] Faces detected: {len(faces_info)}")
        
        img_vis = img_bgr.copy()
        
        for idx, info in enumerate(faces_info):
            x, y, w, h = info['x'], info['y'], info['w'], info['h']
            face_x, face_y, face_w, face_h = info['face']
            
            print(f"    Face #{idx}:")
            print(f"      Face box: x={face_x}, y={face_y}, w={face_w}, h={face_h}")
            print(f"      Eye region (Rf): x={x}, y={y}, w={w}, h={h}")
            
            cv2.rectangle(img_vis, (face_x, face_y), (face_x + face_w, face_y + face_h), (0, 255, 0), 2)
            cv2.rectangle(img_vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            rf_crop = img_bgr[y:y+h, x:x+w]
            cv2.imwrite(str(output_dir / f"{base_name}_face_{idx}_rf_crop.png"), rf_crop)
            
            redness = compute_redness(rf_crop)
            print(f"      Redness stats - Min: {redness.min():.2f}, Max: {redness.max():.2f}, Mean: {redness.mean():.2f}")
            
            redness_vis = normalize_channel(redness)
            cv2.imwrite(str(output_dir / f"{base_name}_face_{idx}_redness.png"), redness_vis)
            
            # Test skin extraction for the crop
            _, cg_crop, cr_crop = rgb_to_ycgcr(rf_crop)
            skin_mask_crop = is_skin_color(cg_crop, cr_crop)
            
            # Test binarize_redness + remove_skin
            redness_mask = binarize_redness(redness)
            skin_removed_mask = remove_skin(redness_mask, skin_mask_crop)
            print(f"      Skin-removed red pixels count: {np.sum(skin_removed_mask)}")
            cv2.imwrite(str(output_dir / f"{base_name}_face_{idx}_skin_removed.png"), (skin_removed_mask * 255).astype(np.uint8))
            
            # Test adaptive closing
            kernel_size = compute_adaptive_kernel_size(face_w)
            closed_mask = apply_closing(skin_removed_mask, kernel_size=kernel_size)
            print(f"      Adaptive kernel size: {kernel_size}")
            print(f"      Closed mask pixels count: {np.sum(closed_mask)}")
            cv2.imwrite(str(output_dir / f"{base_name}_face_{idx}_closed.png"), (closed_mask * 255).astype(np.uint8))
            
            # Test label_components (connectedComponentsWithStats)
            num_labels, label_matrix, stats, centroids = label_components(closed_mask)
            print(f"      Connected components: {num_labels} (including background)")
            if num_labels > 1:
                label_vis = ((label_matrix.astype(np.float32) / (num_labels - 1)) * 255.0).astype(np.uint8)
            else:
                label_vis = np.zeros_like(label_matrix, dtype=np.uint8)
            cv2.imwrite(str(output_dir / f"{base_name}_face_{idx}_labels.png"), label_vis)
            
            # Test shape_filter
            approved_labels = shape_filter(stats, num_labels, info['face'])
            print(f"      Approved components labels: {approved_labels}")
            approved_mask = np.isin(label_matrix, approved_labels)
            cv2.imwrite(str(output_dir / f"{base_name}_face_{idx}_approved.png"), (approved_mask * 255).astype(np.uint8))
            
            # Test region_growing (with boundary)
            expanded_mask_with = region_growing_with_boundary(approved_labels, label_matrix, stats, centroids, redness, rf_crop, info['face'])
            print(f"      Expanded mask (With Boundary) pixels count: {np.sum(expanded_mask_with > 0)}")
            cv2.imwrite(str(output_dir / f"{base_name}_face_{idx}_expanded_with_boundary.png"), expanded_mask_with)
            
            # Test region_growing (without boundary)
            expanded_mask_no = region_growing_no_boundary(approved_labels, label_matrix, stats, centroids, redness, rf_crop, info['face'])
            print(f"      Expanded mask (Without Boundary) pixels count: {np.sum(expanded_mask_no > 0)}")
            cv2.imwrite(str(output_dir / f"{base_name}_face_{idx}_expanded_no_boundary.png"), expanded_mask_no)
            
        cv2.imwrite(str(output_dir / f"{base_name}_vis_detection.png"), img_vis)
        
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    run_pipeline()
