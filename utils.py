import numpy as np

def rgb_to_ycgcr(img):
    B = img[..., 0].astype(np.float32)
    G = img[..., 1].astype(np.float32)
    R = img[..., 2].astype(np.float32)
    
    Y = 0.257 * R + 0.504 * G + 0.098 * B + 16.0
    Cg = -0.317 * R + 0.438 * G - 0.121 * B + 128.0
    Cr = 0.439 * R - 0.368 * G - 0.071 * B + 128.0
    
    cos_30 = np.cos(np.radians(30.0))
    sin_30 = np.sin(np.radians(30.0))
    
    Cg_prime = Cg * cos_30 + Cr * sin_30 - 48.0
    Cr_prime = -Cg * sin_30 + Cr * cos_30 + 80.0
    
    return Y, Cg_prime, Cr_prime

def is_skin_color(cg_prime, cr_prime):
    return (
        (cg_prime >= 125.0) & (cg_prime <= 140.0) &
        (cr_prime >= 136.0) & (cr_prime <= 217.0)
    )


def safe_read_image(filepath):
    import cv2
    from pathlib import Path
    path_str = str(filepath)
    if not Path(path_str).exists():
        raise FileNotFoundError(f"Image file not found: {path_str}")
    img = cv2.imread(path_str)
    if img is None:
        raise IOError(f"Could not read image file (corrupted or unsupported format): {path_str}")
    return img


def safe_write_image(filepath, img):
    import cv2
    from pathlib import Path
    path_str = str(filepath)
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(path_str, img)
    if not success:
        raise IOError(f"Could not write image to: {path_str}")
    return True


def save_comparison_grid(steps_dict, output_path):
    import cv2
    import numpy as np
    
    ref_h, ref_w = None, None
    for name, img in steps_dict.items():
        if img is not None:
            ref_h, ref_w = img.shape[:2]
            break
            
    if ref_h is None or ref_w is None:
        raise ValueError("steps_dict contains no valid images")
        
    processed_imgs = []
    for name, img in steps_dict.items():
        if img is None:
            continue
            
        if len(img.shape) == 2:
            if img.dtype == bool:
                img_8u = (img * 255).astype(np.uint8)
            elif np.issubdtype(img.dtype, np.floating):
                c_min, c_max = img.min(), img.max()
                if c_max - c_min > 0:
                    img_8u = ((img - c_min) / (c_max - c_min) * 255.0).astype(np.uint8)
                else:
                    img_8u = np.zeros_like(img, dtype=np.uint8)
            elif np.issubdtype(img.dtype, np.integer):
                c_max = img.max()
                if c_max > 0:
                    img_8u = ((img.astype(np.float32) / c_max) * 255.0).astype(np.uint8)
                else:
                    img_8u = np.zeros_like(img, dtype=np.uint8)
            else:
                img_8u = img.astype(np.uint8)
                
            img_bgr = cv2.cvtColor(img_8u, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = img.copy()
            
        img_resized = cv2.resize(img_bgr, (ref_w, ref_h), interpolation=cv2.INTER_NEAREST)
        
        cv2.rectangle(img_resized, (2, 2), (ref_w - 2, 20), (0, 0, 0), -1)
        cv2.putText(img_resized, name, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        processed_imgs.append(img_resized)
        
    grid = cv2.hconcat(processed_imgs)
    safe_write_image(output_path, grid)
    return True


def evaluate_metrics(ground_truth, detections, tolerance=10.0):
    if len(ground_truth) == 0:
        detection_rate = 0.0
        false_alarms = len(detections)
        return detection_rate, false_alarms
        
    gt_matched = [False] * len(ground_truth)
    det_matched = [False] * len(detections)
    
    while True:
        min_dist = float('inf')
        best_gt_idx = -1
        best_det_idx = -1
        
        for i, gt in enumerate(ground_truth):
            if gt_matched[i]:
                continue
            for j, det in enumerate(detections):
                if det_matched[j]:
                    continue
                dist = np.sqrt((gt[0] - det[0])**2 + (gt[1] - det[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    best_gt_idx = i
                    best_det_idx = j
                    
        if min_dist <= tolerance:
            gt_matched[best_gt_idx] = True
            det_matched[best_det_idx] = True
        else:
            break
            
    correctly_detected = sum(gt_matched)
    detection_rate = float(correctly_detected) / len(ground_truth)
    false_alarms = len(detections) - sum(det_matched)
    
    return detection_rate, false_alarms
