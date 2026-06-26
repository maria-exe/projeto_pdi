import cv2
import numpy as np

def detect_iris(expanded_mask, img_rf, param2=20, tau_g=40.0):
    mask_uint8 = expanded_mask.astype(np.uint8)
    if np.max(mask_uint8) == 1:
        mask_uint8 = mask_uint8 * 255
        
    edges = cv2.Canny(mask_uint8, 100, 200)
    
    H, W = mask_uint8.shape
    min_r = max(3, int(round(min(H, W) * 0.05)))
    max_r = int(round(min(H, W) * 0.3))
    
    coords = np.argwhere(mask_uint8 > 0)
    if len(coords) > 0:
        cy_mask, cx_mask = coords.mean(axis=0)
    else:
        cy_mask, cx_mask = H / 2, W / 2
        
    circles = cv2.HoughCircles(
        edges, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=20, 
        param1=50, 
        param2=param2,
        minRadius=min_r, 
        maxRadius=max_r
    )
    
    if circles is not None:
        circles = circles[0]
        dists = (circles[:, 0] - cx_mask)**2 + (circles[:, 1] - cy_mask)**2
        best_idx = np.argmin(dists)
        cx, cy, r_pupil = np.round(circles[best_idx]).astype(int)
    else:
        if len(coords) > 0:
            cx, cy = int(round(cx_mask)), int(round(cy_mask))
            area = len(coords)
            r_pupil = int(round(np.sqrt(area / np.pi)))
        else:
            cx, cy = W // 2, H // 2
            r_pupil = int(round(min(H, W) * 0.1))
            
    r_pupil = max(3, r_pupil)
    
    B_grad = cv2.Sobel(img_rf[..., 0], cv2.CV_64F, 1, 0, ksize=3)
    G_grad = cv2.Sobel(img_rf[..., 1], cv2.CV_64F, 1, 0, ksize=3)
    R_grad = cv2.Sobel(img_rf[..., 2], cv2.CV_64F, 1, 0, ksize=3)
    
    g_B = np.abs(B_grad)
    g_G = np.abs(G_grad)
    g_R = np.abs(R_grad)
    
    g_joint = np.minimum(np.minimum(g_R, g_G), g_B)
    
    min_dist = int(round(1.1 * r_pupil))
    max_dist = int(round(3.0 * r_pupil))
    
    y_min = max(0, cy - 8)
    y_max = min(H - 1, cy + 8) + 1
    
    profile = np.max(g_joint[y_min:y_max, :], axis=0)
    
    d_left = None
    left_start = max(0, cx - max_dist)
    left_end = max(0, cx - min_dist)
    if left_end > left_start:
        left_profile = profile[left_start : left_end + 1]
        left_masked = np.where(left_profile > tau_g, left_profile, 0.0)
        if np.max(left_masked) > 0.0:
            peak_idx = np.argmax(left_masked)
            d_left = cx - (left_start + peak_idx)
            
    d_right = None
    right_start = min(W - 1, cx + min_dist)
    right_end = min(W - 1, cx + max_dist)
    if right_end > right_start:
        right_profile = profile[right_start : right_end + 1]
        right_masked = np.where(right_profile > tau_g, right_profile, 0.0)
        if np.max(right_masked) > 0.0:
            peak_idx = np.argmax(right_masked)
            d_right = (right_start + peak_idx) - cx
            
    if d_left is None:
        d_left = int(round(2.25 * r_pupil))
    if d_right is None:
        d_right = int(round(2.25 * r_pupil))
        
    if not (2.0 * r_pupil <= d_left * 2 <= 3.0 * r_pupil):
        d_left = int(round(2.25 * r_pupil))
    if not (2.0 * r_pupil <= d_right * 2 <= 3.0 * r_pupil):
        d_right = int(round(2.25 * r_pupil))
        
    r_iris = (d_left + d_right) / 2.0
    d_iris = 2.0 * r_iris
    
    return float(d_iris), (int(cx), int(cy))

def calculate_pupil_size(d_iris, r_pi=0.3316):
    return max(1.0, float(d_iris * r_pi))

