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
        cx, cy = int(round(cx_mask)), int(round(cy_mask))
        if len(coords) > 0:
            r_pupil = int(round(np.sqrt(len(coords) / np.pi)))
        else:
            r_pupil = int(round(min(H, W) * 0.1))
            
    r_pupil = max(3, r_pupil)
    
    B_grad = cv2.Sobel(img_rf[..., 0], cv2.CV_64F, 1, 0, ksize=3)
    G_grad = cv2.Sobel(img_rf[..., 1], cv2.CV_64F, 1, 0, ksize=3)
    R_grad = cv2.Sobel(img_rf[..., 2], cv2.CV_64F, 1, 0, ksize=3)
    
    g_B = np.abs(B_grad)
    g_G = np.abs(G_grad)
    g_R = np.abs(R_grad)
    
    g_joint = np.minimum(np.minimum(g_R, g_G), g_B)
    
    min_dist = int(round(1.3 * r_pupil))
    max_dist = int(round(2.5 * r_pupil))
    
    margin = max(8, r_pupil // 2)
    y_min = max(0, cy - margin)
    y_max = min(H - 1, cy + margin) + 1
    
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
            
    fallback_reasons = []
    
    if d_left is None:
        d_left = int(round(1.5 * r_pupil))
        fallback_reasons.append("Borda direita nao encontrada")
    elif not (1.3 * r_pupil <= d_left <= 2.5 * r_pupil):
        fallback_reasons.append(f"d_left={d_left} fora de faixa [{1.3 * r_pupil:.1f}, {2.5 * r_pupil:.1f}]")
        d_left = int(round(1.5 * r_pupil))
        
    if d_right is None:
        d_right = int(round(1.5 * r_pupil))
        fallback_reasons.append("Borda esquerda nao encontrada")
    elif not (1.3 * r_pupil <= d_right <= 2.5 * r_pupil):
        fallback_reasons.append(f"d_right={d_right} fora de faixa [{1.3 * r_pupil:.1f}, {2.5 * r_pupil:.1f}]")
        d_right = int(round(1.5 * r_pupil))
        
    if fallback_reasons:
        print(f"[Aviso] Deteccao de esclera falhou, usando valor estimado: {', '.join(fallback_reasons)}")
        
    r_iris = (d_left + d_right) / 2.0
    d_iris = 2.0 * r_iris

    return float(d_iris), (int(cx), int(cy))

def calculate_pupil_size(d_iris, r_pi=0.5507):
    return max(1.0, float(d_iris * r_pi))


def inpaint_exemplar(img_rf, expanded_mask, patch_size=3, search_size=7):
    img_out = img_rf.copy()
    H, W, C = img_out.shape
    M = (expanded_mask > 0).astype(bool)
    
    coords = np.argwhere(M)
    if len(coords) > 0:
        xc_y, xc_x = coords.mean(axis=0)
    else:
        xc_y, xc_x = H / 2.0, W / 2.0
        
    gray = cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    
    hp = patch_size // 2
    hs = search_size // 2
    
    struct = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=np.uint8)
    
    while np.any(M):
        eroded = cv2.erode(M.astype(np.uint8), struct, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        boundary = (M.astype(np.uint8) - eroded) > 0
        
        boundary_coords = np.argwhere(boundary)
        if len(boundary_coords) == 0:
            boundary_coords = np.argwhere(M)
            if len(boundary_coords) == 0:
                break
                
        priorities = 100.0 * np.abs(boundary_coords[:, 1] - xc_x) - np.abs(boundary_coords[:, 0] - xc_y)
        
        best_idx = np.argmax(priorities)
        uy, ux = boundary_coords[best_idx]
        
        best_v = None
        min_dist = float('inf')
        
        r_start = max(0, uy - hs)
        r_end = min(H - 1, uy + hs)
        c_start = max(0, ux - hs)
        c_end = min(W - 1, ux + hs)
        
        for vr in range(r_start, r_end + 1):
            for vc in range(c_start, c_end + 1):
                if M[vr, vc]:
                    continue
                
                sum_diff = 0.0
                count = 0
                
                for dr in range(-hp, hp + 1):
                    for dc in range(-hp, hp + 1):
                        tr, tc = uy + dr, ux + dc
                        sr, sc = vr + dr, vc + dc
                        
                        if 0 <= tr < H and 0 <= tc < W and 0 <= sr < H and 0 <= sc < W:
                            if not M[tr, tc] and not M[sr, sc]:
                                db = float(img_out[tr, tc, 0]) - float(img_out[sr, sc, 0])
                                dg = float(img_out[tr, tc, 1]) - float(img_out[sr, sc, 1])
                                dr_val = float(img_out[tr, tc, 2]) - float(img_out[sr, sc, 2])
                                color_diff_sq = db*db + dg*dg + dr_val*dr_val
                                
                                dgx = gx[tr, tc] - gx[sr, sc]
                                dgy = gy[tr, tc] - gy[sr, sc]
                                grad_diff_sq = dgx*dgx + dgy*dgy
                                
                                sum_diff += color_diff_sq + grad_diff_sq
                                count += 1
                                
                if count > 0:
                    dist = np.sqrt(sum_diff / count)
                    if dist < min_dist:
                        min_dist = dist
                        best_v = (vr, vc)
                        
        if best_v is None:
            outside_coords = np.argwhere(~M)
            
            if len(outside_coords) > 0:
                dists = (outside_coords[:, 0] - uy)**2 + (outside_coords[:, 1] - ux)**2
                fallback_idx = np.argmin(dists)
                best_v = tuple(outside_coords[fallback_idx])
            else:
                break
                
        vr, vc = best_v
        img_out[uy, ux] = img_out[vr, vc]
        gx[uy, ux] = gx[vr, vc]
        gy[uy, ux] = gy[vr, vc]
        
        M[uy, ux] = False
        
    return img_out


def paint_pupil_and_highlight(img_inpainted, center, d_pupil, d_iris=None):
    img_out = img_inpainted.copy()
    H, W, C = img_out.shape
    cx, cy = center
    
    y_indices, x_indices = np.indices((H, W))
    dists_sq = (x_indices - cx)**2 + (y_indices - cy)**2
    
    r_pupil_sq = (d_pupil / 2.0)**2
    pupil_mask = dists_sq < r_pupil_sq
    img_out[pupil_mask] = [27, 27, 27]
    
    d_highlight = d_pupil / 4.0
    r_highlight_sq = (d_highlight / 2.0)**2
    highlight_mask = dists_sq < r_highlight_sq
    img_out[highlight_mask] = [235, 235, 235]
    
    return img_out

# Suaviza os pixels da correcao da pupila
def smooth_boundaries(img_painted, center, d_iris, d_pupil):
    img_out = img_painted.copy()
    H, W, C = img_out.shape
    cx, cy = center
    
    y_indices, x_indices = np.indices((H, W))
    dists_sq = (x_indices - cx)**2 + (y_indices - cy)**2
    
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]], dtype=np.float32) / 16.0
    
    img_blurred = img_out.copy()
    for _ in range(3):
        img_blurred = cv2.filter2D(img_blurred, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    
    r_iris = d_iris / 2.0

    smooth_mask = dists_sq < (r_iris)**1.95
    img_out[smooth_mask] = img_blurred[smooth_mask]
    
    return img_out
