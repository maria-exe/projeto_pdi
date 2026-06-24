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

def is_skin_color(ycgcr):
    _, Cg_prime, Cr_prime = ycgcr
    return (
        (Cg_prime >= 125.0) & (Cg_prime <= 140.0) &
        (Cr_prime >= 136.0) & (Cr_prime <= 217.0)
    )
