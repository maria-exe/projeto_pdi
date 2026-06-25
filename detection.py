import cv2
import numpy as np

def detect_face_and_eye_region(img, face_cascade):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    results = []
    
    for (x, y, w, h) in faces:
        cw = w / 5.0
        ch = h / 4.0
        
        x_start = int(round(x + cw))
        y_start = int(round(y + ch))
        w_rf = int(round(3.0 * cw))
        h_rf = int(round(ch))
        
        results.append({
            'x': x_start,
            'y': y_start,
            'w': w_rf,
            'h': h_rf,
            'face': (x, y, w, h)
        })
        
    return results

def compute_redness(img_rf, kr=1.0):
    B = img_rf[..., 0].astype(np.float32)
    G = img_rf[..., 1].astype(np.float32)
    R = img_rf[..., 2].astype(np.float32)
    
    redness_map = (R ** 2) / (G ** 2 + B ** 2 + kr)
    return redness_map