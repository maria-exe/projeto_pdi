import cv2
import numpy as np
from utils import rgb_to_ycgcr, is_skin_color

# Detecta rosto com viola-jones e para cada rosto calcula a região candidata dos olhos usando a grade 4×5
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
        # Define as coordenadas da regiao dos olhos
        cw = w / 5.0
        ch = h / 4.0
        
        x_start = int(round(x + cw))
        y_start = int(round(y + ch))
        width = int(round(3.0 * cw))
        height = int(round(ch))
        
        results.append({
            'x': x_start,
            'y': y_start,
            'w': width,
            'h': height,
            'face': (x, y, w, h)
        })
        
    return results

# Implementa a medida de redness de Gaubatz e Ulichne 
def compute_redness(img_rf, kr=1.0):
    B = img_rf[..., 0].astype(np.float32)
    G = img_rf[..., 1].astype(np.float32)
    R = img_rf[..., 2].astype(np.float32)
    
    redness_map = (R ** 2) / (G ** 2 + B ** 2 + kr)
    return redness_map

def binarize_redness(redness_map, tau_r1=1.5):
    return redness_map > tau_r1

# Remove da mascara de redness os pixels de pele
def remove_skin(redness_mask, skin_mask):
    return redness_mask & (~skin_mask)

# Calcula o tamanho do kernel morfologico adaptativo com base na largura do rosto
def compute_adaptive_kernel_size(face_width, factor=0.015):
    k_size = int(round(face_width * factor))
    # Garante que o kernel seja de tamanho impar
    if k_size % 2 == 0:
        k_size += 1
    # Mantem o tamanho minimo do kernel como 3
    return max(3, k_size)

# Preenche buracos e junta pedacos proximos que deveriam ser de apenas uma regiao
def apply_closing(mask, kernel_size=5):
    mask_uint8 = mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)

# Identifica os componentes conexos presentes na mascara
def label_components(mask):
    mask_uint8 = mask.astype(np.uint8)
    num_labels, label_matrix, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=4)
    return num_labels, label_matrix, stats, centroids

# Filtra componentes conexos usando restricoes geometricas
def shape_filter(stats, num_labels, face_bbox):
    W, H = face_bbox[2], face_bbox[3]
    approved_labels = []
    
    for i in range(1, num_labels):
        w_i = stats[i, cv2.CC_STAT_WIDTH]
        h_i = stats[i, cv2.CC_STAT_HEIGHT]
        a_i = stats[i, cv2.CC_STAT_AREA]
        
        # Condicao 1: dimensoes do componente proporcionais ao tamanho do rosto
        cond1 = (W / 50.0 <= w_i <= W / 12.0) and (H / 50.0 <= h_i <= H / 12.0)
        # Condicao 2: razao de aspecto proxima a um circulo ou elipse
        cond2 = 0.5 <= (w_i / h_i) <= 2.0
        # Condicao 3: solidez geometrica minima para evitar formas muito espalhadas
        cond3 = a_i >= (w_i * h_i) / 2.0
        
        if cond1 and cond2 and cond3:
            approved_labels.append(i)
            
    return approved_labels

# Expande a mascara inicial usando regiao baseada em vermelhidao e luminancia
def region_growing(approved_labels, label_matrix, redness_map, img_rf, face_bbox, tau_r1=1.5, tau_l=250.0):
    # Define um limiar de vermelhidao mais tolerante para a expansao
    tau_r2 = tau_r1 * (5.0 / 6.0)
    current_mask = np.isin(label_matrix, approved_labels)
    
    if not np.any(current_mask):
        return np.zeros(label_matrix.shape, dtype=np.uint8)
        
    Y, cg_line, cr_line = rgb_to_ycgcr(img_rf)
    skin_mask = is_skin_color(cg_line, cr_line)
    
    struct_element = np.array([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]], dtype=np.uint8)
                               
    iteration = 0
    while True:
        # Encontra a vizinhanca da mascara atual
        dilated = cv2.dilate(current_mask.astype(np.uint8), struct_element)
        frontier = (dilated > 0) & (~current_mask)
        
        if not np.any(frontier):
            break
            
        cond_base = (redness_map > tau_r2) | (Y > tau_l)
        
        # Evita a invasao em areas de pele apos as primeiras iteracoes
        if iteration < 3:
            admissable = frontier & cond_base
        else:
            admissable = frontier & cond_base & (~skin_mask)
            
        pixels_added = np.sum(admissable)
        if pixels_added > 0:
            current_mask = current_mask | admissable
            
        if iteration >= 10 and pixels_added == 0:
            break
            
        iteration += 1
        
    # Realiza fechamento dos ruidos
    face_width = face_bbox[2]
    kernel_size = compute_adaptive_kernel_size(face_width)
    closed_mask = apply_closing(current_mask, kernel_size=kernel_size)
    
    return (closed_mask * 255).astype(np.uint8)
