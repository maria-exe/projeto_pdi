import sys, os
import numpy as np, cv2
from pathlib import Path

# Comandos apenas para verificacaoo e criacao de diretorios
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "data"  
OUTPUT_DIR = BASE_DIR / "output"       

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Converte RGB para espaco ycgcr
def rgb_to_ycgcr(img):
    # Usa BGR, padrao da biblioteca cv2    
    B = img[..., 0].astype(np.float32)
    G = img[..., 1].astype(np.float32)
    R = img[..., 2].astype(np.float32)
    
    # Matriz de conversao
    Y = 0.257 * R + 0.504 * G + 0.098 * B + 16.0
    Cg = -0.317 * R + 0.438 * G - 0.121 * B + 128.0
    Cr = 0.439 * R - 0.368 * G - 0.071 * B + 128.0
    
    cos_30 = np.cos(np.radians(30.0))
    sin_30 = np.sin(np.radians(30.0))
    
    Cg_line = Cg * cos_30 + Cr * sin_30 - 48.0
    Cr_line = -Cg * sin_30 + Cr * cos_30 + 80.0
    
    return Y, Cg_line, Cr_line

# Verifica cor de pele
def is_skin_color(cg_line, cr_line):
    return (
        (cg_line >= 125.0) & (cg_line <= 140.0) &
        (cr_line >= 136.0) & (cr_line <= 217.0)
    )

# Lê diretório para encontrar arquivo
def read_image(filename):
    file_path = INPUT_DIR / filename
    path_str = str(file_path)
    
    if not Path(path_str).exists():
        raise FileNotFoundError(f"Erro: imagem nao encontrada: {path_str}")
    
    img = cv2.imread(path_str)
    
    if img is None:
        raise IOError(f"Erro ao processar imagem: {path_str}")
    return img

# Salva imagem na diretorio output
def write_image(filename, img):
    file_path = OUTPUT_DIR / filename
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    success = cv2.imwrite(file_path, img)
    
    if not success:
        raise IOError(f"Nao foi possivel salvar imagem para: {file_path}")
    return True

# Salva as imagens passo a passo do processo de deteccao e remocao do red eye
# Processo: eye candidate region, redness detection, skin color removal, closing operation,labelin e shape filtering
def save_comparison_grid(steps_dict, output_path):
    height, width = None, None
    
    for name, img in steps_dict.items():
        if img is not None:
            height, width = img.shape[:2]
            break
            
    if height is None or width is None:
        raise ValueError("steps_dict nao tem imagens validas")
        
    # Normaliza cada imagem para o formato BGR uint8
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
            

        # Redimensiona, desenha uma faixa no topo de cada imagem e escreve o nome da etapa nela
        img_resized = cv2.resize(img_bgr, (width, height), interpolation=cv2.INTER_NEAREST)
        
        cv2.rectangle(img_resized, (2, 2), (width - 2, 20), (0, 0, 0), -1)
        cv2.putText(img_resized, name, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        processed_imgs.append(img_resized)
        
    # Salva a imagem de comparacao final
    grid = cv2.hconcat(processed_imgs)
    write_image(output_path, grid)
    return True


def evaluate_metrics(ground_truth, detections, tolerance=10.0):
    # Se a foto nao tem olhos vermelhos, retorna alarme falso
    if len(ground_truth) == 0:
        detection_rate = 0.0
        false_alarms = len(detections)
        return detection_rate, false_alarms
        
    gt_matched = [False] * len(ground_truth)
    det_matched = [False] * len(detections)
    
    # Testa todas as combinacoes possiveis de pares de olhos vermelhos e escolhe o par com menor distancia
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
    false_alarms = len(detections) - sum(det_matched) # Deteccoes que sobraram sem par
    
    return detection_rate, false_alarms
