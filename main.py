import cv2
import numpy as np
from pathlib import Path
from utils import (
    rgb_to_ycgcr, is_skin_color,
    read_image, write_image,
    save_comparison_grid,

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

IMAGES = ["teste1.png", "teste2.jpg", "teste3.jpg", "teste4.jpg", "teste5.jpg", "teste6.jpg", "teste7.jpg", "teste8.jpg", "teste9.jpg", "teste10.jpg", 
          "teste11.jpg", "teste12.jpg", "teste13.jpg", "teste14.jpg", "teste15.png"]

def process_image(img_bgr, base_name, face_cascade):
    img_corrected = img_bgr.copy()
    img_vis       = img_bgr.copy()

    # Detecta tons de pele na imagem completa
    _, Cg_line, Cr_line = rgb_to_ycgcr(img_bgr)
    skin_mask = is_skin_color(Cg_line, Cr_line)
    write_image(f"{base_name}_mascara.png",
                     (skin_mask * 255).astype(np.uint8))

    faces_info = detect_face_and_eye_region(img_bgr, face_cascade)
    print(f"Rostos detectados: {len(faces_info)}")

    for idx, info in enumerate(faces_info):
        x, y, w, h       = info['x'], info['y'], info['w'], info['h']
        face_x, face_y, face_w, face_h = info['face']

        # Retangulo verde nos rostos e vermelho nas regioes dos olhos
        cv2.rectangle(img_vis, (face_x, face_y),
                      (face_x + face_w, face_y + face_h), (0, 255, 0), 2)
        cv2.rectangle(img_vis, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Recorta a regiao dos olhos detectada
        rf_crop = img_bgr[y:y+h, x:x+w]

        # Executa a binarizacao e remocao de pele na regiao recortada
        redness          = compute_redness(rf_crop)
        _, cg_crop, cr_crop = rgb_to_ycgcr(rf_crop)
        skin_mask_crop   = is_skin_color(cg_crop, cr_crop)
        redness_mask     = binarize_redness(redness)
        skin_removed     = remove_skin(redness_mask, skin_mask_crop)
        
        # Aplica operadores morfologicos e rotulacao de componentes conexos
        kernel_size      = compute_adaptive_kernel_size(face_w)
        closed_mask      = apply_closing(skin_removed, kernel_size=kernel_size)
        num_labels, label_matrix, stats, centroids = label_components(closed_mask)
        approved_labels  = shape_filter(stats, num_labels, info['face'])

        save_comparison_grid({
            "Original":     rf_crop,
            "Redness":      redness,
            "Skin Removed": skin_removed,
            "Closing":      closed_mask,
            "Labels":       label_matrix,
            "Shape Filter": np.isin(label_matrix, approved_labels),
        }, f"{base_name}_face_{idx}_steps.png")

        if not approved_labels:
            continue

        expanded_masks = {}
        combined_mask  = np.zeros(rf_crop.shape[:2], dtype=np.uint8)
        iris_info      = {}

        # Processa cada olho vermelho aprovado aplicando crescimento de regiao
        for label in approved_labels:
            exp_mask = region_growing(
                [label], label_matrix, redness, rf_crop, info['face']
            )
            expanded_masks[label] = exp_mask
            combined_mask = cv2.bitwise_or(combined_mask, exp_mask)

            d_iris, center = detect_iris(exp_mask, rf_crop)
            d_pupil        = calculate_pupil_size(d_iris)
            iris_info[label] = (d_iris, center, d_pupil)

            print(f"Label #{label} — d_iris: {d_iris:.1f}px  "
                  f"d_pupil: {d_pupil:.1f}px  center: {center}")

            # Desenha circulo azul contornando a iris na imagem de visualizacao
            cx_full = center[0] + x
            cy_full = center[1] + y
            cv2.circle(img_vis, (cx_full, cy_full),
                       int(round(d_iris / 2)), (255, 0, 0), 2)

        # Remove a cor vermelha da regiao dos olhos
        corrected_crop = inpaint_exemplar(rf_crop, combined_mask)

        for label in approved_labels:
            d_iris, center, d_pupil = iris_info[label]
            corrected_crop = paint_pupil_and_highlight(
                corrected_crop, center, d_pupil)
            corrected_crop = smooth_boundaries(
                corrected_crop, center, d_iris, d_pupil)

        # Salva o corte do olho corrigido e remonta na imagem principal de saida
        write_image(
            f"{base_name}_cortado.png",
            corrected_crop)
        img_corrected[y:y+h, x:x+w] = corrected_crop

    write_image(f"{base_name}_vis.png",      img_vis)
    write_image(f"{base_name}_corrigido.png", img_corrected)

# Inicializa o detector Haar Cascade e executa o processamento para todas as imagens de teste
def run_pipeline():
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(str(cascade_path))
    if face_cascade.empty():
        raise IOError(f"Erro ao carregar classificador de: {cascade_path}")

    for filename in IMAGES:
        print(f"Processando: {filename}")
        
        try:
            img_bgr = read_image(filename)
        
        except Exception as e:
            print(f"Erro: {e}")
            continue

        process_image(img_bgr, Path(filename).stem, face_cascade)

    print("\nProcessamento finalizado.")

if __name__ == "__main__":
    run_pipeline()