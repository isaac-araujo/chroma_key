import cv2
import os
import numpy as np

def get_dominant_hue(hues):
    unique_hue, counts = np.unique(hues, return_counts=True)

    # Verificando cor dominante
    hue_color = None
    biggest = -1
    for a in range(len(unique_hue)):
        if counts[a] > biggest:
            biggest = counts[a]
            hue_color = int(unique_hue[a])

    return hue_color

def create_mask(hue_color, tolerance, hues, values, output_path):
    min_green = hue_color - tolerance
    max_green = hue_color + tolerance
    
    # Criando Mascara
    mask_hue = cv2.inRange(hues, min_green, max_green)
    # cv2.imwrite(output_path + 'mask_hue.png', mask_hue)

    mask_val = cv2.inRange(values, 100, 255)
    # cv2.imwrite(output_path + 'mask_val.png', mask_val)

    mask = cv2.bitwise_and(mask_hue, mask_val)
    # cv2.imwrite(output_path + 'mask.png', mask)

    return mask

def main(tolerance=13):
    images = ['corvo', 'corvos', 'formas', 'rainha', 'spider', 'mamaco']
    img_background = cv2.imread('images/background.bmp')
    for img_name in images:

        # Criando pasta de output
        output_path = f'./images_{img_name}/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Carregando imagem
        img = cv2.imread(f'images/{img_name}.bmp')

        # Redimensionando imagem
        height, width, _ = img.shape
        img_background_resized = cv2.resize(img_background, (width, height))
        # cv2.imwrite('back_resized.png', background_resized)

        # Covertendo para HSV
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        hues, _, values = cv2.split(img_hsv)

        dominant_hue = get_dominant_hue(hues)

        img_mask = create_mask(dominant_hue, tolerance, hues, values, output_path)

        # Invertendo mascara
        img_mask_inv = cv2.bitwise_not(img_mask)

        # Aplicando mascara na imagem 
        img_mask_aplly  = cv2.bitwise_and(img, img, mask=img_mask_inv)

        # Aplicando mascara no backgroud
        img_background_rmv = cv2.bitwise_and(img_background_resized, img_background_resized, mask=img_mask)

        # Somando as duas imagens
        img_chroma_key = cv2.add(img_mask_aplly, img_background_rmv)

        # Aplicando Anti-Aliasing 
        img_bgr = cv2.cvtColor(img_mask_aplly, cv2.COLOR_HSV2BGR)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        img_edges = cv2.Canny(img_gray, 50, 100)
        cv2.imwrite(output_path + 'img_edges.png', img_edges)

        # Borrando Bordas
        blurred_edges = cv2.GaussianBlur(img_edges, (23, 23), 0)
        blurred_edges_bgr = cv2.cvtColor(blurred_edges, cv2.COLOR_GRAY2BGR)

        anti_aliased_image = cv2.addWeighted(img_mask_aplly, 1, blurred_edges_bgr, 0.65, 0)
        cv2.imwrite(output_path + 'anti_aliased_image.png', anti_aliased_image)

        img_chroma_key_aa = cv2.add(anti_aliased_image, img_background_rmv)

        cv2.imwrite(output_path + 'result_before_aa.png', img_chroma_key)
        cv2.imwrite(output_path + f'binarizacao_{img_name}.png', img_mask_inv)
        cv2.imwrite(output_path + f'gauss_{img_name}.png', blurred_edges_bgr)
        cv2.imwrite(output_path + 'resultado_final.png', img_chroma_key_aa)

if __name__ == '__main__':
    main()