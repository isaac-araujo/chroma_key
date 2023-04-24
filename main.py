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
    background = cv2.imread('images/background.bmp')
    for img_name in images:
        # Criando pasta de output
        output_path = f'./images_{img_name}/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Carregando imagem
        img = cv2.imread(f'images/{img_name}.bmp')

        # Redimensionando imagem
        height, width, _ = img.shape
        background_resized = cv2.resize(background, (width, height))
        # cv2.imwrite('back_resized.png', background_resized)

        # Covertendo para HSV
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        hues, _, values = cv2.split(img_hsv)

        dominant_hue = get_dominant_hue(hues)

        mask = create_mask(dominant_hue, tolerance, hues, values, output_path)

        # Invertendo mascara
        mask_inv = cv2.bitwise_not(mask)

        # Aplicando mascara na imagem 
        mask_aplly_img = cv2.bitwise_and(img, img, mask=mask_inv)
        # cv2.imwrite(output_path + 'img_removed_green.png', mask_aplly_img)

        # Aplicando mascara no backgroud
        background_rmv = cv2.bitwise_and(background_resized, background_resized, mask=mask)
        # cv2.imwrite('background_rmv.png', background_rmv)

        result_before_aa = cv2.add(mask_aplly_img, background_rmv)

        cv2.imwrite(output_path + 'result_before_aa.png', result_before_aa)
        cv2.imwrite(output_path + f'binarizacao_{img_name}.png', mask_inv)
        # cv2.imwrite(output_path + f'gauss_{img_name}.png', blur)
        # cv2.imwrite(output_path + 'resultado_final.png', img_result)

if __name__ == '__main__':
    # tolerance = abs(int(input('hue tolerance: ')))
    main()