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

def main(tolerance=15):
    images = ['corvo', 'corvos', 'formas', 'rainha']
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

        hue_color = get_dominant_hue(hues)

        min_green = hue_color - tolerance
        max_green = hue_color + tolerance

        # Criando Mascara
        mask_hue = cv2.inRange(hues, min_green, max_green)
        cv2.imwrite(output_path + 'mask_hue.png', mask_hue)

        mask_val = cv2.inRange(values, 80, 255)
        cv2.imwrite(output_path + 'mask_val.png', mask_val)

        mask = cv2.bitwise_and(mask_hue, mask_val)
        cv2.imwrite(output_path + 'mask.png', mask)

        # Invertendo mascara
        mask_inv = cv2.bitwise_not(mask)

        # Aplicando mascara na imagem 
        mask_aplly_img = cv2.bitwise_and(img, img, mask=mask_inv)
        cv2.imwrite(output_path + 'img_removed_green.png', mask_aplly_img)

        # Aplicando mascara no backgroud
        background_rmv = cv2.bitwise_and(background_resized, background_resized, mask=mask)
        # cv2.imwrite('background_rmv.png', background_rmv)

        gray = cv2.cvtColor(mask_aplly_img, cv2.COLOR_BGR2GRAY)
        hsv_gray = cv2.cvtColor(img_hsv, cv2.COLOR_BGR2GRAY)

        gauss = cv2.GaussianBlur(gray, (3, 3), 0) 
        canny = cv2.Canny(gray, 100, 200) 

        result = cv2.add(mask_aplly_img, background_rmv)
        
        cv2.imwrite(output_path + f'binarizacao_{img_name}.png', mask_inv)
        cv2.imwrite(output_path + f'gauss_{img_name}.png', gauss)
        cv2.imwrite(output_path + f'canny_{img_name}.png', canny)
        cv2.imwrite(output_path + 'resultado_final.png', result)

if __name__ == '__main__':
    # tolerance = abs(int(input('hue tolerance: ')))
    main()