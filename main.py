import cv2
import os
import numpy as np


images = ['corvo', 'corvos', 'formas', 'rainha']

background = cv2.imread('images/background.bmp')

for img_name in images:
    output_path = f'./images_{img_name}/'
    # Carregando imagem
    img = cv2.imread(f'images/{img_name}.bmp')

    # Redimensionando imagem
    height, width, _ = img.shape
    background_resized = cv2.resize(background, (width, height))
    # cv2.imwrite('back_resized.png', background_resized)

    # Covertendo para HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(img_hsv)

    # Pegando todas a cores 
    # unique_hue, counts = np.unique(h, return_counts=True)

    # Verificando cor domintante
    # hue_color = None
    # biggest = -1
    # for a in range(len(unique_hue)):
    #     if counts[a] > biggest:
    #         biggest = counts[a]
    #         hue_color = int(unique_hue[a])

    hue_color = 57

    tolerance = 2
    min_green = hue_color - tolerance
    max_green = hue_color + tolerance

    # Criando Masacara
    mask = cv2.inRange(h, min_green, max_green)
    # cv2.imwrite('mask_green.png', mask)

    # Invertendo mascara
    mask_inv = cv2.bitwise_not(mask)
    # cv2.imwrite('mask_green_inv.png', mask_inv)

    # Aplicando mascara na imagem 
    mask_aplly_img = cv2.bitwise_and(img, img, mask=mask_inv)
    # cv2.imwrite('img_removed_green.png', mask_aplly_img)

    # Aplicando mascara no backgroud
    background_rmv = cv2.bitwise_and(background_resized, background_resized, mask=mask)
    # cv2.imwrite('background_rmv.png', background_rmv)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    result = cv2.add(mask_aplly_img, background_rmv)
    cv2.imwrite(output_path + 'resultado_final.png', result)