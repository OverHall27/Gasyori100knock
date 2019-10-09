import cv2
import math
import numpy as np

img = cv2.imread("myans_02.jpg", cv2.IMREAD_GRAYSCALE)
#gree_scale読み込み

if img is None:
    print("failed to lead image")

medium_0 = 0.0
medium_1 = 0.0
weight_0 = 0.0
weight_1 = 0.0

current_sb_pw2 = 0.0
line = 0

for i in range(1, 255):
    for j in range(img.shape[0]):
        for k in range(img.shape[1]):
            if i > img[j,k]:
                weight_0 += 1.0
                medium_0 += img[j,k]
            else:
                weight_1 += 1.0
                medium_1 += img[j,k]

    if weight_0:
        medium_0 = medium_0/weight_0
        weight_0 = weight_0/img.size
    if weight_1:
        medium_1 = medium_1/weight_1
        weight_1 = weight_1/img.size

    sb_pw2 = weight_0 * weight_1 * math.pow((medium_0 - medium_1), 2)
    if current_sb_pw2 < sb_pw2:
        current_sb_pw2 = sb_pw2
        line = i

    medium_0 = 0
    medium_1 = 0
    weight_0 = 0
    weight_1 = 0

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i,j] < line:
            img[i,j] = 0.0
        else:
            img[i,j] = 255.0

ans_img = cv2.imwrite("myans_04.jpg", img)
