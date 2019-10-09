import cv2
import math

img = cv2.imread("imori.jpg")
#BGRの順番で読み込んでいる
if img is None:
    print("failed to lead image")

ans_img = img[:,:,2] * 0.2126 + img[:,:,1] * 0.7152 + img[:,:,0] * 0.0722

for i in range(ans_img.shape[0]):
    for j in range(ans_img.shape[1]):
        if ans_img[i,j] < 128.0:
            ans_img[i,j] = 0.0
        else:
            ans_img[i,j] = 255.0

ans_img = cv2.imwrite("myans_03.jpg", ans_img)
