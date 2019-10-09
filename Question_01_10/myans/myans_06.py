import cv2
import numpy as np

#BGRの順番で読み込んでいる
img = cv2.imread("imori.jpg")
result = img.copy()
result = ((result // 64) * 2 + 1) * 32

cv2.imwrite("myans_06.jpg", result)
cv2.imshow("result", result)
