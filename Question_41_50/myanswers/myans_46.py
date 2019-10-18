import cv2
import numpy as np
import heapq

def HoughTransform(img):

    def BGRTOGRAY(img):
        gray_img = img[..., 2] * 0.2126 + img[..., 1] * 0.7152 + img[..., 0] * 0.0722
        gray_img = gray_img.astype(np.uint8)

        return gray_img


    gray = BGRTOGRAY(img)
    edge = cv2.Canny(gray, threshold1=30., threshold2=100., apertureSize=3, L2gradient=True)

    Ver, Hor = edge.shape
    diagonal = int(np.round(np.sqrt(Ver ** 2 + Hor ** 2)))

    vote_table = np.zeros((diagonal, 180), dtype=np.uint8)
    ind = np.where(edge != 0)
    x = ind[0]
    y = ind[1]

    for t in range(180):
        it = np.pi * t / 180
        r = x * np.cos(it) + y * np.sin(it)
        for i in range(len(ind[0])):
            ir = int( np.round(r[i]) )
            vote_table[ir, t] += 1


    vote_table10 = vote_table.copy()
    # 平坦化して，sort -> 最大10値並ぶ
    max_10_val = np.sort(vote_table10.flatten())[::-1]
    vote_table10[max_10_val[9] > vote_table10] = 0

    ind = np.where(vote_table10 != 0)
    for t in range(180):
        for r in range(diagonal):
            iy = - np.cos(t) / np.sin(t) * x + r / np.sin(t)
            ix = - np.sin(t) / np.cos(t) * y + r / np.sin(t)
            swor(

    return vote_table10

img = cv2.imread("../thorino.jpg")

vote_table, vote_table10 = HoughTransform(img)

cv2.imwrite("myans_44.jpg", vote_table)
cv2.imwrite("myans_45.jpg", vote_table10)
cv2.imshow("vote_table", vote_table)
cv2.imshow("vote_table10", vote_table10)

