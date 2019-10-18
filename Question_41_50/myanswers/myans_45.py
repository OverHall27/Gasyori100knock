import cv2
import numpy as np
import heapq

def HoughTransform(img):

    def k_largest_index_argsort(a, k):
        idx = np.argsort(a.ravel())[:-k-1:-1]
        return np.column_stack(np.unravel_index(idx, a.shape))


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

    '''
    # 8-nearest
    ran, theta = vote_table.shape
    tmp_table = np.pad(vote_table, ([1, 1], [1, 1]), 'constant')
    for th in range(1, theta+1):
        for r in range(1, ran+1):
            arg = np.argmax(tmp_table[r-1:r+2, th-1:th+2])
            if arg != 4:
                vote_table[r-1, th-1] = 0
    '''

    vote_table10 = vote_table.copy()
    # 平坦化して，sort -> 最大10値並ぶ
    max_10_val = np.sort(vote_table10.flatten())[::-1]
    vote_table10[max_10_val[9] > vote_table10] = 0

    '''
    # 上から10個の値のindex
    max_10 = k_largest_index_argsort(vote_table, 10)
    vote_table10 = np.zeros_like(vote_table)
    vote_table10[max_10] = vote_table[max_10]
    '''

    return vote_table, vote_table10

img = cv2.imread("../thorino.jpg")

vote_table, vote_table10 = HoughTransform(img)

cv2.imwrite("myans_44.jpg", vote_table)
cv2.imwrite("myans_45.jpg", vote_table10)
cv2.imshow("vote_table", vote_table)
cv2.imshow("vote_table10", vote_table10)
cv2.waitKey(0)
