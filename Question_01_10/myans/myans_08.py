import cv2
import numpy as np

def MaxPooling(_img):
    img = _img.copy()
    result = np.zeros_like(img)
    for i in range(img.shape[0]//8):
        ind_11 = i * 8
        ind_12 = ind_11 + 8
        for j in range(img.shape[1]//8):
            ind_21 = j * 8
            ind_22 = ind_21 + 8
            result[ind_11:ind_12, ind_21:ind_22, 0] = np.max(img[ind_11:ind_12, ind_21:ind_22, 0])
            result[ind_11:ind_12, ind_21:ind_22, 1] = np.max(img[ind_11:ind_12, ind_21:ind_22, 1])
            result[ind_11:ind_12, ind_21:ind_22, 2] = np.max(img[ind_11:ind_12, ind_21:ind_22, 2])

    return result

img = cv2.imread("imori.jpg")

result = MaxPooling(img)
cv2.imwrite("myans_08.jpg", result)
