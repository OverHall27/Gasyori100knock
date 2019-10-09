import cv2
import numpy as np

def BGRGRAY(_img):
    img = np.zeros((_img.shape[0], _img.shape[1]))
    img = _img[:,:,2] * 0.2126 + _img[:,:,1] * 0.7152 + _img[:,:,0] * 0.0722
    return img

def MaxMinFilter(img, K_size=3):
    Hol, Ber = img.shape
    
    ## zero padding
    result = np.zeros((Hol + 2, Ber + 2), dtype=np.float)
    temp = result.copy()
    temp[1: 1 + Hol, 1: 1 + Ber] += img.copy().astype(np.float)

    for x in range(Hol):
        for y in range(Ber):
            Max = np.max(temp[x: x + K_size, y: y + K_size])
            Min = np.min(temp[x: x + K_size, y: y + K_size])
            result[1 + x, 1 + y] = Max - Min
    result =  result[1: 1 + Hol, 1: 1 + Ber].astype(np.uint8)

    return result

img = cv2.imread("imori.jpg")
#maxminフィルタは白黒画像に!!
img = BGRGRAY(img)

result = MaxMinFilter(img, K_size=3)
cv2.imwrite("myans_13.jpg", result)
