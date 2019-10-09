import cv2
import numpy as np

def BGRGRAY(_img):
    img = np.zeros((_img.shape[0], _img.shape[1]))
    img = _img[:,:,2] * 0.2126 + _img[:,:,1] * 0.7152 + _img[:,:,0] * 0.0722
    return img

def DifferentialFilter(img, K_size=3, axis=0):
    Hol, Ber = img.shape

    ## zero padding
    result = np.zeros((Hol + 2, Ber + 2), dtype=np.float)
    temp = result.copy()
    temp[1: 1 + Hol, 1: 1 + Ber] += img.copy().astype(np.float)

    kernel = np.zeros((K_size, K_size), dtype=np.float32)
    ##create filter
    pad = K_size // 2
    if axis == 0:
        kernel[pad - 1, pad] = -1.
        kernel[pad, pad] = 1.
    elif axis == 1:
        kernel[pad, pad - 1] = -1.
        kernel[pad, pad] = 1.

    for x in range(Hol):
        for y in range(Ber):
            result[1 + x, 1 + y] = np.sum(temp[x: x + K_size, y: y + K_size] * kernel)

    ## このclipが必要
    result = np.clip(result, 0, 255)
    result = result[pad: pad + Hol, pad: pad + Ber].astype(np.uint8)


    return result

img = cv2.imread("imori.jpg")
#エッジ検出は白黒画像に!!
img = BGRGRAY(img)

result_v = DifferentialFilter(img, K_size=3, axis=0)
result_h = DifferentialFilter(img, K_size=3, axis=1)

cv2.imwrite("myans_14_v.jpg", result_v)
cv2.imwrite("myans_14_h.jpg", result_h)
