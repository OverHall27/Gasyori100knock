import cv2
import numpy as np

def BGRGRAY(_img):
    img = np.zeros((_img.shape[0], _img.shape[1]))
    img = _img[:,:,2] * 0.2126 + _img[:,:,1] * 0.7152 + _img[:,:,0] * 0.0722
    return img

def SobelFilter(img, K_size=3):
    Hol, Ver = img.shape

    ## zero padding
    result_v = np.zeros((Hol + 2, Ver + 2), dtype=np.float)
    result_h = np.zeros((Hol + 2, Ver + 2), dtype=np.float)
    temp = result_h.copy()
    temp[1: 1 + Hol, 1: 1 + Ver] += img.copy().astype(np.float)

    kernel_v = np.zeros((K_size, K_size), dtype=np.float32)
    kernel_h = np.zeros((K_size, K_size), dtype=np.float32)
    ##create filter
    pad = K_size // 2

    kernel_v = [[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]]
    kernel_h = [[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]]

    for x in range(Hol):
        for y in range(Ver):
            result_v[1 + x, 1 + y] = np.sum(temp[x: x + K_size, y: y + K_size] * kernel_v)
            result_h[1 + x, 1 + y] = np.sum(temp[x: x + K_size, y: y + K_size] * kernel_h)

    ## このclipが必要
    result_v = np.clip(result_v, 0, 255)
    result_h = np.clip(result_h, 0, 255)

    result_v = result_v[pad: pad + Hol, pad: pad + Ver].astype(np.uint8)
    result_h = result_h[pad: pad + Hol, pad: pad + Ver].astype(np.uint8)

    return result_v, result_h

img = cv2.imread("imori.jpg")
#エッジ検出は白黒画像に!!
img = BGRGRAY(img)

result_v, result_h = SobelFilter(img, K_size=3)

cv2.imwrite("myans_15_v.jpg", result_v)
cv2.imwrite("myans_15_h.jpg", result_h)
