import cv2
import numpy as np

def BGRGRAY(_img):

    img = np.zeros((_img.shape[0], _img.shape[1]), dtype=np.float32)
    img = _img[:,:,2].copy() * 0.2126 + _img[:,:,1].copy() * 0.7152 + _img[:,:,0].copy() * 0.0722

    return img.astype(np.uint8)

def EmbossFilter(img, K_size=3):

    Hol, Ver = img.shape
    pad = K_size // 2

    result = np.zeros((Hol + pad * 2, Ver + pad * 2), dtype=np.float)
    tmp = result.copy()
    tmp[pad: pad + Hol, pad: pad + Ver] = img.copy().astype(np.float)

    ##create filter
    kernel = [[-2., -1., 0.], [-1, 1., 1.], [0., 1., 2.]]

    for x in range(Hol):
        for y in range(Ver):
            result[pad + x, pad + y] = np.sum(tmp[x: x + K_size, y: y + K_size] * kernel)

    result = np.clip(result, 0, 255)
    result = result[pad: pad + Hol, pad: pad + Ver].astype(np.uint8)

    return result


img = cv2.imread("imori.jpg")

gray_img = BGRGRAY(img)
result = EmbossFilter(gray_img, K_size=3)
cv2.imwrite("myans_18.jpg", result)
