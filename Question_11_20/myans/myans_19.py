import cv2
import numpy as np

def BGRGRAY(_img):

    img = np.zeros((_img.shape[0], _img.shape[1]), dtype=np.float32)
    img = _img[:,:,2].copy() * 0.2126 + _img[:,:,1].copy() * 0.7152 + _img[:,:,0].copy() * 0.0722

    return img.astype(np.uint8)

def LogFilter(img, K_size=5, sigma=3.):

    Hol, Ver = img.shape
    pad = K_size // 2

    result = np.zeros((Hol + pad * 2, Ver + pad * 2), dtype=np.float)
    tmp = result.copy()
    tmp[pad: pad + Hol, pad: pad + Ver] = img.copy().astype(np.float)

    ##create filter
    kernel = np.zeros((K_size, K_size), dtype=np.float)
    for x in range(-pad, -pad + K_size):
        for y in range(-pad, -pad + K_size):
            kernel[pad + x, pad + y] = np.exp( - ( x ** 2 + y ** 2) / ( 2. * (sigma ** 2)))
            kernel[pad + x, pad + y] *= ( (x ** 2 + y ** 2) - sigma ** 2)

    kernel /= ( (sigma ** 6) * (2. * np.pi))
    kernel /= np.abs(kernel.sum())
    print(kernel)

    for x in range(Hol):
        for y in range(Ver):
            result[pad + x, pad + y] = np.sum(tmp[x: x + K_size, y: y + K_size] * kernel)

    result = np.clip(result, 0, 255)
    result = result[pad: pad + Hol, pad: pad + Ver].astype(np.uint8)

    return result


img = cv2.imread("imori.jpg")

gray_img = BGRGRAY(img)
result = LogFilter(gray_img, K_size=5, sigma=3.)

cv2.imwrite("myans_19.jpg", result)
