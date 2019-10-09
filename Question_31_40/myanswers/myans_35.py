import cv2
import numpy as np

# parameters
channel = 3

def BGRTOGRAY(img):
    gray_img = img[..., 2] * 0.2126 + img[..., 1] * 0.7152 + img[..., 0] * 0.0722
    
    return gray_img

def DFT(img):
    Ver = img.shape[0]
    Hor = img.shape[1]

    x = np.tile(np.arange(Hor), (Ver, 1))
    y = np.arange(Ver).repeat(Hor).reshape(Ver, -1)

    result = np.zeros((Ver, Hor, channel), dtype=np.complex)

    for c in range(channel):
        for u in range(Hor):
            for v in range(Ver):
                result[u, v, c] = np.sum(img[..., c] * np.exp( -2j * np.pi * ((v * x / Hor) + (u * y / Ver)) ) )
    result /= np.sqrt(Hor * Ver)

    return result

def AssembleFreq(img):

    Ver = img.shape[0]
    Hor = img.shape[1]
    ix = Hor // 2
    iy = Ver // 2

    # 第N象限で分ける
    _img = np.zeros_like(img)
    _img[:iy, ix:] = img[iy:, :ix]
    _img[:iy, :ix] = img[iy:, ix:]
    _img[iy:, :ix] = img[:iy, ix:]
    _img[iy:, ix:] = img[:iy, :ix]

    return _img

def BandPassFilter(_img, low_ratio, high_rario):
    img = AssembleFreq(_img)
    Ver = img.shape[0]
    Hor = img.shape[1]
    low_r = (Ver // 2) * low_ratio
    high_r = (Ver // 2) * high_rario

    for x in range(Hor):
        ix = x - (Hor // 2)
        for y in range(Ver):
            iy = y - (Ver // 2)
            pos = np.sqrt( ix ** 2 + iy ** 2)
            if (pos < low_r) or (pos > high_r):
                img[y, x] = 0

    img = AssembleFreq(img)

    return img

def IDFT(img):
    Ver = img.shape[0]
    Hor = img.shape[1]

    result = np.zeros((Ver, Hor, channel), dtype=np.float32)

    u = np.tile(np.arange(Hor), (Ver, 1))
    v = np.arange(Ver).repeat(Hor).reshape(Ver, -1)

    for c in range(channel):
        for x in range(Hor):
            for y in range(Ver):
                result[x, y, c] = np.abs(np.sum(img[..., c] * np.exp( 2j * np.pi * ((u * y / Ver) + (v * x / Hor)) ) ))
    result /= np.sqrt(Hor * Ver)

    return result

img = cv2.imread("../imori.jpg").astype(np.float)

dft = DFT(img)
low_pass = BandPassFilter(dft, low_ratio=0.1, high_rario=0.5)
result = IDFT(low_pass)
result = np.clip(result, 0, 255).astype(np.uint8)

cv2.imshow("result", result)
cv2.imwrite("myans_35.jpg", result)
cv2.waitKey(0)
