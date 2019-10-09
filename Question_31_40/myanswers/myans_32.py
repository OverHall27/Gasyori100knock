import cv2
import numpy as np

# parameters
channel = 3

def memo():
    '''
    u = np.arange(Hor // 2)
    u = np.hstack((x[::-1], x))
    v = x.repeat(Hor).reshape(Ver, -1)
    u = np.tile(x, (Ver, 1))
    '''

def BGRTOGRAY(_img):
    img = np.zeros((_img.shape[0], _img.shape[1]))
    img = _img[:,:,2] * 0.2126 + _img[:,:,1] * 0.7152 + _img[:,:,0] * 0.0722

    return img.astype(np.uint8)

def DFT(_img):
    Ver, Hor, _ = _img.shape

    x = np.tile(np.arange(Hor), (Ver, 1))
    y = np.arange(Ver).repeat(Hor).reshape(Ver, -1)

    result = np.zeros((Ver, Hor), dtype=np.complex)
    img = BGRTOGRAY(_img)

    for u in range(Hor):
        for v in range(Ver):
            result[v, u] = np.sum(img * np.exp( -2j * np.pi * ((v * y / Ver) + (u * x / Hor)) ) )
    result /= np.sqrt(Hor * Ver)


    '''
    result = np.zeros((Ver, Hor, channel), dtype=np.complex)
    img = _img.copy()

    for c in range(channel):
        for u in range(Hor):
            for v in range(Ver):
                result[v, u, c] = np.sum(img[..., c] * np.exp( -2j * np.pi * ((v * y / Ver) + (u * x / Hor)) ) )
    result /= np.sqrt(Hor * Ver)
    '''

    return result

def IDFT(img):
    Ver, Hor = img.shape
    u = np.tile(np.arange(Hor), (Ver, 1))
    v = np.arange(Ver).repeat(Hor).reshape(Ver, -1)

    '''
    result = np.zeros((Ver, Hor, channel), dtype=np.float)
    for c in range(channel):
        for x in range(Hor):
            for y in range(Ver):
                result[y, x, c] = np.abs(np.sum(img[..., c] * np.exp( 2j * np.pi * ((v * y / Ver) + (u * x / Hor)) ) ))
    result /= np.sqrt(Hor * Ver)
    '''

    result = np.zeros((Ver, Hor), dtype=np.float)
    for x in range(Hor):
        for y in range(Ver):
            result[y, x] = np.abs(np.sum(img * np.exp( 2j * np.pi * ((v * y / Ver) + (u * x / Hor)) ) ))
    result /= np.sqrt(Hor * Ver)

    return np.clip(result, 0, 255).astype(np.uint8)

img = cv2.imread("../imori.jpg")

result_dft = DFT(img)
result = IDFT(result_dft)

result_ps1 = np.clip(np.abs(result_dft), 0, 255)
result_ps2 = (np.abs(result_dft) / np.abs(result_dft).max() * 255).astype(np.uint8)

cv2.imshow("result_ps1", result_ps1)
cv2.imshow("result_ps2", result_ps2)
cv2.imshow("inv result", result)

cv2.imwrite("myans_32_ps.jpg", result_ps2)
cv2.imwrite("myans_32.jpg", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
