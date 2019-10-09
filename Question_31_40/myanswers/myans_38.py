import cv2
import numpy as np

def DCTConstant(u, v):

    if u == 0 and v == 0:
        return 1 / 2
    elif u == 0 or v == 0:
        return 1 / np.sqrt(2)
    else:
        return 1

def DCT(img, block=8):

    Ver, Hor, Col = img.shape

    result = np.zeros((Ver, Hor, Col)).astype(np.float32)
    x = np.tile(np.arange(block), (block, 1))
    y = np.arange(block).repeat(block).reshape([block, block])

    for c in range(Col):
        for u in range(Hor):
            cu = u % block
            iu = u // block
            for v in range(Ver):
                cv = v % block
                iv = v // block
                result[v, u, c] = np.sum(img[iv*block:(iv+1)*block, iu*block:(iu+1)*block, c] * np.cos((2*x+1)*cu*np.pi/(2*block)) * np.cos((2*y+1)*cv*np.pi/(2*block)))
                result[v, u, c] *= DCTConstant(cu, cv)
    result *= (2/block)

    return result

def InvDCT(dct, block=8, K=8):
    Ver, Hor, Col = img.shape

    result = np.zeros((Ver, Hor, Col)).astype(np.float32)

    u = np.tile(np.arange(K), (K, 1))
    v = np.arange(K).repeat(K).reshape([K, K])
    c_uv = np.zeros((K, K))

    for x in range(K):
        for y in range(K):
            c_uv[y, x] = DCTConstant(y, x)
    
    for c in range(Col):
        for x in range(Hor):
            cx = x % block
            ix = x // block
            for y in range(Ver):
                cy = y % block
                iy = y // block
                result[y, x, c] = np.sum(dct[iy*block:iy*block+K, ix*block:ix*block+K, c] * np.cos((2*cx+1)*u*np.pi/(2*block)) * np.cos((2*cy+1)*v*np.pi/(2*block)) * c_uv)
    result *= (2/block)

    return result

def Quantization(img, block=8):
    Ver, Hor, Col = img.shape
    N_Ver = Ver // block
    N_Hor = Hor // block
    Q = np.array(((16, 11, 10, 16, 24, 40, 51, 61),
        (12, 12, 14, 19, 26, 58, 60, 55),
        (14, 13, 16, 24, 40, 57, 69, 56),
        (14, 17, 22, 29, 51, 87, 80, 62),
        (18, 22, 37, 56, 68, 109, 103, 77),
        (24, 35, 55, 64, 81, 104, 113, 92),
        (49, 64, 78, 87, 103, 121, 120, 101),
        (72, 92, 95, 98, 112, 100, 103, 99)), dtype=np.float32)

    result = np.zeros((Ver, Hor, Col)).astype(np.float32)
    for c in range(Col):
        for x in range(N_Hor):
            for y in range(N_Ver):
                result[y*block:(y+1)*block, x*block:(x+1)*block, c] = np.round(img[y*block:(y+1)*block, x*block:(x+1)*block, c] / Q) * Q

    return result

img = cv2.imread("../imori.jpg").astype(np.float32)
K_ = 8
dct = DCT(img, block=8)

dct = Quantization(dct, block=8)
result = InvDCT(dct, block=8, K=K_)
#result = np.clip(result, 0, 255).astype(np.uint8)

cv2.imwrite("myans_38.jpg", result)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

