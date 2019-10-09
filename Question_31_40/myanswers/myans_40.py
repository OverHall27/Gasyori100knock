import cv2
import numpy as np

IMG_V = 128
IMG_H = 128
channel = 3
K_ = 8

def DCTConstant(u, v):

    if u == 0 and v == 0:
        return 1 / 2
    elif u == 0 or v == 0:
        return 1 / np.sqrt(2)
    else:
        return 1

def DCT(img, block=8):

    if len(img.shape) == 3:
        Ver, Hor, Col = img.shape
        result = np.zeros((Ver, Hor, Col)).astype(np.float32)
    else:
        Ver, Hor = img.shape
        Col = 1
        result = np.zeros((Ver, Hor)).astype(np.float32)

    x = np.tile(np.arange(block), (block, 1))
    y = np.arange(block).repeat(block).reshape([block, block])

    for u in range(Hor):
        cu = u % block
        iu = u // block
        for v in range(Ver):
            cv = v % block
            iv = v // block
            if Col == 3:
                for c in range(Col):
                    result[v, u, c] = np.sum(img[iv*block:(iv+1)*block, iu*block:(iu+1)*block, c] * np.cos((2*x+1)*cu*np.pi/(2*block)) * np.cos((2*y+1)*cv*np.pi/(2*block)))
                    result[v, u, c] *= DCTConstant(cu, cv)
            else:
                result[v, u] = np.sum(img[iv*block:(iv+1)*block, iu*block:(iu+1)*block] * np.cos((2*x+1)*cu*np.pi/(2*block)) * np.cos((2*y+1)*cv*np.pi/(2*block)))
                result[v, u] *= DCTConstant(cu, cv)


    result *= (2/block)

    return result

def InvDCT(dct, block=8, K=8):
    if len(dct.shape) == 3:
        Ver, Hor, Col = dct.shape
        result = np.zeros((Ver, Hor, Col)).astype(np.float32)
    else:
        Ver, Hor = dct.shape
        Col = 1
        result = np.zeros((Ver, Hor)).astype(np.float32)

    u = np.tile(np.arange(K), (K, 1))
    v = np.arange(K).repeat(K).reshape([K, K])
    c_uv = np.zeros((K, K))

    for x in range(K):
        for y in range(K):
            c_uv[y, x] = DCTConstant(y, x)
    
    for x in range(Hor):
        cx = x % block
        ix = x // block
        for y in range(Ver):
            cy = y % block
            iy = y // block
            if Col == 3:
                for c in range(Col):
                    result[y, x, c] = np.sum(dct[iy*block:iy*block+K, ix*block:ix*block+K, c] * np.cos((2*cx+1)*u*np.pi/(2*block)) * np.cos((2*cy+1)*v*np.pi/(2*block)) * c_uv)
            else:
                result[y, x] = np.sum(dct[iy*block:iy*block+K, ix*block:ix*block+K] * np.cos((2*cx+1)*u*np.pi/(2*block)) * np.cos((2*cy+1)*v*np.pi/(2*block)) * c_uv)
    result *= (2/block)

    return result

def Quantization1(img, block=8):

    if len(img.shape) == 3:
        Ver, Hor, Col = img.shape
        result = np.zeros((Ver, Hor, Col)).astype(np.float32)
    else:
        Ver, Hor = img.shape
        Col = 1
        result = np.zeros((Ver, Hor)).astype(np.float32)
    N_Ver = Ver // block
    N_Hor = Hor // block
    Q1 = np.array(((16, 11, 10, 16, 24, 40, 51, 61),
        (12, 12, 14, 19, 26, 58, 60, 55),
        (14, 13, 16, 24, 40, 57, 69, 56),
        (14, 17, 22, 29, 51, 87, 80, 62),
        (18, 22, 37, 56, 68, 109, 103, 77),
        (24, 35, 55, 64, 81, 104, 113, 92),
        (49, 64, 78, 87, 103, 121, 120, 101),
        (72, 92, 95, 98, 112, 100, 103, 99)), dtype=np.float32)

    for x in range(N_Hor):
        for y in range(N_Ver):
            if Col == 3:
                for c in range(Col):
                    result[y*block:(y+1)*block, x*block:(x+1)*block, c] = np.round(img[y*block:(y+1)*block, x*block:(x+1)*block, c] / Q1) * Q1
            else:
                result[y*block:(y+1)*block, x*block:(x+1)*block] = np.round(img[y*block:(y+1)*block, x*block:(x+1)*block] / Q1) * Q1


    return result

def Quantization2(img, block=8):
    if len(img.shape) == 3:
        Ver, Hor, Col = img.shape
        result = np.zeros((Ver, Hor, Col)).astype(np.float32)
    else:
        Ver, Hor = img.shape
        Col = 1
        result = np.zeros((Ver, Hor)).astype(np.float32)
    N_Ver = Ver // block
    N_Hor = Hor // block
    Q2 = np.array(((17, 18, 24, 47, 99, 99, 99, 99),
        (18, 21, 26, 66, 99, 99, 99, 99),
        (24, 26, 56, 99, 99, 99, 99, 99),
        (47, 66, 99, 99, 99, 99, 99, 99),
        (99, 99, 99, 99, 99, 99, 99, 99),
        (99, 99, 99, 99, 99, 99, 99, 99),
        (99, 99, 99, 99, 99, 99, 99, 99),
        (99, 99, 99, 99, 99, 99, 99, 99)), dtype=np.float32)
    for x in range(N_Hor):
        for y in range(N_Ver):
            if Col == 3:
                for c in range(Col):
                    result[y*block:(y+1)*block, x*block:(x+1)*block, c] = np.round(img[y*block:(y+1)*block, x*block:(x+1)*block, c] / Q2) * Q2
            else:
                result[y*block:(y+1)*block, x*block:(x+1)*block] = np.round(img[y*block:(y+1)*block, x*block:(x+1)*block] / Q2) * Q2
    return result

def BGRtoYCbCr(img):

    Y = 0.299 * img[..., 2] + 0.5870 * img[..., 1] + 0.114 * img[..., 0]
    Cb = -0.1687 * img[..., 2] - 0.3313 * img[..., 1] + 0.5 * img[..., 0] + 128.
    Cr = 0.5 * img[..., 2] - 0.4187 * img[..., 1] - 0.0813 * img[..., 0] + 128.

    return Y, Cb, Cr

def YCbCrtoBGR(Y, Cb, Cr):

    R = Y + (Cr - 128) * 1.402
    G = Y - (Cb - 128) * 0.3441 - (Cr - 128) * 0.7139
    B = Y + (Cb - 128) * 1.7718

    result = np.zeros((IMG_V, IMG_H, channel)).astype(np.float)

    result[..., 0] = B
    result[..., 1] = G
    result[..., 2] = R

    return result

img = cv2.imread("../imori.jpg").astype(np.float32)

Y, Cb, Cr = BGRtoYCbCr(img)

Y_dct = DCT(Y, block=8)
Cb_dct = DCT(Cb, block=8)
Cr_dct = DCT(Cr, block=8)

Y_dct = Quantization1(Y_dct, block=8)
Cb_dct = Quantization2(Cb_dct, block=8)
Cr_dct = Quantization2(Cr_dct, block=8)

Y_result  = InvDCT(Y_dct, block=8, K=K_)
Cb_result = InvDCT(Cb_dct, block=8, K=K_)
Cr_result = InvDCT(Cr_dct, block=8, K=K_)

result = YCbCrtoBGR(Y_result, Cb_result, Cr_result)
result = np.clip(result, 0, 255).astype(np.uint8)

cv2.imwrite("myans_40.jpg", result)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
