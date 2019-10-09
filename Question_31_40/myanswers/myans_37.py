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

    result = np.zeros((Ver, Hor, Col)).astype(np.float)
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

    result = np.zeros((Ver, Hor, Col)).astype(np.float)

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

def PSNR(img, app_img, Max=255):
    Ver, Hor, Col = img.shape

    x = np.tile(np.arange(Hor), (Ver, 1))
    y = np.arange(Ver).repeat(Hor).reshape([Ver, Hor])
    Max = np.max(img)

    MSE = np.sum((img[y, x] - app_img[y, x]) ** 2) / (Ver * Hor)

    return 10 * np.log10((Max ** 2) / MSE)

img = cv2.imread("../imori.jpg")
K_ = 4
dct = DCT(img, block=8)
cv2.imshow("dct", dct)
result = InvDCT(dct, block=8, K=K_)
result = np.clip(result, 0, 255).astype(np.uint8)

psnr = PSNR(img, result, 255)
bitrate = 8 * (K_ ** 2) / (8 ** 2)
print("psnr", psnr)
print("bitrate", bitrate)

cv2.imshow("result", result)
cv2.imwrite("myans_37.jpg", result)
cv2.imshow("result", result)
cv2.waitKey(0)
