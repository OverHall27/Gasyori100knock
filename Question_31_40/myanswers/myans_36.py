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
            #block内での画像位置
            cu = u % block
            #何番目のブロックか
            iu = u // block
            for v in range(Ver):
                #block内での画像位置
                cv = v % block
                #何番目のブロックか
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

img = cv2.imread("../imori.jpg")

dct = DCT(img, block=8)
cv2.imshow("dct", dct)
result = InvDCT(dct, block=8, K=8)
result = np.clip(result, 0, 255).astype(np.uint8)

cv2.imwrite("myans_36.jpg", result)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
