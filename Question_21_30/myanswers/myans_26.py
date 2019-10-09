import cv2
import numpy as np

#実装ミス？とか色々な要因に悩んだ挙げ句，タイポでした．死にたい
def BilinearNeighbor(img, ax=1., ay=1.):
    Hol, Ver, Col = img.shape

    aH = int(ax * Hol)
    aV = int(ay * Ver)

    result = np.zeros((aH, aV, Col))

    # x_, y_ がオリジナル画像での画素位置を示すものになる
    for h in range(aH):
        x = h / ax
        ix = np.floor(x).astype(np.int)
        if ix >= Hol - 2:
            ix = int(Hol - 2)
        dx = float(x - ix)

        for v in range(aV):
            y = v / ay
            iy = np.floor(y).astype(np.int)
            if iy >= Ver - 2:
                iy = int(Ver - 2)
            dy = float(y - iy)


            result[h, v] = (1-dx) * (1-dy) * img[ix, iy] + (1-dx) * dy * img[ix, iy+1] + dx * (1-dy) * img[ix+1, iy] + dx * dy * img[ix+1, iy+1]

            result = np.clip(result, 0, 255)
            result = result.astype(np.uint8)

    return result


img = cv2.imread("../imori.jpg").astype(np.float32)

result = BilinearNeighbor(img, ax=1.5, ay=1.5)

cv2.imwrite("myans_26.jpg", result)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
