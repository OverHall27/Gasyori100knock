import cv2
import numpy as np

def BGRtoGRAY(img):
    gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
    return gray

def BilinearNeighbor(img, ax=1., ay=1.):

    if len(img.shape) == 3:
        Ver, Hor, Col = img.shape
    else:
        Ver, Hor = img.shape

    aH = int(ax * Hor)
    aV = int(ay * Ver)

    result = np.zeros((aV, aH))

    # x_, y_ がオリジナル画像での画素位置を示すものになる
    for h in range(aH):
        x = h / ax
        ix = np.floor(x).astype(np.int)
        if ix >= Hor - 2:
            ix = int(Hor - 2)
        dx = float(x - ix)

        for v in range(aV):
            y = v / ay
            iy = np.floor(y).astype(np.int)
            if iy >= Ver - 2:
                iy = int(Ver - 2)
            dy = float(y - iy)

            s1 = (1-dx) * (1-dy) * img[iy, ix]
            s2 = (1-dx) * dy * img[iy+1, ix]
            s3 = dx * (1-dy) * img[iy, ix+1]
            s4 = dx * dy * img[ix+1, iy+1]

            result[v, h] = s1 + s2 + s3 + s4
            result = np.clip(result, 0, 255)
            result = result.astype(np.uint8)

    return result

img = cv2.imread("../imori.jpg").astype(np.float32)
gray = BGRtoGRAY(img)

gray = BilinearNeighbor(gray, 0.5, 0.5)
result = BilinearNeighbor(gray, 2.0, 2.0)

cv2.imwrite("myans_73.jpg", result)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

