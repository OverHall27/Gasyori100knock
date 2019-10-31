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
        Col = 1

    aH = int(ax * Hor)
    aV = int(ay * Ver)

    # x, y がオリジナル画像での画素位置を示すものになる

    y = np.arange(aV).repeat(aH).reshape(aV, -1)
    x = np.tile(np.arange(aH), (aV, 1))
    y = y / ay
    x = x / ax

    ix = np.floor(x).astype(np.int)
    iy = np.floor(y).astype(np.int)

    ix = np.minimum(ix, Hor-2)
    iy = np.minimum(iy, Ver-2)

    dx = (x - ix)
    dy = (y - iy)
 	
    if Col > 1:
        dx = np.repeat(np.expand_dims(dx, axis=-1), Col, axis=-1)
        dy = np.repeat(np.expand_dims(dy, axis=-1), Col, axis=-1)

    result = (1.-dx) * (1.-dy) * img[iy, ix] + dx * (1.-dy) * img[iy, ix+1] + (1.-dx) * dy * img[iy+1, ix] + dx * dy * img[iy+1, ix+1]

    result = np.clip(result, 0, 255)

    return result

img = cv2.imread("../imori.jpg").astype(np.float)
gray = BGRtoGRAY(img)

result = BilinearNeighbor(gray, 0.5, 0.5)
result = BilinearNeighbor(result, 2.0, 2.0)

result = np.abs(result - gray)

result = result / result.max() * 255

result = result.astype(np.uint8)

cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("myans_74.jpg", result)
