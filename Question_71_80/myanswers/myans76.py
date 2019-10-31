import cv2
import numpy as np

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

def SaliencyMap(imgs):
    Ver, Hor = imgs[0].shape

    result = np.zeros((Ver, Hor))

    result += np.abs(imgs[0] - imgs[1])
    result += np.abs(imgs[0] - imgs[3])
    result += np.abs(imgs[0] - imgs[5])
    result += np.abs(imgs[1] - imgs[4])
    result += np.abs(imgs[2] - imgs[3])
    result += np.abs(imgs[3] - imgs[5])

    result = result / result.max() * 255

    return result

A = [2, 4, 8, 16, 32]
imgs = np.zeros((6, 128, 128))
imgs[0] = cv2.imread("../imori.jpg", cv2.IMREAD_GRAYSCALE)

for a in range(len(A)):
    fname = "myans_" + str(A[a]) + ".jpg"
    img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    img = BilinearNeighbor(img, A[a], A[a])
    imgs[a+1] = img

result = SaliencyMap(imgs)
result = result.astype(np.uint8)

cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("myans_76.jpg", result)


