import cv2
import numpy as np

def NearestNeighbor(_img, ax=1.0, ay=1.0):
    Hol, Ver, Col = _img.shape

    aH = int(Hol * ax)
    aV = int(Ver * ay)

    result = np.zeros((aH, aV, Col))

    for x in range(aH):
        x_ = int(np.round(x/ax))
        for y in range(aV):
            y_ = int(np.round(y/ay))

            result[x, y] = _img[x_, y_]

    return result


img = cv2.imread("imori.jpg")

result = NearestNeighbor(img, 1.5, 1.5)

cv2.imwrite("myans_25.jpg", result)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
