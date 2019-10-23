import cv2
import numpy as np

def reverse01(img):

    ind0 = np.where(img == 0)
    ind1 = np.where(img == 1)
    img[ind0] = 1
    img[ind1] = 0

    return img

def nichika(img, th=128):
    ind0 = np.where(img < th)
    ind1 = np.where(img >= th)
    img[ind0] = 0
    img[ind1] = 1

    return img

def Renketsu(img):
    Ver, Hor = img.shape
    img = np.pad(img, ([1, 1], [1, 1]), 'edge')
    result = np.zeros((Ver+1, Hor+1, 3))

    for x in range(1, Hor+1):
        for y in range(1, Ver+1):
            if img[y, x] != 0:
                tmp = img.copy()
                tmp = reverse01(tmp)
                s1 = tmp[y, x+1] - (tmp[y, x+1] * tmp[y-1, x+1] * tmp[y-1, x])
                s2 = tmp[y-1, x] - (tmp[y-1, x] * tmp[y-1, x-1] * tmp[y, x-1])
                s3 = tmp[y, x-1] - (tmp[y, x-1] * tmp[y+1, x-1] * tmp[y+1, x])
                s4 = tmp[y+1, x] - (tmp[y+1, x] * tmp[y+1, x+1] * tmp[y, x+1])
                S = s1 + s2 + s3 + s4

                if S == 0:
                   result[y,x] = [0, 0, 255]
                elif S == 1:
                   result[y,x] = [0, 255, 0]
                elif S == 2:
                   result[y,x] = [255, 0, 0]
                elif S == 3:
                   result[y,x] = [255, 255, 0]
                elif S == 4:
                   result[y,x] = [255, 0, 255]

    return result[1:1+Ver, 1:1+Hor]


img = cv2.imread("../renketsu.png", cv2.IMREAD_GRAYSCALE)

img = nichika(img, 128)

result = Renketsu(img)

cv2.imwrite("myans_62.png", result)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

