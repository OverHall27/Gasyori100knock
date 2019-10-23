import cv2
import numpy as np

def nichika01(img, th=128):
    ind0 = np.where(img < th)
    ind1 = np.where(img >= th)
    img[ind0] = 0
    img[ind1] = 1

    return img

def reverse01(img):

    ind0 = np.where(img == 0)
    ind1 = np.where(img == 1)
    img[ind0] = 1
    img[ind1] = 0

    return img

def connect8(img, x, y):

    tmp = img.copy()
    tmp = reverse01(tmp)

    s1 = (tmp[y, x+1] - (tmp[y, x+1] * tmp[y-1, x+1] * tmp[y-1, x]))
    s2 = (tmp[y-1, x] - (tmp[y-1, x] * tmp[y-1, x-1] * tmp[y, x-1]))
    s3 = (tmp[y, x-1] - (tmp[y, x-1] * tmp[y+1, x-1] * tmp[y+1, x]))
    s4 = (tmp[y+1, x] - (tmp[y+1, x] * tmp[y+1, x+1] * tmp[y, x+1]))

    S = s1 + s2 + s3 + s4
    return S

def SearchNearby8(img, x, y, item):

    dxs = [1, 1, 0, -1, -1, -1, 0, 1]
    dys = [0, -1, -1, -1, 0, 1, 1, 1]

    return any(img[y+dys[i], x+dxs[i]] == item for i in range(8))

def AroundConnent8(img, x, y):

    dxs = [1, 1, 0, -1, -1, -1, 0, 1]
    dys = [0, -1, -1, -1, 0, 1, 1, 1]
    aroundS = np.zeros(8)

    for i in range(8):
        tmp = img.copy()
        tmp[y+dys[i], x+dxs[i]] = 0
        aroundS[i] = connect8(tmp, x, y)

    return all(aroundS == 1)

def ItemizationHilditch(img):
    Ver, Hor = img.shape

    img = np.pad(img, ([1, 1], [1, 1]), 'edge')

    count = 1
    while count != 0:
        count = 0
        for y in range(1, Ver+1):
            for x in range(1, Hor+1):

                # 背景画像ではない
                if img[y, x] == 1:

                    # condition-1(境界点である)
                    nearby4 = [img[y-1, x], img[y, x-1], img[y, x+1], img[y+1, x]]
                    if any(pixel == 0 for pixel in nearby4):

                        # condition-2(
                        S = connect8(img, x, y)
                        if S == 1:

                            # condition-3(端点を削除しない)
                            if np.sum(np.abs(img[y-1:y+2, x-1:x+2])) - np.abs(img[y, x]) >= 2:

                                # condition-4(孤立点を削除しない,8近傍に1(前景画像)がある)
                                if SearchNearby8(img, x, y, 1):

                                    # condition-5(線幅2の線分の片側だけを削除)
                                    condition_51 = not(SearchNearby8(img, x, y, -1))
                                    condition_52 = AroundConnent8(img, x, y)
                                    if condition_51 ^ condition_52:
                                        img[y, x] = -1
                                        count += 1
        indm1 = np.where(img == -1)
        if(len(indm1[0]) > 0):
            img[indm1] = 0
        else:
            print("ok")

    return img[1:Ver+1, 1:Hor+1]

img = cv2.imread("../gazo.png", cv2.IMREAD_GRAYSCALE).astype(np.int)
img = nichika01(img, 128)

result = ItemizationHilditch(img)
result = result * 255
result = result.astype(np.uint8)

cv2.imwrite("myans_64.png", result)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

