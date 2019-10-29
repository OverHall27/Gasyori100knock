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

def ZhangSuen_step1(img):
    Ver , Hor = img.shape
    img = np.pad(img, ([1, 1], [1, 1]), 'edge')
    _img = img.copy()

    dxs = [0, 1, 1, 1, 0, -1, -1, -1, 1]
    dys = [-1, -1, 0, 1, 1, 1, 0, -1, -1]
    changed = False

    for y in range(1, Ver+1):
        for x in range(1, Hor+1):
            C1 = False
            C2 = False
            C3 = False
            C4 = False

            # condition-1
            if img[y, x] == 1:
                _img[y, x] = 1
                C1 = True

            # condition-2
            #is_edge = (x == 1 or x == Hor+1) or (y == 1 or y == Ver+1)
            is_edge = False
            if not(C1 or is_edge):
                count = 0
                for i in range(8):
                    dt = img[y+dys[i+1], x+dxs[i+1]] - img[y+dys[i], x+dxs[i]]
                    if dt == 1:
                        count += 1
                if count == 1:
                    _img[y, x] = 1
                    C2 = True
                    changed = True

            # condition-3
            if not(C1 or C2 or is_edge):
                num_one = np.count_nonzero(img[y-1:y+2, x-1:x+2] == 1) - np.count_nonzero(img[y, x] == 1)
                if (num_one >= 2) & (num_one <= 6):
                    _img[y, x] = 1
                    C3 = True
                    changed = True

            # condition-4
            if not(C1 or C2 or C3 or is_edge):
                if any(item == 1 for item in [img[y-1, x], img[y, x+1], img[y+1, x]]):
                    _img[y, x] = 1
                    C4 = True
                    changed = True

            # condition-5
            if not(C1 or C2 or C3 or C4 or is_edge):
                if any(item == 1 for item in [img[y, x+1], img[y+1, x], img[y, x-1]]):
                    _img[y, x] = 1
                    changed = True

    return _img[1:Ver+1, 1:Hor+1], changed


def ZhangSuen_step2(img):
    Ver, Hor = img.shape
    img = np.pad(img, ([1, 1], [1, 1]), 'edge')
    _img = img.copy()

    dxs = [0, 1, 1, 1, 0, -1, -1, -1, 1]
    dys = [-1, -1, 0, 1, 1, 1, 0, -1, -1]
    changed = False
    
    for y in range(1, Ver+1):
        for x in range(1, Hor+1):

            C1 = False
            C2 = False
            C3 = False
            C4 = False
            # condition-1
            if img[y, x] == 1:
                _img[y, x] = 1
                C1 = True

            # condition-2
            #is_edge = (x == 1 or x == Hor+1) or (y == 1 or y == Ver+1)
            is_edge = False
            if not(C1 or is_edge):
                count = 0
                for i in range(8):
                    dt = img[y+dys[i+1], x+dxs[i+1]] - img[y+dys[i], x+dxs[i]]
                    if dt == 1:
                        count += 1
                if count == 1:
                    _img[y, x] = 1
                    C2 = True
                    changed = True

            # condition-3
            if not(C1 or C2 or is_edge):
                num_one = np.count_nonzero(img[y-1:y+2, x-1:x+2] == 1) - np.count_nonzero(img[y, x] == 1)

                if (num_one >= 2) & (num_one <= 6):
                    _img[y, x] = 1
                    C3 = True
                    changed = True

            # condition-4
            if not(C1 or C2 or C3 or is_edge):
                if any(item == 1 for item in [img[y-1, x], img[y, x+1], img[y, x-1]]):
                    _img[y, x] = 1
                    C4 = True
                    changed = True

            # condition-5
            if not(C1 or C2 or C3 or C4 or is_edge):
                if any(item == 1 for item in [img[y-1, x], img[y+1, x], img[y, x-1]]):
                    _img[y, x] = 1
                    changed = True

    return _img[1:Ver+1, 1:Hor+1], changed

def ItemizationZhangSuen(img):
    changed_step1 = False
    changed_step2 = False

    while True:

        img, changed_step1 = ZhangSuen_step1(img)
        print(changed_step1)
        if not(changed_step1):
            break

        img, changed_step2 = ZhangSuen_step2(img)
        print(changed_step2)
        if not(changed_step2):
            break

    return img


img = cv2.imread("../gazo.png", cv2.IMREAD_GRAYSCALE).astype(np.int)
img = nichika01(img, 128)
img = reverse01(img)

result = ItemizationZhangSuen(img)
result = reverse01(result)
print("finish process")
result = (result * 255).astype(np.uint8)

cv2.imwrite("myans_65.png", result)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

