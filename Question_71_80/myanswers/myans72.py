import cv2
import numpy as np

def Mofology(binary, times=1, mode='opening'):

    def expansion(binary):
        Ver, Hor = binary.shape

        fileter = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
        tmp = np.pad(binary, [(1, 1), (1, 1)], mode='edge')
        binary = np.pad(binary, [(1, 1), (1, 1)], mode='constant')

        for x in range(1, Hor+1):
            for y in range(1, Ver+1):
                if np.sum(fileter * tmp[y-1:y+2, x-1:x+2]) >= 1:
                    binary[y, x] = 1

        return binary[1:Ver+1, 1:Hor+1]

    def shrink(binary):
        Ver, Hor = binary.shape

        fileter = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
        tmp = np.pad(binary, [(1, 1), (1, 1)], mode='edge')
        binary = np.pad(binary, [(1, 1), (1, 1)], mode='constant')

        for x in range(1, Hor+1):
            for y in range(1, Ver+1):
                if np.sum(fileter * tmp[y-1:y+2, x-1:x+2]) < 1 * 4:
                    binary[y, x] = 0

        return binary[1:Ver+1, 1:Hor+1]

    if mode == 'opening':
        for i in range(times):
            binary = shrink(binary)
        for i in range(times):
            binary = expansion(binary)

        return binary

    elif mode == 'closing':
        for i in range(times):
            binary = expansion(binary)
        for i in range(times):
            binary = shrink(binary)

        return binary
 
def Masking(img, color='blue'):

    def BGRTOHSV(_img):
        img = _img.copy() / 255.0

        hsv = np.zeros_like(img, dtype=np.float32)
        ## axis=2 で 0-2要素まで最大最小値を計算し，それを配列で返している．
        max_v = np.max(img, axis=2).copy()
        min_v = np.min(img, axis=2).copy() 
        #axisは軸指定，今回imgは入れ子だから，axis3まで考えられる
        min_arg = np.argmin(img, axis=2)

        ##pythonではforしなくても処理できる表記が備わっている
        hsv[...,0][np.where(max_v == min_v)] = 0
        ind = np.where(min_arg == 0)
        hsv[...,0][ind] = 60 * (img[...,1][ind] - img[...,2][ind]) / (max_v[ind] - min_v[ind]) + 60

        ind = np.where(min_arg == 2)
        hsv[...,0][ind] = 60 * (img[...,0][ind] - img[...,1][ind]) / (max_v[ind] - min_v[ind]) + 180

        ind = np.where(min_arg == 1)
        hsv[...,0][ind] = 60 * (img[...,2][ind] - img[...,0][ind]) / (max_v[ind] - min_v[ind]) + 300

        hsv[...,1] = max_v.copy() - min_v.copy()
        hsv[...,2] = max_v.copy()

        return hsv

    def GetMaskImg(hsv, color='blue'):
        Ver, Hor, _ = hsv.shape
        mask = np.zeros((Ver, Hor))

        if color == 'blue':
            hmin = 180
            hmax = 260
        for x in range(Hor):
            for y in range(Ver):
                if (hmin < hsv[y, x, 0]) & (hsv[y, x, 0] < hmax):
                    mask[y, x] = 1

        return mask

    hsv = BGRTOHSV(img)

    mask = GetMaskImg(hsv, color)

    mask = Mofology(mask, 5, 'closing')
    mask = Mofology(mask, 5, 'opening')

    mask = 1 - mask

    result = np.zeros_like(img)

    for c in range(3):
        result[..., c] = img[..., c] * mask

    return result
   
img = cv2.imread("../imori.jpg").astype(np.float32)

result = Masking(img, color='blue')
result = result.astype(np.uint8)

cv2.imwrite("myans_72.jpg", result)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

