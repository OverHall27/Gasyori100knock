import cv2
import numpy as np

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

    def HSVTOBGR(_img, hsv):
        img = _img.copy() / 255.0

        # get max and min
        max_v = np.max(img, axis=2).copy()
        min_v = np.min(img, axis=2).copy()
        result = np.zeros_like(img)

        H = hsv[...,0]
        S = hsv[...,1]
        V = hsv[...,2]

        C = S
        H_ = H / 60.
        X = C * (1 - np.abs((H_ % 2) - 1))
        Z = np.zeros_like(H)

        ##BGRで変換するから注意
        vals = [[Z,X,C], [Z,C,X], [X,C,Z], [C,X,Z], [C,Z,X], [X,Z,C]]

        ##H(色相)は0 - 360までだから,H_は0 - 6まで
        for i in range(6):
            ind = np.where((i <= H_) & (H_ < (i+1)))
            result[...,0][ind] = (V - C)[ind] + vals[i][0][ind]
            result[...,1][ind] = (V - C)[ind] + vals[i][1][ind]
            result[...,2][ind] = (V - C)[ind] + vals[i][2][ind]

        result[np.where(max_v == min_v)] = 0
        ##result を 0-1に正規化
        result = np.clip(result, 0, 1)
        ##maxを255に戻す
        result = result * 255

        return result

    def GetMaskImg(hsv, color='blue'):
        Ver, Hor, _ = hsv.shape
        img = np.full((Ver, Hor), 1)

        if color == 'blue':
            hmin = 180
            hmax = 260
        for x in range(Hor):
            for y in range(Ver):
                if hmin <= hsv[y, x, 0] and hsv[y, x, 0] <= hmax:
                    img[y, x] = 0

        return img

    hsv = BGRTOHSV(img)

    mask = GetMaskImg(hsv, color)

    result = np.zeros_like(img)

    for c in range(3):
        result[..., c] = img[..., c] * mask

    return result
    
    
#BGRの順番で読み込んでいる
img = cv2.imread("../imori.jpg").astype(np.float32)

result = Masking(img, color='blue')
result = result.astype(np.uint8)
cv2.imwrite("myans_71.jpg", result)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

