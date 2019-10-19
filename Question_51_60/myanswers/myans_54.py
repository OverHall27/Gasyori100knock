import cv2
import numpy as np

def k_largest_index_argsort(a, k):
    idx = np.argsort(a.ravel())[:-k-1:-1]
    return np.column_stack(np.unravel_index(idx, a.shape))

def k_smallest_index_argsort(a, k):
    # ravel -> listに
    # sortは単純sort, argsortはsortしたときのindex
    # どちらも昇順
    idx = np.argsort(a.ravel())[0:k]
    # np.unravel_index(idx, a)で水平スキャン順でidx番目のの座標を取得できる. (array0, array1)の返り値
    # column_stack(a, b) でa,bそれぞれの第N要素を座標とするN個のndarryを返す
    return np.column_stack(np.unravel_index(idx, a.shape))



def TemplateMatch(obj_img, temp_img):

    Ver, Hor, Col = obj_img.shape
    tVer, tHor, tCol = temp_img.shape

    ssd = np.zeros((Ver - tVer, Hor - tHor), dtype=np.float)
    for sx in range(Hor - tHor):
        for sy in range(Ver - tVer):
            ssd[sy, sx] = np.sum(np.power(obj_img[sy:sy+tVer, sx:sx+tHor] - temp_img, 2))

    min_idx = k_smallest_index_argsort(ssd, 1)
    # imgに(0, 0)->(width-1, height-1)の範囲で赤い線を引く，太さは10
    # lineじゃなくでrectangleにすれば四角
    # cv2.line(img, (0, 0), (width-1, height-1), (0, 0, 255), 10)

    iy = min_idx[0][0]
    ix = min_idx[0][1]
    obj_img = obj_img.astype(np.uint8)
    result = cv2.rectangle(obj_img, (ix, iy), (ix+tHor, iy+tVer), (0,0,255), 1)

    return result

obj = cv2.imread("../imori.jpg").astype(np.float32)
temp = cv2.imread("../imori_part.jpg").astype(np.float32)

result = TemplateMatch(obj, temp)

cv2.imwrite("myans_54.jpg", result)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
