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

def TemplateMatchZNCC(obj_img, temp_img):

    Ver, Hor, Col = obj_img.shape
    tVer, tHor, tCol = temp_img.shape

    temp_img = temp_img - np.mean(temp_img, axis=(0, 1))

    ssd = np.zeros((Ver - tVer, Hor - tHor), dtype=np.float)
    for sx in range(Hor - tHor):
        for sy in range(Ver - tVer):
            obj_mean = np.mean(obj_img[sy:sy+tVer, sx:sx+tHor], axis=(0, 1))

            ssd[sy, sx] += np.sum((obj_img[sy:sy+tVer, sx:sx+tHor] - obj_mean) * temp_img)
            ssd[sy, sx] /= np.sqrt(np.sum((obj_img[sy:sy+tVer, sx:sx+tHor] - obj_mean) ** 2))
            ssd[sy, sx] /= np.sqrt(np.sum(temp_img ** 2))

    max_idx = k_largest_index_argsort(ssd, 1)
    print(np.max(ssd))

    iy = max_idx[0][0]
    ix = max_idx[0][1]
    obj_img = obj_img.astype(np.uint8)
    # rectangleでの描写位置は(x, y)指定
    result = cv2.rectangle(obj_img, (ix, iy), (ix+tHor, iy+tVer), (0,0,255), 1)

    return result

obj = cv2.imread("../imori.jpg").astype(np.float32)
temp = cv2.imread("../imori_part.jpg").astype(np.float32)

result = TemplateMatchZNCC(obj, temp)
cv2.imwrite("myans_57.jpg", result)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
