import cv2
import numpy as np

def Labeling8(img):

    Ver, Hor = img.shape

    img = np.pad(img, ([1, 1], [1, 1]), 'constant')
    label = np.zeros_like(img)
    lu_table = np.zeros((Ver * Hor) // 2, dtype=np.uint8)

    dist = 1
    kernel = [[1, 1, 1], [1, 0, 0]]
    # 8近傍なので上側と左のみ注意すればいい
    # img[y, x]周りのラベリングでlook-up-tableに書く必要が出てくるのは, 右上と左の組み合わせ
    for y in range(1, Ver+1):
        for x in range(1, Hor+1):

            if img[y ,x] != 0:

                # 左上に画素あり
                if img[y-1, x-1] != 0:
                    label[y, x] = label[y-1, x-1].copy()

                # 上に画像あり
                elif img[y-1, x] != 0:
                    label[y, x] = label[y-1, x].copy()
                    
                # 右上あり
                elif img[y-1, x+1] != 0:
                    label[y, x] = label[y-1, x+1].copy()
                    if img[y, x-1] != 0:
                        lu_table[label[y, x-1].copy()] = label[y-1, x+1].copy()

                elif (img[y, x-1] != 0) and (img[y-1, x+1] == 0):
                    label[y, x] = label[y, x-1].copy()

                else:
                    label[y, x] = dist
                    dist += 1

    for i in range(lu_table.size):
        
        li = lu_table.size - i - 1
        if lu_table[li] != 0:
            clabel = lu_table[li].copy()
            label[label == li] = clabel
            
    return label[1:Ver+1, 1:Hor+1]

img = cv2.imread("../seg.png", cv2.IMREAD_GRAYSCALE).astype(np.int)
result = Labeling8(img)

# 見えやすくしたい
result = result.astype(np.float)
result /= (np.max(result) / 255.)
result = result.astype(np.uint8)

cv2.imwrite("myans_59.png", result)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
