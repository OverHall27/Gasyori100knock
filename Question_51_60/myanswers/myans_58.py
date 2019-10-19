import cv2
import numpy as np

def Labeling(img, near=4):

    Ver, Hor = img.shape

    img = np.pad(img, ([1, 1], [1, 1]), 'constant')
    label = np.zeros_like(img)
    lu_table = np.zeros((Ver * Hor) // 2, dtype=np.uint8)

    dist = 1
    for y in range(1, Ver+1):
        for x in range(1, Hor+1):

            if img[y ,x] != 0:

                # 上だけ画素あり
                if (img[y-1, x] != 0) and (img[y, x-1] == 0):
                    label[y, x] = label[y-1, x].copy()
                    
                # 両方に画素あり
                elif (img[y-1, x] != 0) and (img[y, x-1] != 0):
                    label[y, x] = label[y-1, x].copy()
                    if label[y-1, x] != label[y, x-1]:
                        lu_table[label[y, x-1].copy()] = label[y-1, x].copy()

                # 左だけ画素あり
                elif (img[y-1, x] == 0) and (img[y, x-1] != 0):
                    label[y, x] = label[y, x-1].copy()

                #上下どちらも画素なし
                elif (img[y-1, x] == 0) and (img[y, x-1] == 0):
                    label[y, x] = dist
                    dist += 1

    for i in range(lu_table.size):
        
        li = lu_table.size - i - 1
        if lu_table[li] != 0:
            clabel = lu_table[li].copy()
            label[label == li] = clabel
            
    return label[1:Ver+1, 1:Hor+1]

img = cv2.imread("../seg.png", cv2.IMREAD_GRAYSCALE).astype(np.int)
result = Labeling(img, 4)

# 見えやすくしたい
result = result.astype(np.float)
result /= (np.max(result) / 255.)
result = result.astype(np.uint8)

'''
nLabels, labelImages = cv2.connectedComponents(img)
cv2.imshow("cv2", labelImages.astype(np.uint8))
'''

cv2.imwrite("myans_58.png", result)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
