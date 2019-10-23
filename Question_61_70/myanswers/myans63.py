import cv2
import numpy as np


def nichika(img, th=128):
    ind0 = np.where(img < th)
    ind1 = np.where(img >= th)
    img[ind0] = 0
    img[ind1] = 1

    return img

def Itemization(img):
    Ver, Hor = img.shape

    img = np.pad(img, ([1, 1], [1, 1]), 'edge')

    count = 1
    while count != 0:
        count = 0
        for y in range(1, Ver+1):
            for x in range(1, Hor+1):
                if img[y, x] == 1:

                    # condition1
                    nearby4 = [img[y-1, x], img[y, x-1], img[y, x+1], img[y+1, x]] 
                    if any(pixel == 0 for pixel in nearby4):

                        # condition2
                        s1 = img[y, x+1] - (img[y, x+1] * img[y-1, x+1] * img[y-1, x])
                        s2 = img[y-1, x] - (img[y-1, x] * img[y-1, x-1] * img[y, x-1])
                        s3 = img[y, x-1] - (img[y, x-1] * img[y+1, x-1] * img[y+1, x])
                        s4 = img[y+1, x] - (img[y+1, x] * img[y+1, x+1] * img[y, x+1])
                        S = s1 + s2 + s3 + s4
                        if S == 1:

                            # condition3
                            one_count = np.count_nonzero(img[y-1:y+2, x-1:x+2] == 1) - np.count_nonzero(img[y, x] == 1)
                            if (one_count >= 3):
                                img[y, x] = 0
                                count += 1

    return img[1:Ver+1, 1:Hor+1]
   

img = cv2.imread("../gazo.png", cv2.IMREAD_GRAYSCALE)

img = nichika(img, 128)

result = Itemization(img)
result = result * 255

cv2.imwrite("myans_63.png", result)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
