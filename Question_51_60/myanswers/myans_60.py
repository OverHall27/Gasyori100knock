import cv2
import numpy as np

def AlphaBlend(img1, img2, alpha):

    blendS = img1 * alpha + img2 * (1. - alpha)

    return blendS

img1 = cv2.imread("../imori.jpg").astype(np.float)
img2 = cv2.imread("../thorino.jpg").astype(np.float)

blendS = AlphaBlend(img1, img2, 0.6)
blendS = blendS.astype(np.uint8)

cv2.imwrite("myans_60.jpg", blendS)
cv2.imshow("result", blendS)
cv2.waitKey(0)
cv2.destroyAllWindows()
