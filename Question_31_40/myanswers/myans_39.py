import cv2
import numpy as np

IMG_V = 128
IMG_H = 128
channel = 3
def BGRtoYCbCr(img):

    Y = 0.299 * img[..., 2] + 0.5870 * img[..., 1] + 0.114 * img[..., 0]
    Cb = -0.1687 * img[..., 2] - 0.3313 * img[..., 1] + 0.5 * img[..., 0] + 128.
    Cr = 0.5 * img[..., 2] - 0.4187 * img[..., 1] - 0.0813 * img[..., 0] + 128.

    return Y, Cb, Cr

def YCbCrtoBGR(Y, Cb, Cr):

    R = Y + (Cr - 128) * 1.402
    G = Y - (Cb - 128) * 0.3441 - (Cr - 128) * 0.7139
    B = Y + (Cb - 128) * 1.7718

    result = np.zeros((IMG_V, IMG_H, channel)).astype(np.float)

    result[..., 0] = B
    result[..., 1] = G
    result[..., 2] = R

    return result

img = cv2.imread("../imori.jpg").astype(np.float)
Y, Cb, Cr = BGRtoYCbCr(img)
Y = Y * 0.7
result = YCbCrtoBGR(Y, Cb, Cr)

cv2.imwrite("myans_39.jpg", result)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
