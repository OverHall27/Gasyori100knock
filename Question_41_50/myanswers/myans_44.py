import cv2
import numpy as np

def HoughTransform(img):

    def BGRTOGRAY(img):
        gray_img = img[..., 2] * 0.2126 + img[..., 1] * 0.7152 + img[..., 0] * 0.0722
        gray_img = gray_img.astype(np.uint8)

        return gray_img


    gray = BGRTOGRAY(img)
    edge = cv2.Canny(gray, threshold1=30., threshold2=100., apertureSize=3, L2gradient=True)

    Ver, Hor = edge.shape
    diagonal = int(np.round(np.sqrt(Ver ** 2 + Hor ** 2)))

    r_table = np.zeros((diagonal, 180), dtype=np.uint8)
    ind = np.where(edge != 0)
    x = ind[0]
    y = ind[1]

    for t in range(180):
        it = np.pi * t / 180
        r = x * np.cos(it) + y * np.sin(it)
        for i in range(len(ind[0])):
            ir = int( np.round(r[i]) )
            r_table[ir, t] += 1

    return r_table

def HoughTransform2(img):

    def BGRTOGRAY(img):
        gray_img = img[..., 2] * 0.2126 + img[..., 1] * 0.7152 + img[..., 0] * 0.0722
        gray_img = gray_img.astype(np.uint8)

        return gray_img

    gray = BGRTOGRAY(img)
    edge = cv2.Canny(gray, threshold1=30., threshold2=100., apertureSize=3, L2gradient=True)

    hough = cv2.HoughLinesP(edge, 1, np.pi/360, 10)

    return hough

img = cv2.imread("../thorino.jpg")

result = HoughTransform(img)

cv2.imwrite("myans_44.jpg", result)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
