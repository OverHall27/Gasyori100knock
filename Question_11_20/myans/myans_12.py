import cv2
import numpy as np

def MotionFilter(img, K_size=3):
    Hol, Ber, Col = img. shape
    
    ## zero padding
    result = np.zeros((Hol + 2, Ber + 2, Col), dtype=np.float)
    temp = result.copy()
    temp[1: 1 + Hol, 1: 1 + Ber] += img.copy().astype(np.float)

    ## create kernel
    kernel = np.zeros((K_size, K_size), dtype=np.float)
    for x in range(-1, 2):
        kernel[x + 1, x + 1] = 1.
    kernel /= 3.

    for x in range(Hol):
        for y in range(Ber):
            for c in range(Col):
                result[1 + y, 1 + x, c] = np.sum(kernel * temp[y: y + K_size, x: x + K_size, c])
    result =  result[1: 1 + Hol, 1: 1 + Ber].astype(np.uint8)

    return result

img = cv2.imread("imori.jpg")

result = MotionFilter(img, K_size=3)
cv2.imwrite("myans_12.jpg", result)
