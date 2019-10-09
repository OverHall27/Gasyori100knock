import cv2
import numpy as np

def MedianFilter(img, K_size=3):
    Hol, Ber, Col = img. shape
    
    ## zero padding
    result = np.zeros((Hol + 2, Ber + 2, Col), dtype=np.float)
    result[1: 1 + Hol, 1: 1 + Ber] = img.copy().astype(np.float)
    temp = result.copy()

    for x in range(Hol):
        for y in range(Ber):
            for c in range(Col):
                result[x + 1, y + 1, c] = np.median(temp[x: x + K_size, y: y + K_size, c])

    result =  result[1: 1 + Hol, 1: 1 + Ber].astype(np.uint8)

    return result

img = cv2.imread("imori_noise.jpg")

result = MedianFilter(img, K_size=3)
cv2.imwrite("myans_10.jpg", result)
