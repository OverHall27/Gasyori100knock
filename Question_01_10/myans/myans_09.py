import cv2
import numpy as np

def GaussianFilter(img, K_size=3, sigma=1.3):
    Hol, Ber, Col = img. shape
    
    ## zero padding
    result = np.zeros((Hol + 2, Ber + 2, Col), dtype=np.float)
    result[1: 1 + Hol, 1: 1 + Ber] = img.copy().astype(np.float)

    ## create kernel
    kernel = np.zeros((K_size, K_size), dtype=np.float)
    for x in range(-1, 2):
        for y in range(-1, 2):
            kernel[y + 1, x + 1] = np.exp( -(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    ##kernel /= ( (sigma ** 2) * (2. * np.pi))
    kernel /= ( sigma * np.sqrt(2. * np.pi))
    print(kernel)
    kernel /= kernel.sum()
    print(kernel)
    tmp = result.copy()

    for x in range(Hol):
        for y in range(Ber):
            for c in range(Col):
                result[x + 1, y + 1, c] = np.sum(kernel * tmp[x: x + K_size, y: y + K_size, c])
    result =  result[1: 1 + Hol, 1: 1 + Ber].astype(np.uint8)

    return result

img = cv2.imread("imori_noise.jpg")


result = GaussianFilter(img, K_size=3, sigma=1.3)
cv2.imwrite("myans_09.jpg", result)
