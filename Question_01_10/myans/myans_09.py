import cv2
import numpy as np

def GaussianFilter(img, K_size=3, sigma=1.3):
    Ver, Hor, Col = img. shape
    
    ## zero padding
    pad = K_size // 2
    tmp = np.pad(img, [(pad,pad), (pad,pad), (0, 0)], 'constant')
    result = np.zeros_like(tmp)

    ## create kernel
    kernel = np.zeros((K_size, K_size), dtype=np.float)
    for x in range(-pad, pad+1):
        for y in range(-pad, pad+1):
            kernel[y + pad, x + pad] = np.exp( -(x ** 2 + y ** 2) / (2 * (sigma ** 2)))

    kernel /= ( sigma * np.sqrt(2. * np.pi))
    kernel /= kernel.sum()

    for x in range(pad, Hor+pad):
        for y in range(pad, Ver+pad):
            for c in range(Col):
                result[y, x, c] = np.sum(kernel * tmp[y-pad: y+pad+1, x-pad: x+pad+1, c])
    result =  result[pad: pad + Ver, pad: pad + Ver].astype(np.uint8)

    return result

img = cv2.imread("../imori_noise.jpg").astype(np.float)

result = GaussianFilter(img, K_size=3, sigma=1.3)
official = cv2.GaussianBlur(img, (3,3), 1.3)

cv2.imwrite("myans_09.jpg", result)
cv2.imwrite("cv2_09.jpg", official)
cv2.imshow("result", result)
cv2.imshow("official", official)
cv2.waitKey(0)
cv2.destroyAllWindows()
