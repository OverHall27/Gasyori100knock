import cv2
import numpy as np

def Canny_step1(img):

    def BGRTOGRAY(img):
        gray_img = img[..., 2] * 0.2126 + img[..., 1] * 0.7152 + img[..., 0] * 0.0722

        return gray_img

    def GaussianFilter(img, K_size=3, sigma=1.3):
        if len(img.shape) == 3:
            Ver, Hor, Col = img.shape
        else:
            img = np.expand_dims(img, axis=-1)
            Ver, Hor, Col = img.shape

        pad = K_size // 2
        tmp = np.pad(img, [(pad, pad), (pad, pad), (0, 0)], mode='constant')
        kernel = np.zeros((K_size, K_size), dtype=np.float)

        for x in range(-pad, pad+1):
            for y in range(-pad, pad+1):
                kernel[y + pad, x + pad] = np.exp( -(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
        kernel /= ( sigma * np.sqrt(2. * np.pi))
        kernel /= kernel.sum()

        result = np.zeros_like(tmp)
        for c in range(Col):
            for x in range(pad, pad+Hor):
                for y in range(pad, pad+Ver):
                    result[y, x, c] = np.sum(kernel * tmp[y-pad:y+pad+1, x-pad:x+pad+1, c])
        result =  result[pad: pad + Hor, pad: pad + Ver]

        result = np.clip(result, 0, 255).astype(np.uint8)

        return result[..., 0]

    gray = BGRTOGRAY(img)

    gaussian = GaussianFilter(gray, K_size=5, sigma=1.4)

    return gaussian


img = cv2.imread("../imori.jpg")
gaussian = Canny_step1(img)

cv2.imwrite("myans_41.jpg", gaussian)
cv2.imshow("Gaussian", gaussian)
cv2.waitKey(0)
cv2.destroyAllWindows()
