import cv2
import numpy as np

# C = 1.0の理由は，ガンマ変換でIMGmaxで割っているから1.0,なのであり，画像を[0, 1]に正規化していなければ，c = Imaxである．

def GammaCorrection(_img, c=1.0, g=2.2):

    img = _img.copy().astype(np.float)
    Max = img.max()
    img /= Max

    img = (1.0 / c * img) ** (1.0 / g)

    img *= 255.

    return img.astype(np.uint8)

def AnchiLinearConversion(_img, c=1.0, g=2.2):
    
    img = _img.copy()
    img = c * (img ** g)
    img /= img.max()

    return img

img = cv2.imread("imori_gamma.jpg")
result = GammaCorrection(img, c=1.0, g=2.2)
cv2.imwrite("myans_24.jpg", result)

origin_img = cv2.imread("imori.jpg")
gamma_result = AnchiLinearConversion(origin_img, c=1.0, g=2.2)
cv2.imshow("result-1", gamma_result)

test_result = GammaCorrection(gamma_result, c=1.0, g=2.2)
cv2.imshow("result-2", test_result)

cv2.waitKey(0)
cv2.destroyAllWindows()
