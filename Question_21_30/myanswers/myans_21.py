import cv2
import numpy as np
import matplotlib.pyplot as plt

def GrayScaleTransform(_img, Low, High):
    Max = np.max(_img)
    Min = np.min(_img)

    Hol, Ver, Col = _img.shape
    img = _img.copy()

    img = (High - Low) / (Max - Min) * (img - Min) + Low
    img[img < Low] = Low
    img[img > High] = High

    return img.astype(np.uint8)

img = cv2.imread("imori_dark.jpg").astype(np.float)
H, W, C = img.shape

result = GrayScaleTransform(img, 0, 255)

plt.hist(result.ravel(), bins=255, range=(0, 255))
plt.show()
plt.savefig("myans_21-2.jpg")

cv2.imwrite("myans_21.jpg", result)
