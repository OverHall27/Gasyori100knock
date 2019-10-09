import cv2
import numpy as np
import matplotlib.pyplot as plt

def HistEqualization(img):
    print(img)
    S = img.size
    Zmax = np.max(img)
    result = img.copy()

    h_z = 0
    for z in range(255):
        ind = np.where(img == z)
        h_z += img[img == z].size
        result[ind] = Zmax / S * h_z

    return result

img = cv2.imread("imori_dark.jpg")
plt.hist(img.ravel(), bins=255, range=(0, 255))
plt.show()

result = HistEqualization(img)
plt.hist(result.ravel(), bins=255, range=(0, 255))
plt.show()
plt.savefig("myans_23_1.jpg")

cv2.imwrite("myans_23_2.jpg", result)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
