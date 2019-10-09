import cv2
import numpy as np
import matplotlib.pyplot as plt

def HistManiplation(img, mean=128, sigma=52):

    pr_sigma = np.std(img)
    pr_mean  = np.mean(img)

    img = (sigma / pr_sigma) * (img - pr_mean) + mean

    return img


img = cv2.imread("imori_dark.jpg")

result = HistManiplation(img, mean=128, sigma=52)

plt.hist(result.ravel(), bins=255, range=(0, 255))
plt.show()
plt.savefig("myans_22_hist.jpg")

cv2.imwrite("myans_22.jpg", result)
cv2.imshow("result", result)
waitKey(0)

