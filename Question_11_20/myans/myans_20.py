import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("imori_dark.jpg").astype(np.float)
##plt.hist(img.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.hist(img.ravel())
plt.savefig("myans_20.png")
plt.show()
