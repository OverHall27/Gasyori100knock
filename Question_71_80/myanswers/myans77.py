import cv2
import numpy as np

def GetGaborFilter(K_size=111, Sigma=10., Gamma=1.2, Lambda=10., Phi=0., Angle=0.):

    gabor = np.zeros((K_size, K_size), dtype=np.float32)
    d = K_size // 2
    for x in range(K_size):
        px = x - d
        for y in range(K_size):
            py = y - d
            theta = Angle / 180 * np.pi

            gx = px * np.cos(theta) + py * np.sin(theta)
            gy = px * -np.sin(theta) + py * np.cos(theta)
            gabor[y, x] = np.exp(-(gx ** 2 + Gamma ** 2 * gy ** 2) / (2 * Sigma ** 2)) * np.cos(2 * np.pi * gx / Lambda + Phi)
    gabor /= np.sum(np.abs(gabor))

    return gabor

gabor = GetGaborFilter(K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Phi=0, Angle=0)

gabor = gabor - np.min(gabor)
gabor = gabor * 255 / np.max(gabor)
gabor = gabor.astype(np.uint8)

cv2.imshow("result", gabor)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("myans_77.jpg", gabor)

