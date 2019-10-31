import cv2
import numpy as np
import matplotlib.pyplot as plt

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

As = [0, 45, 90, 135]
for i, A in enumerate(As):
    # get gabor kernel
    gabor = GetGaborFilter(K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Phi=0, Angle=A)

    # normalize to [0, 255]
    out = gabor - np.min(gabor)
    out /= np.max(out)
    out *= 255
    
    out = out.astype(np.uint8)
    plt.subplot(1, 4, i+1)
    plt.imshow(out, cmap='gray')
    plt.axis('off')
    plt.title("Angle "+str(A))

plt.savefig("myans_78.png")
plt.show()
