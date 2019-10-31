import cv2
import numpy as np
import matplotlib.pyplot as plt

def BGRtoGRAY(img):
    gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]

    return gray

def GaborFiltering(img, K_size=111, Sigma=10., Gamma=1.2, Lambda=10., Phi=0., Angle=0.):

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

    Ver, Hor = img.shape
    result = np.zeros((Ver, Hor))
    img = np.pad(img, (d, d), 'edge')

    for x in range(d, Hor+d):
        for y in range(d, Ver+d):
            result[y-d, x-d] = np.sum(img[y-d:y+d+1, x-d:x+d+1] * gabor)

    result = np.clip(result, 0, 255)
    return result

img = cv2.imread("../imori.jpg").astype(np.float32)
gray = BGRtoGRAY(img)
As = [0, 45, 90, 135]

plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0.2)
for i, angle in enumerate(As):
    gabor = GaborFiltering(gray, K_size=11, Sigma=1.5, Gamma=1.2, Lambda=3, Phi=0, Angle=angle)

    gabor= gabor - np.min(gabor)
    gabor = gabor * 255 / np.max(gabor)
    gabor = gabor.astype(np.uint8)

    plt.subplot(1, 4, i+1)
    plt.imshow(gabor, cmap='gray')
    plt.axis('off')
    plt.title("Angle "+str(angle))

plt.savefig("myans_79.png")
plt.show()
