import cv2
import numpy as np
import matplotlib.pyplot as plt

def BGRtoGRAY(img):
    gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
    return gray

def HOG(gray):

    def GetGradXY(gray):
        Ver, Hor = gray.shape

        gray = np.pad(gray, (1, 1), 'edge')
        gx = gray[1:Ver+1, 2:] - gray[1:Ver+1, :Hor]
        gy = gray[2:, 1:Hor+1] - gray[:Ver, 1:Hor+1]
        # keep from zero-dividing
        gx[gx == 0] = 1e-6

        return gx, gy

    def GetMagAngle(gx, gy):
        mag = np.sqrt(gx ** 2 + gy ** 2)
        ang = np.arctan(gy / gx)

        # arctanは返り値(-2/pi, 2/pi)なので，負の値はpi回転
        ang[ang < 0] += np.pi

        return mag, ang

    def QuantizationAngle(ang):

        # angはradianなので,[0, 180]のindex化を調節
        quantized_ang = np.zeros_like(ang, dtype=np.int)
        for i in range(9):
            low = (np.pi * i) / 9
            high = (np.pi * (i + 1)) / 9

            quantized_ang[np.where((ang >= low) & (ang < high))] = i + 1

        return quantized_ang

    def GetSlopeHistgram(mag, ang, sell=8):
        Ver, Hor = mag.shape

        CELL_H = Hor // sell
        CELL_V = Ver // sell
        hist = np.zeros((CELL_V, CELL_H, 9), dtype=np.float32)

        for x in range(Hor):
            hx = x // sell
            for y in range(Ver):
                hy = y // sell

                hist[hy, hx, ang[y, x]-1] += mag[y, x]

        return hist

    gx, gy = GetGradXY(gray)

    mag, ang = GetMagAngle(gx, gy)

    ang = QuantizationAngle(ang)

    hist = GetSlopeHistgram(mag, ang, sell=8)

    return hist

img = cv2.imread("../imori.jpg").astype(np.float)
gray = BGRtoGRAY(img)

histogram = HOG(gray)

for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(histogram[..., i])
    plt.axis('off')
    plt.xticks(color="None")
    plt.yticks(color="None")
plt.savefig("out.png")
plt.show()

