import cv2
import numpy as np

def BGRtoGRAY(img):
    gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
    return gray

def HOG(gray):

    def GetGradXY(gray):
        Ver, Hor = gray.shape

        gray = np.pad(gray, (1, 1), 'edge')
        gx = gray[1:Ver+1, 2:] - gray[1:Ver+1, :Hor]
        #gy = gray[:Ver, 1:Hor+1] - gray[2:, 1:Hor+1]
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


    gx, gy = GetGradXY(gray)

    mag, ang = GetMagAngle(gx, gy)

    ang = QuantizationAngle(ang)

    return mag, ang

img = cv2.imread("../imori.jpg").astype(np.float)
gray = BGRtoGRAY(img)

mag, ang = HOG(gray)
mag = mag.astype(np.uint8)
ang = ang.astype(np.uint8)

cv2.imwrite("myans_66mag.jpg", mag)
cv2.imwrite("myans_66ang.jpg", ang)
cv2.imshow("mag", mag)
cv2.imshow("ang", ang)

cv2.waitKey(0)
cv2.destroyAllWindows()
