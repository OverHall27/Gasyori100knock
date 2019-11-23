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
        #gy = gray[:Ver, 1:Hor+1] - gray[2:, 1:Hor+1]
        gy = gray[2:, 1:Hor+1] - gray[:Ver, 1:Hor+1]
        # keep from zero-dividing
        gx[gx == 0] = 1e-6

        return gx, gy

    def GetMagAngle(gx, gy):
        mag = np.sqrt(gx ** 2 + gy ** 2)
        ang = np.arctan(gy / gx)

        # arctanは返り値(-2/pi, 2/pi)なので，負の値はpi回転
        ang[ang < 0] = np.pi + ang[ang < 0]

        return mag, ang

    def QuantizationAngle(ang):

        # angはradianなので,[0, 180]のindex化を調節
        quantized_ang = np.zeros_like(ang, dtype=np.int)
        for i in range(9):
            low = (np.pi * i) / 9
            high = (np.pi * (i + 1)) / 9

            quantized_ang[np.where((ang >= low) & (ang < high))] = i

        return quantized_ang

    def GetSlopeHistgram(mag, ang, sell=8):
        Ver, Hor = mag.shape

        CELL_H = Hor // sell
        CELL_V = Ver // sell
        N = sell
        hist = np.zeros((CELL_V, CELL_H, 9), dtype=np.float32)
        for y in range(CELL_V):
            for x in range(CELL_H):
                for j in range(N):
                    for i in range(N):
                        hist[y, x, ang[y * N + j, x * N + i]] += mag[y * N + j, x * N + i]
        
        '''
        for x in range(Hor):
            hx = x // sell
            for y in range(Ver):
                hy = y // sell

                hist[hy, hx, ang[y, x]-1] += mag[y, x]
        '''
        return hist

    def QuantizationHistogram(histogram, epsilon=1):
        Ver, Hor, Index = histogram.shape

        for x in range(Hor):
            for y in range(Ver):
                histogram[y, x] = histogram[y, x] / np.sqrt(np.sum(histogram[max(y-1, 0) : min(y+2, Ver), max(x-1, 0) : min(x+2, Hor)] ** 2) + epsilon)

        return hist

    gx, gy = GetGradXY(gray)

    mag, ang = GetMagAngle(gx, gy)

    ang = QuantizationAngle(ang)

    hist = GetSlopeHistgram(mag, ang, sell=8)

    hist = QuantizationHistogram(hist, epsilon=1)

    return hist


def DrawHOG(hist, gray, sell=8):
    CELL_V, CELL_H, Index = hist.shape
    gray = np.pad(gray, (1, 1), 'constant')

    for x in range(CELL_H):
        # ix : 各sellの中心x
        ix = x * sell + sell // 2
        # x1,x2は各セルの左端右端
        x1 = ix + sell // 2 - 1
        x2 = ix - sell // 2 + 1
        for y in range(CELL_V):
            # iy : 各sellの中心y
            iy = y * sell + sell // 2
            # y1,y2は各セルの左端右端
            y1 = iy
            y2 = iy

            h = histogram[y, x] / np.sum(histogram[y, x])
            h /= h.max()

            for ind in range(Index):
                angle = (ind * 20 + 10) / 180 * np.pi
                rx = int(np.sin(angle) * (x1 - ix) + np.cos(angle) * (y1 - iy) + ix)
                ry = int(np.cos(angle) * (x1 - ix) - np.cos(angle) * (y1 - iy) + iy)
                lx = int(np.sin(angle) * (x2 - ix) + np.cos(angle) * (y2 - iy) + ix)
                ly = int(np.cos(angle) * (x2 - ix) - np.cos(angle) * (y2 - iy) + iy)

                # color is HOG value
                c = int(255. * h[ind])

                # draw line
                cv2.line(gray, (lx, ly), (rx, ry), c, thickness=1)
    return gray

img = cv2.imread("../imori.jpg").astype(np.float)
gray = BGRtoGRAY(img)

histogram = HOG(gray)

result = DrawHOG(histogram, gray)
result = result.astype(np.uint8)

cv2.imwrite("myans_69.jpg", result)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
