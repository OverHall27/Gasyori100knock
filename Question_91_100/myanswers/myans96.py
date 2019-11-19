import numpy as np
import cv2

np.random.seed(0)

def BGRtoGRAY(img):
    gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
    return gray

def HOG(gray):

    def GetGradXY(gray):
        Ver, Hor = gray.shape

        gray = np.pad(gray, (1, 1), 'edge')
        gx = gray[1:Ver+1, 2:] - gray[1:Ver+1, :Hor]
        gy = gray[:Ver, 1:Hor+1] - gray[2:, 1:Hor+1]
        #gy = gray[2:, 1:Hor+1] - gray[:Ver, 1:Hor+1]
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

class NN:

    def __init__(self, ind=2, w1=64, w2=64, outd=1, lr=0.1):
        # np.random.normal(average, sigma, paxis0, axis1, ...])
        self.w1 = np.random.normal(0, 1, [ind, w1])
        self.b1 = np.random.normal(0, 1, [w1])

        self.w2 = np.random.normal(0, 1, [w1, w2])
        self.b2 = np.random.normal(0, 1, [w2])

        self.wout = np.random.normal(0, 1, [w2, outd])
        self.bout = np.random.normal(0, 1, [outd])
        self.lr = lr

    def forward(self, x):
        # x is input
        self.z1 = x
        # 第１層
        self.z2 = sigmoid(np.dot(self.z1, self.w1) + self.b1)
        # １層増やした中間層.
        self.z3 = sigmoid(np.dot(self.z2, self.w2) + self.b2)
        # 出力はこれ
        self.out = sigmoid(np.dot(self.z3, self.wout) + self.bout)

        return self.out

    def train(self, x, t):
        # backpropagation output layer
        En = 2 * (t - self.out) * self.out * (1 - self.out)
        grad_En = En 
        grad_wout = np.dot(self.z3.T, En)
        grad_bout = np.dot(np.ones([En.shape[0]]), En)

        # 学習係数をかけて加算
        self.wout += self.lr * grad_wout
        self.bout += self.lr * grad_bout

        # backpropagation inter layer
        grad_u2 = np.dot(En, self.wout.T) * self.z3 * (1 - self.z3)
        grad_w2 = np.dot(self.z2.T, grad_u2)
        grad_b2 = np.dot(np.ones([grad_u2.shape[0]]), grad_u2)
        self.w2 += self.lr * grad_w2
        self.b2 += self.lr * grad_b2

        grad_u1 = np.dot(grad_u2, self.w2.T) * self.z2 * (1 - self.z2)
        grad_w1 = np.dot(self.z1.T, grad_u1)
        grad_b1 = np.dot(np.ones([grad_u1.shape[0]]), grad_u1)
        self.w1 += self.lr * grad_w1
        self.b1 += self.lr * grad_b1

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def MakeDataset(img, gt, th=0.5, dataN=200, cropL=60, size=32):

    def CalcIOU(reg1, reg2):

        area1 = (reg1[2] - reg1[0]) * (reg1[3] - reg1[1])
        area2 = (reg2[2] - reg2[0]) * (reg2[3] - reg2[1])
        reg12 = np.zeros(4, dtype=np.float32)

        for i in range(2):
            reg12[i] = max(reg1[i], reg2[i])
            reg12[i+2] = min(reg1[i+2], reg2[i+2])

            area12 = (reg12[2] - reg12[0]) * (reg12[3] - reg12[1])
        return area12 / (area1 + area2 - area12)

    Ver, Hor = img.shape
    set_size = ((size // 8) ** 2) * 9
    db = np.zeros((dataN, set_size + 1))

    for i in range(dataN):
        x1 = np.random.randint(Hor - cropL)
        y1 = np.random.randint(Ver - cropL)
        reg = np.array((x1, y1, x1+cropL, y1+cropL))
        crop = img[y1:y1+cropL, x1:x1+cropL].astype(np.float32)

        iou = CalcIOU(gt, reg)
        label = 0
        if iou <= th:
            label = 1
        else:
            label = 0

        crop = cv2.resize(crop, dsize=(size, size))
        HOG_crop = HOG(crop)
        db[i, :set_size] = HOG_crop.ravel()
        db[i, -1] = label

    return db

img = cv2.imread("../imori.jpg").astype(np.float)
gray = BGRtoGRAY(img)
gt = np.array((47, 41, 129, 103), dtype=np.float32)

dataset = MakeDataset(gray, gt, th=0.5)
dims = (dataset.shape[1] - 1)
nn = NN(ind=dims, lr=0.01)

train_x = dataset[:, :dims]
train_t = dataset[:, -1][..., None]

# train
for i in range(10000):
    nn.forward(train_x)
    nn.train(train_x, train_t)

# test
accuracy_N = 0.
pred_th = 0.5
# each data
for data, t in zip(train_x, train_t):
    # get prediction
    prob = nn.forward(data)

    # count accuracy
    pred = 1 
    if prob >= pred_th:
        pred = 1
    else:
        pred = 0

    if t == pred:
        accuracy_N += 1

    # get accuracy 
accuracy = accuracy_N / len(dataset)

print("Accuracy >> {} ({} / {})".format(accuracy, accuracy_N, len(dataset)))
