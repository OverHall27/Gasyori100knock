import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
# q88-89はまとめて88に解答

def QuantizeBGRColor(img, k=4):
    uint8 = np.iinfo(np.uint8)
    rank = int(uint8.max + 1) // k

    img = (img // rank + 0.5) * rank

    return img

def CreateHistgram(link):
    fnames = glob.glob(link)
    fnames.sort()

    db = np.zeros((len(fnames), 14), dtype=np.int)
   
    for i, path in enumerate(fnames):
        img = QuantizeBGRColor(cv2.imread(path))

        for j in range(4):
            db[i, j] = len(np.where(img[..., 0] == (64 * j + 32))[0])
            db[i, j+4] = len(np.where(img[..., 1] == (64 * j + 32))[0])
            db[i, j+8] = len(np.where(img[..., 2] == (64 * j + 32))[0])

    falsed_rand_cls = True
    while(falsed_rand_cls):
        for i in range(len(fnames)):
            cls = 0
            if np.random.random() >= 0.5:
                cls = 1
            else:
                cls = 0
            db[i, 13] = cls

        falsed_rand_cls = (len(db[db[..., 13] == 0]) == 0) or (len(db[db[..., 13] == 0]) == 0)
    
    return db

def Kmeans(link):

    np.random.seed(1)
    # 各画像の特徴量
    histogram = CreateHistgram(link)

    # 各クラス(0, 1)の重心
    gs = np.zeros((2, 12), dtype=np.float32)

    is_loop = True
    while(is_loop):
        is_changed = False
        cls0_hist = histogram[histogram[..., 13] == 0]
        cls1_hist = histogram[histogram[..., 13] == 1]

        for i in range(12):
            gs[0, i] = np.mean(cls0_hist[..., i])
            gs[1, i] = np.mean(cls1_hist[..., i])

        for i in range(4):
            d0 = 0
            d1 = 0

            for j in range(12):
                d0 += (histogram[i, j] - gs[0, j]) ** 2
                d1 += (histogram[i, j] - gs[1, j]) ** 2

            d0 = np.sqrt(d0)
            d1 = np.sqrt(d1)
            if d0 < d1:
                if histogram[i, 13] != 0:
                    histogram[i, 13] = 0
                    is_changed = True

            else:
                if histogram[i, 13] != 1:
                    histogram[i, 13] = 1
                    is_changed = True

        cls0 = len(histogram[histogram[..., 13] == 0]) != 0
        cls1 = len(histogram[histogram[..., 13] == 1]) != 0

        is_loop = is_changed & (cls0 & cls1)
        print(is_loop)

    return histogram

link = '../dataset/test_*'
fnames = glob.glob(link)
fnames.sort()

histogram = Kmeans(link)

for i, path in enumerate(fnames):
    print(path)
    print(histogram[i, 13])
