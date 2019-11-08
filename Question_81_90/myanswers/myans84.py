import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

def QuantizeBGRColor(img, k=4):
    uint8 = np.iinfo(np.uint8)
    rank = int(uint8.max + 1) // k

    img = (img // rank + 0.5) * rank

    return img

def CreateHistgram():
    fnames = glob.glob('../dataset/train_*')
    fnames.sort()

    db = np.zeros((len(fnames), 14), dtype=np.int)
    
    for i, path in enumerate(fnames):
        img = QuantizeBGRColor(cv2.imread(path))

        for j in range(4):
            db[i, j] = len(np.where(img[..., 0] == (64 * j + 32))[0])
            db[i, j+4] = len(np.where(img[..., 1] == (64 * j + 32))[0])
            db[i, j+8] = len(np.where(img[..., 2] == (64 * j + 32))[0])

        cls = 0
        if path in 'akahara':
            cls = 1
        elif path in 'madara':
            cls = 2
        db[i, 13] = cls


        img_h = img.copy() // 64
        img_h[..., 1] += 4
        img_h[..., 2] += 8
        plt.subplot(2, 5, i+1)
        plt.hist(img_h.ravel(), bins=12, rwidth=0.8)
        plt.title(path)

    plt.hist(db, bins=12, rwidth=1.0)
    plt.show()


CreateHistgram()
