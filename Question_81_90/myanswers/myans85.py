import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

# Q85-86は回答したが，87も上書きで回答してしまった．
# Q86は計算しなくても出力で正解数を目視判断した
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
            cls = 0
        elif path in 'madara':
            cls = 1
        db[i, 13] = cls


        img_h = img.copy() // 64
        img_h[..., 1] += 4
        img_h[..., 2] += 8
        plt.subplot(2, 5, i+1)
        plt.hist(img_h.ravel(), bins=12, rwidth=0.8)
        plt.title(path)

    plt.hist(db, bins=12, rwidth=1.0)
    plt.show()

    return db

def KNNRecognition(db, K):
    db = CreateHistgram()

    fnames = glob.glob('../dataset/test_*')
    fnames.sort()

    train_names = glob.glob('../dataset/train_*')
    train_names.sort()

    for i, path in enumerate(fnames):

        img = QuantizeBGRColor(cv2.imread(path))

        # dbのclass別
        diff_list = []
        cls_1 = 0
        cls_2 = 0

        for k in range(10):
            # 4段階での計算
            tmp = 0
            for j in range(4):
                # BGRでの計算
                tmp += np.abs(db[k, j] - len(np.where(img[..., 0] == ( 64 * j + 32))[0]))
                tmp += np.abs(db[k, j+4] - len(np.where(img[..., 1] == ( 64 * j + 32))[0]))
                tmp += np.abs(db[k, j+8] - len(np.where(img[..., 2] == ( 64 * j + 32))[0]))

            diff_list.append(tmp)

        for l in range(K):
            ind = diff_list.index(min(diff_list))
            diff_list[ind] = np.iinfo(np.int32).max

            if ind < 5:
                cls_1 += 1
            else:
                cls_2 += 1

        print(path)
        if cls_1 > cls_2:
            print("akahara")
        else:
            print("madara")
        print("")

db = CreateHistgram()
KNNRecognition(db, 3)
