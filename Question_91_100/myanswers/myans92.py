import cv2
import numpy as np

def QuantizeBGRColor(img, k=4):
    uint8 = np.iinfo(np.uint8)
    rank = int(uint8.max + 1) // k

    img = (img // rank + 0.5) * rank

    return img

def Kmeans(img, K=5):
    Ver, Hor, _ = img.shape

    img = img.reshape(Ver*Hor, 3)
    inds = np.zeros(Ver*Hor).reshape(Ver*Hor, 1)
    img = np.hstack((img, inds))

    cls = np.random.choice(np.arange(Ver*Hor), K, replace=False)
    for x in range(Ver*Hor):
        d_list = []
        for k in range(K):
            dis = np.sqrt(np.sum((img[cls[k]] - img[x]) ** 2))
            d_list.append(dis)

        ind = d_list.index(min(d_list))
        img[x, 3] = ind

    bfr_cls_vals = img[cls]

    while True:

        # update each class value
        cls_vals = []
        for k in range(K):
            ind = np.where(img[..., -1] == k)
            cls_vals.append(np.mean(img[ind], axis=0))
        # list -> ndarray transform
        cls_vals = np.array(cls_vals)

        # check class update
        if np.all(cls_vals == bfr_cls_vals):
            break

        # update befor class valus
        bfr_cls_vals = cls_vals

        # update each pixel label
        for x in range(Ver*Hor):
            d_list = []
            for k in range(K):
                dis = np.sqrt(np.sum((cls_vals[k, 0:3] - img[x, 0:3]) ** 2))
                d_list.append(dis)

            ind = d_list.index(min(d_list))
            img[x, 3] = ind

    img = img.reshape(Ver, Hor, 4)

    # set each class value
    for k in range(K):
        ind = np.where(img[..., -1] == k)
        img[ind] = cls_vals[k]

    return img[..., 0:3]

fname = "../imori.jpg"
ansfname = "myans_92_imori.jpg"
img = cv2.imread(fname).astype(np.float)

result = Kmeans(img, 10)
result = result.astype(np.uint8)

cv2.imwrite(ansfname, result)
cv2.imshow("result", result)
cv2.waitKey(0)

