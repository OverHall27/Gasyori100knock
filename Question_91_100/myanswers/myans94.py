import cv2
import numpy as np
import matplotlib.pyplot as plt

def RandomCropping(img, gt):

    def CalcIOU(reg1, reg2):

        area1 = (reg1[2] - reg1[0]) * (reg1[3] - reg1[1])
        area2 = (reg2[2] - reg2[0]) * (reg2[3] - reg2[1])
        reg12 = np.zeros(4, dtype=np.float32)

        for i in range(2):
            reg12[i] = max(reg1[i], reg2[i])
            reg12[i+2] = min(reg1[i+2], reg2[i+2])

        area12 = (reg12[2] - reg12[0]) * (reg12[3] - reg12[1])
        return area12 / (area1 + area2 - area12)

    Ver, Hor, _ = img.shape
    np.random.seed(0)

    for i in range(200):
        x1 = np.random.randint(Hor - 60)
        y1 = np.random.randint(Ver - 60)
        reg = np.array((x1, y1, x1+60, y1+60), dtype=np.float32)

        iou = CalcIOU(reg, gt)

        if iou >= 0.5:
            cv2.rectangle(img, (reg[0], reg[1]), (reg[2], reg[3]), (0, 0, 255))
        else:
            cv2.rectangle(img, (reg[0], reg[1]), (reg[2], reg[3]), (255, 0, 0))
        cv2.rectangle(img, (gt[0], gt[1]), (gt[2], gt[3]), (0, 255, 0))

    return img

img = cv2.imread("../imori_1.jpg")
gt = np.array((47, 41, 129, 103), dtype=np.float32)

result = RandomCropping(img, gt)

cv2.imshow("result", result)
cv2.waitKey(0)

cv2.imwrite("myans_94.jpg", result)
