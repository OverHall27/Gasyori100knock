import cv2
import numpy as np

def CalcIOU(reg1, reg2):

    area1 = (reg1[2] - reg1[0]) * (reg1[3] - reg1[1])
    area2 = (reg2[2] - reg2[0]) * (reg2[3] - reg2[1])

    reg12 = np.zeros(4, dtype=np.float32)
    for i in range(2):
        reg12[i] = max(reg1[i], reg2[i])
        reg12[i+2] = min(reg1[i+2], reg2[i+2])
    
    area12 = (reg12[2] - reg12[0]) * (reg12[3] - reg12[1])

    return area12 / (area1 + area2 - area12)


a = np.array((50, 50, 150, 150), dtype=np.float32)
b = np.array((60, 60, 170, 160), dtype=np.float32)

print(CalcIOU(a, b))
