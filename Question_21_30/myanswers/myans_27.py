import cv2
import numpy as np

def WeightFunction(d, a=-1.):
    abs_d = np.abs(d)
    if abs_d <= 1:
        return (a + 2) * (abs_d ** 3) - (a + 3) * (abs_d ** 2) + 1
    elif abs_d <= 2:
        return a * (abs_d ** 3) - (5 * a) * (abs_d ** 2) + (8 * a * abs_d) - (4 * a)
    else:
        return 0

def BiCubicNeighbor(img, ax=1.0, ay=1.0):
    Hol, Ver, Col = img.shape
    
    aH = int(Hol * ax)
    aV = int(Ver * ay)
    
    result = np.zeros((aH, aV, Col))
    
    # x, y がオリジナル画像での画素位置を示すものになる

    '''
    #コード短くしたい
    x_list = np.floor(np.arange(aH) / ax).astype(np.int)
    y_list = np.floor(np.arange(aV) / ay).astype(np.int)
    
    h_list = np.zeros((aH, 4))
    h_list[:] = np.arange(-1, 3)
    adx_list = np.zeros((aH, 4))
    ady_list = np.zeros((aV, 4))
    
    for i in range(4):
        adx_list[..., i] = np.abs(h_list[..., i] + x_list - (np.arange(aH) / ax)).astype(np.float)
        ady_list[..., i] = np.abs(h_list[..., i] + y_list - (np.arange(aV) / ay)).astype(np.float)

    calc_1 = np.zeros(aH * aV)
    calc_2 = np.zeros(aH * aV)

    for j in range(4):
        for i in range(4):
            print(map(WeightFunction, adx_list[:, i]) * map(WeightFunction, ady_list[:, j]))
            #calc_1 += WeightFunction(adx_list[:, i], ax) * WeightFunction(ady_list[:, j], ax)
            #img_h = x_list + i - 1
            #img_v = y_list + j - 1
            #calc_2 += WeightFunction(dx_list[:, i], ax) * WeightFunction(dy_list[:, j], ax) * img[img_h, img_v]
    #result = calc_2 / calc_1

    '''
    for h in range(aH):
       dx_list = np.arange(-1, 3)
       x = np.floor(h / ax).astype(np.int)
       dx_list = np.abs(dx_list + x - (h / ax)).astype(np.float)

       #dx1 = np.abs((x - 1) - (h / ax))
       #dx2 = np.abs(   x    - (h / ax))
       #dx3 = np.abs((x + 1) - (h / ax))
       #dx4 = np.abs((x + 2) - (h / ax))

       for v in range(aV):
            dy_list = np.arange(-1, 3)
            y = np.floor(v / ay).astype(np.int)
            dy_list = np.abs(dy_list + y - (v / ay)).astype(np.float)
            
            #dy1 = np.abs((y - 1) - (v / ay))
            #dy2 = np.abs(   y    - (v / ay))
            #dy3 = np.abs((y + 1) - (v / ay))
            #dy4 = np.abs((y + 2) - (v / ay))
            
            calc_1 = 0
            calc_2 = 0
            for j in range(4):
                for i in range(4):
                    calc_1 += WeightFunction(dx_list[i], a=-1.) * WeightFunction(dy_list[j], a=-1.)
                    img_h = x + i - 1
                    img_v = y + j - 1
                    if img_h > (Hol - 2):
                        img_h = Hol - 2
                    if img_v > (Ver - 2):
                        img_v = Ver - 2

                    calc_2 += WeightFunction(dx_list[i], a=-1.) * WeightFunction(dy_list[j], a=-1.) * img[img_h, img_v]
            result[h, v] = calc_2 / calc_1
    return result

img = cv2.imread("imori.jpg")

result = BiCubicNeighbor(img, 1.5, 1.5)

cv2.imwrite("myans_27.jpg", result)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
