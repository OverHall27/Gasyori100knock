import cv2
import numpy as np
import sys

def BGRTOGRAY(img):
    gray_img = img[..., 2] * 0.2126 + img[..., 1] * 0.7152 + img[..., 0] * 0.0722

    return gray_img

def Canny_step1(img):

    def GaussianFilter(img, K_size=3, sigma=1.3):
        if len(img.shape) == 3:
            Ver, Hor, Col = img.shape
        else:
            img = np.expand_dims(img, axis=-1)
            Ver, Hor, Col = img.shape

        pad = K_size // 2
        print(img)
        #tmp = np.pad(img, [(pad, pad), (pad, pad), (0, 0)], mode='constant')
        tmp = np.pad(img, [(pad, pad), (pad, pad), (0, 0)], mode='edge')
        print(tmp)
        kernel = np.zeros((K_size, K_size), dtype=np.float)

        for x in range(-pad, pad+1):
            for y in range(-pad, pad+1):
                kernel[y + pad, x + pad] = np.exp( -(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
        kernel /= ( sigma * np.sqrt(2. * np.pi))
        kernel /= kernel.sum()

        result = np.zeros_like(tmp)
        for c in range(Col):
            for x in range(pad, pad+Hor):
                for y in range(pad, pad+Ver):
                    result[y, x, c] = np.sum(kernel * tmp[y-pad:y+pad+1, x-pad:x+pad+1, c])
        result =  result[pad: pad + Hor, pad: pad + Ver]

        result = np.clip(result, 0, 255).astype(np.uint8)

        return result[..., 0]

    gray = BGRTOGRAY(img)

    gaussian = GaussianFilter(gray, K_size=5, sigma=1.4)

    return gaussian


def Canny_step2(img):

    def SobelFilter(img, K_size=3):
        Hor, Ver = img.shape
        pad = K_size // 2
        ## zero padding

        #tmp = np.pad(img, [(pad, pad), (pad, pad)], 'constant').astype(np.float32)
        tmp = np.pad(img, [(pad, pad), (pad, pad)], 'edge').astype(np.float32)
        tmp_v = np.zeros_like(tmp).astype(np.float32)
        tmp_h = np.zeros_like(tmp).astype(np.float32)
      
        ##filter
        kernel_v = np.zeros((K_size, K_size), dtype=np.float32)
        kernel_h = np.zeros((K_size, K_size), dtype=np.float32)
        kernel_v = [[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]]
        kernel_h = [[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]]

        for x in range(pad, pad+Hor):
            for y in range(pad, pad+Ver):
                tmp_v[y, x] = np.sum(tmp[y-pad: y+pad+1, x-pad:x+pad+1] * kernel_v)
                tmp_h[y, x] = np.sum(tmp[y-pad: y+pad+1, x-pad:x+pad+1] * kernel_h)

        tmp_v = np.clip(tmp_v, 0, 255)
        tmp_h = np.clip(tmp_h, 0, 255)

        result_v = tmp_v[pad: pad + Ver, pad: pad + Hor].astype(np.uint8)
        result_h = tmp_h[pad: pad + Ver, pad: pad + Hor].astype(np.uint8)

        return result_v, result_h

    def CalcAngele(tan):

        result = np.zeros_like(tan, dtype=np.uint8)
        ind = np.where((-0.4142 < tan) & (tan <= 0.4142))
        result[ind] = 0
        ind = np.where((0.4142 < tan) & (tan <= 2.4142))
        result[ind] = 45
        ind = np.where(np.abs(tan) >= 2.4142)
        result[ind] = 90
        ind = np.where((-2.4142 < tan) & (tan <= -0.4142))
        result[ind] = 135
        
        return result

    def GetEdgeTan(sobel_h, sobel_v):
        edge = np.sqrt(np.power(sobel_v.astype(np.float32), 2) + np.power(sobel_h.astype(np.float32), 2))

        edge = np.clip(edge, 0, 255)
        # float_info_minは小さすぎる？
        #sobel_h = np.maximum(sobel_h, sys.float_info.min)
        sobel_h = np.maximum(sobel_h, 1e-5)
        tan = np.arctan(sobel_v / sobel_h)

        return edge, tan

    def CalcEdge2(angle, edge):
        H, W = angle.shape

        for y in range(H):
            for x in range(W):
                if angle[y, x] == 0:
                    dx1, dy1, dx2, dy2 = -1, 0, 1, 0
                elif angle[y, x] == 45:
                    dx1, dy1, dx2, dy2 = -1, 1, 1, -1
                elif angle[y, x] == 90:
                    dx1, dy1, dx2, dy2 = 0, -1, 0, 1
                elif angle[y, x] == 135:
                    dx1, dy1, dx2, dy2 = -1, -1, 1, 1
                    if x == 0:
                        dx1 = max(dx1, 0)
                        dx2 = max(dx2, 0)
                        if x == W-1:
                            dx1 = min(dx1, 0)
                            dx2 = min(dx2, 0)
                            if y == 0:
                                dy1 = max(dy1, 0)
                                dy2 = max(dy2, 0)
                                if y == H-1:
                                    dy1 = min(dy1, 0)
                                    dy2 = min(dy2, 0)
                                    if max(max(edge[y, x], edge[y+dy1, x+dx1]), edge[y+dy2, x+dx2]) != edge[y, x]:
                                        edge[y, x] = 0
        return edge


    def CalcEdge1(angle, edge):
        Ver, Hor = angle.shape

        #_edge = np.pad(edge, [(1, 1), (1, 1)], 'constant')
        _edge = np.pad(edge, [(1, 1), (1, 1)], 'edge')

        for x in range(Hor):
            for y in range(Ver):
                if angle[y, x] == 0:
                    dx1, dx2, dy1, dy2 = x-1, x+1, y, y
                elif angle[y, x] == 45:
                    dx1, dx2, dy1, dy2 = x-1, x+1, y+1, y-1
                elif angle[y, x] == 90:
                    dx1, dx2, dy1, dy2 = x, x, y-1,y+1
                elif angle[y, x] == 135:
                    dx1, dx2, dy1, dy2 = x-1, x+1, y-1, y+1
                max_arg = np.argmax([_edge[dy1+1, dx1+1], _edge[y+1, x+1], _edge[dy2+1, dx2+1]])
                if max_arg != 1:
                    edge[y, x] = 0
        return edge

    def CalcEdge0(angle, edge):
        Ver, Hor = edge.shape
        #angle = np.pad(angle, [(1, 1), (1, 1)], 'constant')
        angle = np.pad(angle, [(1, 1), (1, 1)], 'edge')

        ind = np.where(angle[1:Ver+1, 1:Hor+1] == 0)
        for i in range(len(ind[0])):
            y = ind[0][i] - 1
            x = ind[1][i] - 1
            max_arg = np.argmax([edge[y, x-1], edge[y, x], edge[y, x+1]])
        if max_arg != 1:
            edge[y, x] = 0

        ind = np.where(angle[1:Ver+1, 1:Hor+1] == 45)
        for i in range(len(ind[0])):
            y = ind[0][i] - 1
            x = ind[1][i] - 1
            max_arg = np.argmax([edge[y+1, x-1], edge[y, x], edge[y-1, x+1]])
            if max_arg != 1:
                edge[y, x] = 0

        ind = np.where(angle[1:Ver+1, 1:Hor+1] == 90)
        for i in range(len(ind[0])):
            y = ind[0][i] - 1
            x = ind[1][i] - 1
            max_arg = np.argmax([edge[y-1, x], edge[y, x], edge[y+1, x]])
            if max_arg != 1:
                edge[y, x] = 0

        ind = np.where(angle[1:Ver+1, 1:Hor+1] == 135)
        for i in range(len(ind[0])):
            y = ind[0][i] - 1
            x = ind[1][i] - 1
            max_arg = np.argmax([edge[y-1, x-1], edge[y, x], edge[y+1, x+1]])
            if max_arg != 1:
                edge[y, x] = 0
        
        return edge

    sobel_v, sobel_h = SobelFilter(img, K_size=3)

    edge, tan = GetEdgeTan(sobel_h, sobel_v)

    angle = CalcAngele(tan)

    edge0 = CalcEdge0(angle, edge)
    edge1 = CalcEdge1(angle, edge)
    edge2 = CalcEdge2(angle, edge)

    return edge0, edge1, edge2

def Canny_step3(edge, HT=100, LT=30):
    Ver, Hor = edge.shape

    edge[edge >= HT] = 255
    edge[edge <= LT] = 0

    #_edge = np.pad(edge, [(1, 1), (1, 1)], mode='constant').astype(np.float32)
    _edge = np.pad(edge, [(1, 1), (1, 1)], mode='edge').astype(np.float32)
    nn = [[1., 1., 1.], [1., 0., 1.], [1., 1., 1.]]

    for y in range(1, Ver+1):
        for x in range(1, Hor+1):
            if _edge[y, x] < LT or _edge[y, x] > HT:
                continue
            elif (np.max(_edge[y-1:y+2, x-1:x+2] * nn) > HT):
                _edge[y-1, x-1] = 255
            else:
                _edge[y-1, x-1] = 0

    #return edge
    return _edge[1:Ver+1, 1:Hor+1]


img = cv2.imread("../imori.jpg").astype(np.float)

#gaussian = Canny_step1(img)
'''
edge0, edge1, edge2 = Canny_step2(gaussian)
myans0 = Canny_step3(edge0, HT=100., LT=30.)
myans1 = Canny_step3(edge1, HT=100., LT=30.)
myans2 = Canny_step3(edge2, HT=100., LT=30.)
cv2.imwrite("myans_43_0.jpg", myans0)
cv2.imwrite("myans_43_1.jpg", myans1)
cv2.imwrite("myans_43_2.jpg", myans2)
'''
# OpenCVでのCanny法
gray = BGRTOGRAY(img).astype(np.uint8)
official = cv2.Canny(gray, threshold1=30., threshold2=100., apertureSize=3, L2gradient=False)
cv2.imwrite("Canny_sample.jpg", official)
cv2.imshow("official", official)
'''
#edge 0-2はどれも近い
cv2.imshow("0", myans0)
cv2.imshow("1", myans1)
cv2.imshow("2", myans2)
'''

cv2.waitKey(0)
cv2.destroyAllWindows()

