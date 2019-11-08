import cv2
import numpy as np
import numpy.linalg as LA

def BGRtoGRAY(img):
    gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]

    return gray

def DetectHessianCorner(gray):

    def SobelFilter(gray):
        Ver, Hor = gray.shape

        sobel_x = np.zeros_like(gray)
        sobel_y = np.zeros_like(gray)

        ## Sobel vertical
        Kv = [[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]]
        ## Sobel horizontal
        Kh = [[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]]

        pad = 1
        gray = np.pad(gray, (pad, pad), 'edge')
        for x in range(pad, Hor+pad):
            for y in range(pad, Ver+pad):
                sobel_y[y-pad, x-pad] = np.mean(gray[y-pad:y+pad+1, x-pad:x+pad+1] * Kv)
                sobel_x[y-pad, x-pad] = np.mean(gray[y-pad:y+pad+1, x-pad:x+pad+1] * Kh)


        return sobel_x, sobel_y

    def Hessian(gray, Ix, Iy):
        Ver, Hor = gray.shape
        Ix = Ix.astype(np.float32)
        Iy = Iy.astype(np.float32)

        Ixx = Ix * Ix
        Iyy = Iy * Iy
        Ixy = Ix * Iy
        Hessian = np.zeros((Ver, Hor), dtype=np.float32)
        result = np.zeros((Ver, Hor))

        for x in range(Hor):
            for y in range(Ver):
                Hessian[y, x] = Ixx[y, x] * Iyy[y, x] - Ixy[y, x] ** 2

        for x in range(Hor):
            for y in range(Ver):
                if Hessian[y, x] == np.max(Hessian[max(y-1,0) : min(y+2, Ver), max(x-1, 0) : min(x+2, Hor)]) and Hessian[y, x] > np.max(Hessian) * 0.1:
                    result[y, x] = 255

        return result

    sobel_x, sobel_y = SobelFilter(gray)
    Hes = Hessian(gray, sobel_x, sobel_y)

    return Hes

gray = cv2.imread("../thorino.jpg", cv2.IMREAD_GRAYSCALE)
result = DetectHessianCorner(gray)

cv2.imshow("result", result)
cv2.waitKey(0)
cv2.imwrite("myans_81.jpg", result)
