import cv2
import numpy as np

# Q82-83はこれに解答
def BGRtoGRAY(img):
    gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]

    return gray

def DetectHerrisCorner(gray, k=0.04, th=0.1):

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

    def GetDetHessian(Ix, Iy):
        Ix = Ix.astype(np.float32)
        Iy = Iy.astype(np.float32)

        Ixx = Ix * Ix
        Iyy = Iy * Iy
        Ixy = Ix * Iy
        Hessian = np.zeros((Ver, Hor), dtype=np.float32)

        for x in range(Hor):
            for y in eange(Ver):
                Hessian[y, x] = Ixx[y, x] * Iyy[y, x] - Ixy[y, x] ** 2
        
        return Hessian_det

    def GaussianFilter(img, K_size=3, sigma=1.3):
        Ver, Hor = img. shape

        pad = K_size // 2
        tmp = np.pad(img, (pad,pad), 'constant')
        result = np.zeros_like(tmp)

        ## create kernel
        kernel = np.zeros((K_size, K_size), dtype=np.float)
        for x in range(-pad, pad+1):
            for y in range(-pad, pad+1):
                kernel[y + pad, x + pad] = np.exp( -(x ** 2 + y ** 2) / (2 * (sigma ** 2)))

        kernel /= ( sigma * np.sqrt(2. * np.pi))
        kernel /= kernel.sum()

        for x in range(pad, Hor+pad):
            for y in range(pad, Ver+pad):
                result[y, x] = np.sum(kernel * tmp[y-pad: y+pad+1, x-pad: x+pad+1])
        result =  result[pad: pad + Ver, pad: pad + Hor].astype(np.uint8)

        return result

    def DetectCorner(gray, Ixx, Iyy, Ixy, k=0.04, th=0.1):
        result = np.array((gray, gray, gray))
        result = np.transpose(result, (1, 2, 0))

        R = (Ixx * Iyy - Ixy ** 2) - k * ((Ixx + Iyy) ** 2)

        result[R >= np.max(R) * th] = [0, 0, 255]

        return result


    Ix, Iy = SobelFilter(gray)
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    Ixx = GaussianFilter(Ixx, 3, 1.3)
    Iyy = GaussianFilter(Iyy, 3, 1.3)
    Ixy = GaussianFilter(Ixy, 3, 1.3)

    result = DetectCorner(gray, Ixx, Iyy, Ixy, k=k, th=th)

    return result.astype(np.uint8)

gray = cv2.imread("../thorino.jpg", cv2.IMREAD_GRAYSCALE)
result = DetectHerrisCorner(gray, k=0.16, th=0.1)

cv2.imshow("result", result)
cv2.waitKey(0)
cv2.imwrite("myans_83.jpg", result)
