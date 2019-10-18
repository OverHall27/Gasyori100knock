import cv2
import numpy as np

def BGRTOGRAY(img):
    gray = img[:,:,2] * 0.2126 + img[:,:,1] * 0.7152 + img[:,:,0] * 0.0722

    return gray.astype(np.uint8)

def OtsuThreshould(gray):

    Ver, Hor = gray.shape
    x = np.tile(np.arange(Hor), (Ver, 1))
    y = np.arange(Ver).repeat(Hor).reshape(Ver, Hor)
    current_var_between = 0

    boundary = 0
    for i in range(1, 255):
        weight_low = gray[gray < i].size / gray.size
        weight_high = gray[gray >= i].size / gray.size

        if (weight_low > 0) and (weight_high > 0):

            medium_low = np.mean(gray[gray < i])
            #var_low = np.var(gray[gray <= i])
            medium_high = np.mean(gray[gray >= i])
            #var_high = np.var(gray[gray > i])

            #var_inner = weight_low * var_low + weight_high * var_high
            var_between = weight_low * weight_high * ((medium_low - medium_high) ** 2)

            if current_var_between < var_between:
                current_var_between = var_between
                boundary = i

    print(boundary)
    gray[gray > boundary] = 255
    gray[gray <= boundary] = 0

    return gray
 
def Mofology(binary, times=1, mode='expansion'):

    def expansion(binary):
        Ver, Hor = binary.shape

        fileter = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
        tmp = np.pad(binary, [(1, 1), (1, 1)], mode='constant')
        binary = np.pad(binary, [(1, 1), (1, 1)], mode='constant')

        for x in range(1, Hor+1):
            for y in range(1, Ver+1):
                if np.sum(fileter * tmp[y-1:y+2, x-1:x+2]) >= 255:
                    binary[y, x] = 255

        return binary[1:Ver+1, 1:Hor+1]

    def shrink(binary):
        Ver, Hor = binary.shape

        fileter = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
        tmp = np.pad(binary, [(1, 1), (1, 1)], mode='constant')
        binary = np.pad(binary, [(1, 1), (1, 1)], mode='constant')

        for x in range(1, Hor+1):
            for y in range(1, Ver+1):
                if np.sum(fileter * tmp[y-1:y+2, x-1:x+2]) < 255*4:
                    binary[y, x] = 0

        return binary[1:Ver+1, 1:Hor+1]

    if mode == 'expansion':
        for i in range(times):
            binary = expansion(binary)
    elif mode == 'shrink':
         for i in range(times):
            binary = shrink(binary)
    
    return binary


img = cv2.imread("../imori.jpg")
gray = BGRTOGRAY(img)
binary = OtsuThreshould(gray)
result = Mofology(binary, 1, 'shrink')
result = Mofology(result, 1, 'expansion')

cv2.imshow("result", result)
cv2.imwrite("myans_49.jpg", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
