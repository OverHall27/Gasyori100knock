import cv2

img = cv2.imread("imori.jpg")
#BGRの順番で読み込んでいる
if img is None:
    print("failed to lead image")
blue = img[:,:,0].copy()
red = img[:,:,2].copy()
img[:,:,0] = red
img[:,:,2] = blue

ans_img = cv2.imwrite("myans_01.jpg", img)
