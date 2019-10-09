import cv2

img = cv2.imread("imori.jpg")
#BGRの順番で読み込んでいる
if img is None:
    print("failed to lead image")

ans_img = img[:,:,2] * 0.2126 + img[:,:,1] * 0.7152 + img[:,:,0] * 0.0722

cv2.imwrite("myans_02.jpg", ans_img)
