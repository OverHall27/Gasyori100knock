import cv2

answer = cv2.imread("answers/answer_12.jpg")
answer2 = cv2.imread("myans/myans_12.jpg")
print(answer- answer2)
