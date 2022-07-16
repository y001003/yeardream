import cv2
import sys 

print("Hello, openCV", cv2.__version__)
# img = cv2.imread('컴퓨터비전-강의자료-통합본-업데이트-20220129/ch01/cat.bmp')
img = cv2.imread('컴퓨터비전-강의자료-통합본-업데이트-20220129/ch01/cat.bmp', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("imge load failed")
    sys.exit()
cv2.imwrite('컴퓨터비전-강의자료-통합본-업데이트-20220129/ch01/cat_gray.png',img)

cv2.namedWindow('image') # opencv에서 지원하는 창  생성
cv2.imshow('image',img)# image라는 창에 img 를 보여줘라

while True:
    if cv2.waitKey() == 27:
        break
# cv2.waitKey()
cv2.destroyAllWindows()