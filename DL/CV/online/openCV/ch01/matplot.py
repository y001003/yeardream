import matplotlib.pyplot as plt
import cv2


# 컬러 영상 출력
imgBGR = cv2.imread('컴퓨터비전-강의자료-통합본-업데이트-20220129/ch01/cat.bmp')
imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB) # OpenCV는 BGR로 불러옴으로 BGR -> RGB

plt.axis('off') # 가로세로 눈금 없애기
plt.imshow(imgRGB) 
plt.show()

# 그레이스케일 영상 출력
imgGray = cv2.imread('컴퓨터비전-강의자료-통합본-업데이트-20220129/ch01/cat.bmp', cv2.IMREAD_GRAYSCALE)

plt.axis('off')
plt.imshow(imgGray, cmap='gray')
plt.show()

# 두 개의 영상을 함께 출력
plt.subplot(121), plt.axis('off'), plt.imshow(imgRGB)
plt.subplot(122), plt.axis('off'), plt.imshow(imgGray, cmap='gray')
plt.show()
