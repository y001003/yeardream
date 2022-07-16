import cv2
import glob
import sys

img_files = glob.glob('컴퓨터비전-강의자료-통합본-업데이트-20220129/ch01/images/*.jpg')
# for i in img_files:
#     print(i)
if not img_files:
    print("There are no jpg files in 'images' folder")
    sys.exit()
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cnt = len(img_files)
idx = 0

while True:
    img = cv2.imread(img_files[idx])

    if img is None:
        print('Image load failed!')
        break    

    cv2.imshow('image', img)    
    if cv2.waitKey(1000) >= 0: # 뭐냐
        break

    idx += 1
    if idx >= cnt:
        idx = 0

cv2.destroyAllWindows()



