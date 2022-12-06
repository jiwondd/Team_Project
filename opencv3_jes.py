import cv2
import numpy as np
import os
import tensorflow as tf
print(tf.__version__) #2.9.1

path_dir = "D:\project/DATA\crawling\CHA/"
file_list = os.listdir(path_dir)
cap=cv2.VideoCapture(0)

# print(file_list[0])
# print(len(file_list))

file_name_list = []

for i in range(len(file_list)):
    file_name_list.append(file_list[i].replace(".jpg","")) # 최종 이미지 저장 시 이름이 '이름.jpg.jpg'가 되는 것을 방지
print(file_name_list)


# 여러장의 사진을 작업하기 위한 코드 함수화
print(file_name_list[0])
def Cutting_face_save(image, name):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 2)
        cropped = image[y: y+h, x: x+w]
        resize = cv2.resize(cropped, (250,250))
        cv2.imshow("crop&resize", resize)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 이미지 저장하기
        cv2.imwrite(f"D:\project/DATA\crawling\CHA/{name}.jpg", resize)

# 지정한 경로의 모든 사진들에서 이미지를 검출해 자르고 저장하는 작업 수행 코드        
for name in file_name_list:
    img = cv2.imread("D:\project/DATA\crawling\CHA/"+name+".jpg")
    Cutting_face_save(img, name)
    
'''
# cv2.error: OpenCV(4.6.0) D:\a\opencv-python\opencv-python\opencv\modules\imgproc\src\color.cpp:182: error: (-215:Assertion failed) !_src.empty() in function 'cv::cvtColor'
1. 파일, 경로에 한글빼고 영어만 넣기 -> 안됨
2. 해당 폴더에 cv파일 직접 넣기 -> 안됨
3. 해당 폴더에 xml파일을 직접 넣기 -> 안됨
4. 모든경로 통일시키키 -> 안됨

'''