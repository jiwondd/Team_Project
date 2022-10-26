from retinaface import RetinaFace
from deepface import DeepFace
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import os


img_path="D:/project/DL/retina_data"
img_list=os.listdir(img_path)
file_name_list = []

for i in range(len(img_list)):
    file_name_list.append(img_list[i].replace(".jpg","")) # 최종 이미지 저장 시 이름이 '이름.jpg.jpg'가 되는 것을 방지
print(file_name_list)

faces=RetinaFace.extract_faces(img_path,align=True)
img=cv2.imread(img_path)
# print(len(obj.keys())) #6

print(file_name_list[0])
def Cutting_face_save(image, name):
    # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_cascade=RetinaFace.extract_faces(img_path,align=True)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faces.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 2)
        cropped = image[y: y+h, x: x+w]
        resize = cv2.resize(cropped, (150,150))
        cv2.imshow("crop&resize", resize)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 이미지 저장하기
        cv2.imwrite(f"D:/project/DL/retina_data/{name}.jpg", resize)
        
for name in file_name_list:
    img = cv2.imread("D:/project/DL/retina_data/"+name+".jpg")
    Cutting_face_save(img, name)


# for key in obj.keys():
#     identity=obj[key]
#     # print(identity)
#     facial_area=identity["facial_area"]
    
#     cv2.rectangle(img,(facial_area[2],facial_area[3]),(facial_area[0],facial_area[1]),(255,255,255),1)
    
# plt.imshow(img[:,:,::-1])
# plt.show()

# for face in faces :
#     plt.imshow(face)
#     plt.show()

'''
  File "C:\Users\AIA\AppData\Roaming\Python\Python39\site-packages\retinaface\RetinaFace.py", line 43, in get_image
    raise ValueError("Input image file path (", img_path, ") does not exist.")
ValueError: ('Input image file path (', 'D:/project/DL/retina_data', ') does not exist.')

와이라노 진쯔 개빡쳐

'''