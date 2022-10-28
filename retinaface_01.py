from retinaface import RetinaFace
from deepface import DeepFace
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt


img_path="D:/project/DL/retina_data/img01.png"
# obj=RetinaFace.detect_faces(img_path)
obj=RetinaFace.extract_faces(img_path,align=True)
img=cv2.imread(img_path)
# print(len(obj.keys())) #6

# face detection = 얼굴찾기 
for key in obj.keys():
    identity=obj[key]
    # print(identity)
    facial_area=identity["facial_area"]
    
    cv2.rectangle(img,(facial_area[2],facial_area[3]),(facial_area[0],facial_area[1]),(255,255,255),1)
    
plt.imshow(img[:,:,::-1])
plt.show()

# face recognition = 이미지 두개로 서로 맞는 얼굴인지 알아보기
# obj2=DeepFace.verify(img1_path="D:/project/DL/retina_data/ljj01.jpg",img2_path="D:/project/DL/retina_data/ljj02.jpg",
#                      model_name='ArcFace',detector_backend='retinaface')
# print(obj2)