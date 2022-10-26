import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from retinaface import RetinaFace
from deepface import DeepFace


path_dir="D:/project/DL/retina_data/img01.jpg"

faces=RetinaFace.extract_faces(img_path="D:/project/DL/retina_data/img01.jpg",align=True)

for face in faces :
    plt.imshow(face)
    plt.show()





'''
file_list=os.listdir(path_dir)
print(file_list[0]) #img01.jpg

print(len(file_list)) #4

file_name_list=[]
for i in range(len(file_list)):
    file_name_list.append(file_list[i].replace(".jpg",""))
    
print(file_name_list) #['img01', 'img02', 'ljj01', 'ljj02']

image_cut=cv2.imread("d:/project/DL/retina_data/img01/jpg")
retinaface=RetinaFace.extract_faces(image_cut)
gray=cv2.cvtColor(image_cut,cv2.COLOR_BGR2GRAY)
faces=retinaface.detectMultiScale(gray,1.3,5)

'''