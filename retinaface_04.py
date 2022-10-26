import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from retinaface import RetinaFace
from deepface import DeepFace
from PIL import Image

path_dir="d:/project/DL/retina_data/"
file_list=os.listdir(path_dir)
print(file_list[0]) #img01.jpg

print(len(file_list)) #4

file_name_list=[]
for i in range(len(file_list)):
    file_name_list.append(file_list[i].replace(".jpg",""))
    
print(file_name_list) #['img01', 'img02', 'ljj01', 'ljj02']

image_cut=cv2.imread("d:/project/DL/retina_data/01.png")
retinaface=RetinaFace.extract_faces(img_path="d:/project/DL/retina_data/01.png",align=True,threshold=0.5)
# def extract_faces(img_path, threshold=0.9, model = None, align = True, allow_upscaling = True):
# print(np.array(retinaface).shape)
print(retinaface[0].shape)  #(65, 47, 3)

for i in range(len(retinaface)):
    a=np.array(retinaface[i])
    img011=Image.fromarray(a)
    img011.save(f'100{i}.jpg','JPEG')
    

# cv2.imwrite("img011.jpg",retinaface)
# gray=cv2.cvtColor(image_cut,cv2.COLOR_BGR2GRAY)
# faces=retinaface.detectMultiScale(gray,1.3,5)

# saveimg=cv2.imwrite("d:/project/DL/retina_data/img011.jpg",retinaface)