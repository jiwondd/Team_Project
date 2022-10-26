import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from retinaface import RetinaFace
from deepface import DeepFace
from PIL import Image

path_dir="d:/project/DL/retina_data/"
file_list=os.listdir(path_dir)
print(file_list[0]) #01.png

print(len(file_list)) #6

file_name_list=[]
for i in range(len(file_list)):
    file_name_list.append(file_list[i])
    
print(file_name_list) #['01', '02', '03', '04', '05', '06']


# retinaface=RetinaFace.extract_faces(img_path=f"d:/project/DL/retina_data/"{'0'+i}+".png",align=True)
# retinaface=RetinaFace.extract_faces(img_path=f"d:/project/DL/retina_data/0{+i}.png",align=True)


# for i in range(len(file_name_list)):
#     retinaface=RetinaFace.extract_faces(img_path=f"d:/project/DL/retina_data/0{i}.png",align=True)
#     print(retinaface)
#     for i in range(len(retinaface)):
#         a=np.array(retinaface[i])
#         img01=Image.fromarray(a.squeeze())
#         img01.save(f'00{i}.jpg','JPEG')
        
for i in range(len(file_name_list)):
    retinaface=RetinaFace.extract_faces(img_path=f"d:/project/DL/retina_data/0{i}.jpg",align=True)
    print(retinaface)

    # a=np.array(retinaface)
    
for j in range(len(retinaface)):
    a=np.asarray(retinaface[j])
    img01=Image.fromarray(a)
    print(img01)
    img01.show()
    img01.save(f'00{j}.jpg','JPEG')


# image_cut=cv2.imread("d:/project/DL/retina_data/01.png")
# retinaface=RetinaFace.extract_faces(img_path="d:/project/DL/retina_data/01.png",align=True,threshold=0.5)
# def extract_faces(img_path, threshold=0.9, model = None, align = True, allow_upscaling = True):
# print(retinaface[0].shape)  #(65, 47, 3)


