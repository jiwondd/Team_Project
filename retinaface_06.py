import cv2
from matplotlib import image
import numpy as np
import os
import matplotlib.pyplot as plt
from retinaface import RetinaFace
from deepface import DeepFace
from PIL import Image
import imageio
from skimage.transform import resize

path_dir="d:/project/DL/retina_data/"
file_list=os.listdir(path_dir)
# print(file_list[0]) #01.png

print(len(file_list)) #583

file_name_list=[]
for i in range(len(file_list)):
    file_name_list.append(file_list[i])
    
# print(file_name_list)

A=[]        
for i in range(len(file_name_list)):
    retinaface=RetinaFace.extract_faces(img_path=f"d:/project/DL/retina_data/{i+1}",align=True)
    A.append(retinaface)
    
# print(A)
    
for i in range(len(A)):
        
    for j in range(len(A[i])):
        
        img=np.array(A[i][j])
        img=Image.fromarray(img)
        img=img.resize((512,512))
        print(img)
        img.show()
        # cv2.imwrite(f'cut{i}_{j}.jpg',A[i][j])
        imageio.imwrite(f'd:/project/DL/aligned/0000{i}_{j}.png',img)
        
    # a=np.array(retinaface[j])
    # img01=Image.fromarray(a)
    # img01=img01.resize((150,150))
    # print(img01)
    # img01.show()
    # img01.save(f'd:/project/DL/retina_data/00{j}.jpg','JPEG')
    # cv2.imwrite(f'd:/project/DL/retina_data/00{j}.jpg',image)

