import cv2
from matplotlib import image
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
        a=np.array(retinaface[j])
        img01=Image.fromarray(a)
        img01=img01.resize((150,150))
        print(img01)
        img01.show()
        img01.save(f'd:/project/DL/retina_data/00{j}.jpg','JPEG')
        # cv2.imwrite(f'd:/project/DL/retina_data/00{j}.jpg',image)

'''
사이즈문제 아님 리사이즈로 사이즈 다 맞춰봄
파일번호 문제 아님 +1해봄 003까지 생성됨

'''

