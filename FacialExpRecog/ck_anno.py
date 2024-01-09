import os
import pandas as pd
import numpy as np

ck_folder = "D:\EXP_Data\CK plus\CK+48"

# annotation file 열 순서 : 해당 이미지 경로 [path] / 레이블 [label]

emo_img_path = []
        
# 파일 읽기
exp_list = os.listdir(ck_folder) # ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
exp_path = [ os.path.join(ck_folder,exp) for exp in exp_list ]

for one_exp_path in exp_path:
    image_list = os.listdir(one_exp_path)
    image_path = [os.path.join(one_exp_path, img) for img in image_list ]
    

    
    emo_img_path.extend(image_path)  # extend를 사용하여 1차원 리스트로 확장
    emo_list = [i.split("\\")[-2] for i in emo_img_path]
    
    emo_list * len(emo_img_path)
    
    
df = pd.DataFrame({'path': emo_img_path, 'label': emo_list})
    
    
len(df)


output_path = "C:\\data_analysis\\CK_annotation.csv"
df.to_csv(output_path, index=False)



############### index ################

# label_mapping = {
#     "anger": 0,
#     "disgust": 1,
#     "fear": 2,
#     "happy": 3,
#     "sadness": 4,
#     "surprise": 5,
#     "contempt": 6
# }

# df['label'] = df['label'].map(label_mapping)