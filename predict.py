import csv

from main_former import Model

import time
import pandas as pd
import os
from tqdm import tqdm
import cv2
import numpy as np
models = []
checkpoints = [
    r'./checkpoints\convnext_large_fold4_224_bs_32\checkpoint-best.pth',
    r'./checkpoints\convnext_xlarge_fold1_224_bs_32\checkpoint-best.pth',
    r'./checkpoints\convnext_xlarge_fold2_224_bs_32\checkpoint-best.pth',
    r'./checkpoints\convnext_large_fold5_224_bs_12_\checkpoint-best.pth',
    r'./checkpoints\convnext_xlarge_fold3_224_bs_32\checkpoint-best.pth',
]
sizes = [224, 224, 224, 224, 224, 224, 224, 224, 224, 224]
names = ['convnext_large', 'convnext_xlarge', 'convnext_xlarge', 'convnext_large','convnext_xlarge']
print('Loading model ...')
classes = ['가구수정','걸레받이수정','곰팡이','꼬임','녹오염','들뜸','면불량','몰딩수정','반점','석고수정','오염','오타공','울음','이음부불량','창틀,문틀수정','터짐','틈새과다','피스','훼손']
path = r"W:\Paper-defect-type-classification\test"
for cp, s, name in tqdm(zip(checkpoints, sizes, names)):

    models.append(Model(name=name, checkpoint_model=cp))
result_test = "submission1.csv"
for filename in os.listdir(path):
    image = cv2.imread(path+str("/")+filename)
    outputs = []
    for i, model in enumerate(models):
        output = model.predict(image)
        outputs.append( output)
    score = 0.7 *(outputs[4]*0.4+outputs[1]*0.3+outputs[2]*0.3)+0.3 *(outputs[0]*0.5+outputs[3]*0.5)
    list_out = score.tolist()
    index = list_out.index(max(list_out))
    sublist = [x for x in list_out if x < max(list_out)]
    index2 = max(sublist)
    class_predict = classes[index]
    line = "TEST_" + filename.replace(".png", ""), class_predict
    with open(result_test, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(line)
