import random

import albumentations as A

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

annotations = r''
folder_save = r''

transform = A.Compose([
    A.Transpose(p=0.5),
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightness(limit=0.2, p=0.75),
    A.RandomContrast(limit=0.2, p=0.75),
    A.OneOf([
        A.MotionBlur(blur_limit=5),
        A.MedianBlur(blur_limit=5),
        A.GaussianBlur(blur_limit=5),
        A.GaussNoise(var_limit=(5.0, 30.0)),
    ], p=0.7),

    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),

])
list_image = []
for i in tqdm(range(333)):

    for gt in os.listdir(annotations):
        img_path = annotations + str("/") + str(gt)
        list_image.append(img_path)
index =1
for i,a  in enumerate(list_image):
    slect_image_1 = list_image[i]
    if (i + 1) < len(list_image):
        slect_image_2 = list_image[i + 1]
        print(slect_image_1, slect_image_1)
        img1_read = cv2.imread(slect_image_1.replace("\'",'/'))
        img2_read = cv2.imread(slect_image_2.replace("\'",'/'))
        print(img2_read)
        # height, width, _ = img2_read.shape
        # M = np.float32([[1, 0, 5], [0, 1, 5]])

        # perform the translation
        # img = cv2.warpAffine(img2_read, M, (width, height))
        image_h1, image_w1, c = img1_read.shape
        image_h2, image_w2, _ = img2_read.shape

        img1 = cv2.resize(img1_read, (image_h1, image_w1))
        img2 = cv2.resize(img2_read, (image_h2, image_w2))
        spitLit = random.randint(8, 15)

        image_new_1 = img1[int(image_h1/spitLit):image_h1 , int(image_h1/spitLit):image_h1 ]
        image_new_2 = img2[int(image_h2/spitLit):image_h2 , int(image_h2/spitLit):image_h2]
        transformed = transform(image=img1)
        transformed2 = transform(image=img2)

        transformed_image = transformed['image']
        transformed_image2 = transformed2['image']
        file_name1 = str(index) + "_yass1s" + str(i) + str("_.png")
        file_name2 = str(index) + "_yass1s_1" + str(i) + str("_.png")
        # addh = cv2.vconcat([transformed_image, transformed_image2])
        # height, width, _ = addh.shape

        # addh = cv2.hconcat([img1_read, img2_read])
        cv2.imwrite(os.path.join(folder_save, file_name1), transformed_image)
        cv2.imwrite(os.path.join(folder_save, file_name2), transformed_image2)
        index+=1

        # # img = cv2.resize(img, (int(image_w//2), (image_h//2)))
        # # img = cv2.resize(img, (512, 512))
        # M = np.float32([[1, 0, 25], [0, 1, 25]])
        #
        # # perform the translation
        # img = cv2.warpAffine(img, M, (image_w, image_h))
        # transformed = transform(image=img)
        #
        # transformed_image = transformed['image']
        #
        # name = os.path.join(save_path, f"{i}1axccasa{os.path.basename(gt)}")
        # img_name = name
        #
        #
        # cv2.imwrite(os.path.join(save_path,img_name), transformed_image)
