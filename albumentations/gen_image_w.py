
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
    A.ShiftScaleRotate(rotate_limit=45,shift_limit=0,interpolation=2,border_mode=0,value=0,mask_value=0,p=0.7),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),

])

for i in tqdm(range(3)):

    for gt in os.listdir(annotations):
        img = cv2.imread(annotations + str("/") + str(gt))
        image_h, image_w, c = img.shape
        # img = cv2.resize(img, (int(image_w//2), (image_h//2)))
        # img = cv2.resize(img, (512, 512))
        image_new_1 = img[int(image_h/random.randint(8,20)):image_h , int(image_h/random.randint(8,20)):image_h ]
        # M = np.float32([[1, 0, 25], [0, 1, 25]])

        # perform the translation
        # img = cv2.warpAffine(img, M, (image_w, image_h))
        transformed = transform(image=image_new_1)

        transformed_image = transformed['image']

        name = os.path.join(folder_save, f"{i}aaax1a{os.path.basename(gt)}")
        img_name = name

        cv2.imwrite(os.path.join(folder_save, img_name), transformed_image)
