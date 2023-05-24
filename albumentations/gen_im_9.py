import albumentations as A
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#몰딩수정
annotations = r''
folder_save = r''

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(),
    A.RandomRotate90(),
    A.Flip(),
    A.SomeOf([
        A.Blur(),
        A.GaussianBlur(),
        A.MotionBlur(),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.1),
        A.Transpose(),
        A.VerticalFlip(),
    ], n=5, p=0.85),
    A.Transpose(),
    A.VerticalFlip(),
    A.OneOf([
        A.Blur(),
        A.GaussianBlur(),
        A.MotionBlur(),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.1),
        A.Transpose(),
        A.VerticalFlip(),
    ], p=0.5),
])

for i in tqdm(range(3)):

    for gt in os.listdir(annotations):
                img = cv2.imread(annotations + str("/") + str(gt))
                image_h, image_w, c = img.shape
                # img = cv2.resize(img, (int(image_w//2), (image_h//2)))
                # img = cv2.resize(img, (512, 512))
                transformed = transform(image=img)

                transformed_image = transformed['image']

                name = os.path.join(folder_save, f"{i}axxxxaxxx1{os.path.basename(gt)}")
                img_name = name


                cv2.imwrite(os.path.join(folder_save,img_name), transformed_image)

