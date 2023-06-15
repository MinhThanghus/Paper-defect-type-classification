import albumentations as A
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


annotations = r''
folder_save = r''


transform = A.Compose([

    A.MedianBlur(p=0.5),

    # A.ToGray(p=0.1),
    # A.CLAHE(p=0.01),
    # A.RandomBrightnessContrast(contrast_limit=0.0 , brightness_limit=0.2, p=0.5),
    # A.RandomResizedCrop(height=460,width=460),
    A.RandomGamma(p=0.1),
    A.Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=20, p=0.2),
    A.HorizontalFlip(p=1),
    # A.HorizontalFlip(p=0.5),
    A.Flip(p=0.5),
    A.RandomRotate90(p=1),
    A.VerticalFlip(p=0.5),  # for sauce
    A.Transpose(p=0.5),
])

for i in tqdm(range(3)): range 300 with class have number image least, range 3 with class have number image Most

    for gt in os.listdir(annotations):
                img = cv2.imread(annotations + str("/") + str(gt))
                image_h, image_w, c = img.shape
                # img = cv2.resize(img, (int(image_w//2), (image_h//2)))
                # img = cv2.resize(img, (512, 512))
                transformed = transform(image=img)

                transformed_image = transformed['image']

                name = os.path.join(folder_save, f"{i}1saxxxsa{os.path.basename(gt)}")
                img_name = name


                cv2.imwrite(os.path.join(folder_save,img_name), transformed_image)
