import albumentations as A
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 몰딩수정
annotations = r''
folder_save = r''

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.OneOf([
        A.Downscale(scale_min=0.75, scale_max=0.95,
                    interpolation=dict(upscale=cv2.INTER_LINEAR, downscale=cv2.INTER_AREA), p=0.1),
        A.Downscale(scale_min=0.75, scale_max=0.95,
                    interpolation=dict(upscale=cv2.INTER_LANCZOS4, downscale=cv2.INTER_AREA), p=0.1),
        A.Downscale(scale_min=0.75, scale_max=0.95,
                    interpolation=dict(upscale=cv2.INTER_LINEAR, downscale=cv2.INTER_LINEAR), p=0.8),
    ], p=0.125),
    A.OneOf([
        A.RandomToneCurve(scale=0.3, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.2), contrast_limit=(-0.4, 0.5), brightness_by_max=True,
                                   always_apply=False, p=0.5)
    ], p=0.5),
    A.OneOf(
        [
            A.ShiftScaleRotate(shift_limit=None, scale_limit=[-0.15, 0.15], rotate_limit=[-30, 30],
                               interpolation=cv2.INTER_LINEAR,
                               border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=None, shift_limit_x=[-0.1, 0.1],
                               shift_limit_y=[-0.2, 0.2], rotate_method='largest_box', p=0.6),
            A.ElasticTransform(alpha=1, sigma=20, alpha_affine=10, interpolation=cv2.INTER_LINEAR,
                               border_mode=cv2.BORDER_CONSTANT,
                               value=0, mask_value=None, approximate=False, same_dxdy=False, p=0.2),
            A.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=cv2.INTER_LINEAR,
                             border_mode=cv2.BORDER_CONSTANT,
                             value=0, mask_value=None, normalized=True, p=0.2),
        ], p=0.5),
    A.CoarseDropout(max_holes=3, max_height=0.15, max_width=0.25, min_holes=1, min_height=0.05, min_width=0.1,
                    fill_value=0, mask_fill_value=None, p=0.25),
], p=0.9
)

for i in tqdm(range(3)):

    for gt in os.listdir(annotations):
        img = cv2.imread(annotations + str("/") + str(gt))
        image_h, image_w, c = img.shape
        # img = cv2.resize(img, (int(image_w//2), (image_h//2)))
        # img = cv2.resize(img, (512, 512))
        transformed = transform(image=img)

        transformed_image = transformed['image']

        name = os.path.join(folder_save, f"{i}aaxxxxaxx1{os.path.basename(gt)}")
        img_name = name

        cv2.imwrite(os.path.join(folder_save, img_name), transformed_image)
