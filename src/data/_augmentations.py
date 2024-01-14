import os
import sys

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from conf import ExpConfig


def get_transforms(config: ExpConfig, phase: str = "train"):
    if phase == "train":
        aug_list = [
            A.Rotate(limit=45, p=0.5),
            A.RandomScale(
                scale_limit=(0.8, 1.25), interpolation=cv2.INTER_NEAREST, p=0.5
            ),
            A.RandomBrightnessContrast(
                p=0.5,
            ),
            A.GaussianBlur(p=0.2),
            A.MotionBlur(p=0.2),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
            A.Resize(
                height=config.img_height,
                width=config.img_width,
                interpolation=cv2.INTER_NEAREST,
                p=1,
            ),
            ToTensorV2(transpose_mask=True),
        ]
    else:
        aug_list = [
            A.Resize(
                height=config.img_height,
                width=config.img_width,
                interpolation=cv2.INTER_NEAREST,
                p=1,
            ),
            ToTensorV2(transpose_mask=True),
        ]
    aug = A.Compose(aug_list)
    return aug
