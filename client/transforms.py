from albumentations import (Cutout, Compose, Normalize, RandomRotate90, HorizontalFlip,
                            VerticalFlip, ShiftScaleRotate, Transpose, OneOf, IAAAdditiveGaussianNoise,
                            GaussNoise, RandomGamma, RandomContrast, RandomBrightness, HueSaturationValue,
                            RandomBrightnessContrast, Lambda, NoOp, CenterCrop, Resize
                            )
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
)
from albumentations.pytorch import ToTensor
from albumentations.pytorch.transforms import ToTensorV2

import cv2
import albumentations as A
mean_img = [0.22363983, 0.18190407, 0.2523437]
std_img = [0.32451536, 0.2956294,  0.31335256]
transform_train = Compose([
    HorizontalFlip(p=0.5),
    ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05,
                     rotate_limit=20, p=0.3, border_mode=cv2.BORDER_REPLICATE),
    Transpose(p=0.5),
    Normalize(mean=mean_img, std=std_img, max_pixel_value=255.0, p=1.0),
    ToTensor()
])

transform_test = Compose([
    HorizontalFlip(p=1),
    Transpose(p=0),
    Normalize(mean=mean_img, std=std_img, max_pixel_value=255.0, p=1.0),
    ToTensor()
])

transform_train_detection = A.Compose([
    A.Flip(0.2),
    A.RandomRotate90(0.25),
    MotionBlur(p=0.2),
    MedianBlur(blur_limit=3, p=0.1),
    Blur(blur_limit=3, p=0.1),
    ToTensorV2(p=1.0)
], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


transform_test_detection = A.Compose([
    ToTensorV2(p=1.0)
], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
