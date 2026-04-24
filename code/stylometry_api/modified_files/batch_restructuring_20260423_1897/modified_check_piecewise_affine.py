```python
import numpy as np
from PIL import Image
from albumentations import IA, Compose, HorizontalFlip, VerticalFlip, RandomBrightness, RandomContrast, RandomBrightness, RandomContrast, ShiftScaleRotate, Scale, Rotate, Translate
from albumentations.augmentations.transforms import *
from albumentations import random

def keypoints_draw_on_image(image, kps, **kwargs):
    image_aug = None
    img_aug_kps = None
    img = image
    kps = kps
    color = kwargs.get('color', 128)
    border = kwargs.get('border', 50)
    size = kwargs.get('size', 20)
    copy = kwargs.get('copy', True)
    
    if copy:
        image_aug = np.copy(image)
        img = image_aug
        kps = np.copy(kps)

    if border > 0:
        img = np.pad(img, ((border, border), (border, border), (0, 0)), mode='constant', constant_values=color)
        height, width = img.shape[:2]
    else:
        height, width = image.shape[:2]
    
    for k in range(16):
        kps_aug = kps[k]
        x1 = kps_aug[0]
        x2 = x1 + size
        if height > border:
            y1 = kps_aug[1]
            y2 = y1 + size
            if border > height or border > width:
                y1 -= border
                y2 -= border
        else:
            y1 = kps_aug[1]
            y2 = y1 + size
            
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, y1, x2, y2
        
        x1, y1, x2, y2 = x1, selection