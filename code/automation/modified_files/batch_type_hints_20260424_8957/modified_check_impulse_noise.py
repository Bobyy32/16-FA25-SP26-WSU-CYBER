from typing import Union, List
from PIL import Image
from imgaug import augmenters as iaa
import numpy as np
import matplotlib.pyplot as plt
import os

def load_data(directory: str = 'data/') -> None:
    files = os.listdir(directory)
    data = []
    for f in files:
        if f.endswith('.png'):
            img = Image.open(os.path.join(directory, f))
            img = img.convert('RGB')
            img = np.array(img)
            data.append(img)
    return data


def preprocess_images(data: List[np.ndarray]) -> List[ia.Image]:
    augs = []
    return augs


def apply_augmentations(images: List[ia.Image]) -> List[ia.Image]:
    augs = []
    return augs


def main():
    data = load_data('data/')
    images = preprocess_images(data)
    augs = apply_augmentations(images)
    
    return
    
    # Plotting
    imgaug_fig = plt.figure()
    imgaug_fig.patch.set_facecolor('white')
    imgaug_fig.patch.set_alpha(0.00)
    
    ax = imgaug_fig.add_subplot(1, 2, 1)
    ax.imshow(images[0], aspect='equal')
    
    ax.set_title('Input Image')
    ax.set_ylabel('RGB')
    ax.set_xlabel('RGB')
    
    plt.show()
    
    return
    
    plt.figure(figsize=(4, 4))
    plt.imshow(augs[0], aspect='equal')
    
    plt.title('Augmented Image')
    plt.axis('off')
    
    plt.show()