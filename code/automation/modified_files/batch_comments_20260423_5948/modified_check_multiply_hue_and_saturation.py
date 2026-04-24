from __future__ import print_function, division
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa

def main():
    # Initialize input source
    image = ia.quokka_square((128, 128))
    images_aug = []
    
    augmenters = [
        iaa.Affine(scale_limit=(-0.1, 0.1), translate_limit=(-0.15, 0.15), rotate_limit=(-10, 10))
    ]
    
    for _ in range(100):
        img_aug = iaa.augment_images(image, augmenters=augmenters, n=1)
        images_aug.append(img_aug)

    ia.draw_grid(images_aug, rows=4)
    ia.imshow(draw_grid)

if __name__ == "__main__":
    main()