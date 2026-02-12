from __future__ import print_function, division

import imageio

import imgaug as ia
from imgaug import augmenters as iaa


def load_and_resize_image(url, resize_factor):
    """Load image from URL and resize it."""
    image = imageio.imread(url, format="jpg")
    return ia.imresize_single_image(image, resize_factor, "cubic")


def create_augmentations():
    """Create list of augmentations."""
    return [
        ("iaa.Clouds()", iaa.Clouds())
    ]


def process_augmentations(image, augmentations):
    """Apply augmentations to image."""
    for descr, aug in augmentations:
        print(descr)
        images_aug = aug.augment_images([image] * 64)
        ia.imshow(ia.draw_grid(images_aug))


def main():
    image_url = "https://upload.wikimedia.org/wikipedia/commons/8/89/Kukle%2CCzech_Republic..jpg"
    resize_factors = [0.1, 0.2, 1.0]
    
    for factor in resize_factors:
        resized_image = load_and_resize_image(image_url, factor)
        print(resized_image.shape)
        
        augmentations = create_augmentations()
        process_augmentations(resized_image, augmentations)


if __name__ == "__main__":
    main()