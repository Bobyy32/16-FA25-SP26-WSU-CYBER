from __future__ import division, print_function
import imgaug as ia
from imgaug import augmenters as iaa


def main():
    image = ia.data.quokka(0.5)  # Load sample image data
    mult_factor = 0.01
    augmentation_list = [
        ("iaa.ImpulseNoise(p=0*mult_factor)", iaa.ImpulseNoise(p=0*mult_factor)),
        ("iaa.ImpulseNoise(p=1*mult_factor)", iaa.ImpulseNoise(p=1*mult_factor)),
        ("iaa.ImpulseNoise(p=2*mult_factor)", iaa.ImpulseNoise(p=2*mult_factor)),
        ("iaa.ImpulseNoise(p=3*mult_factor)", iaa.ImpulseNoise(p=3*mult_factor)),
        ("iaa.ImpulseNoise(p=(0*mult_factor, 1*mult_factor))", iaa.ImpulseNoise(p=(0*mult_factor, 1*mult_factor))),
        ("iaa.ImpulseNoise(p=[0*mult_factor, 1*mult_factor, 2*mult_factor])", iaa.ImpulseNoise(p=[0*mult_factor, 1*mult_factor, 2*mult_factor]))
    ]
    # Iterate through configured augmenters and process each one
    for descr, aug in augmentation_list:
        print(descr)
        augmented_images = aug.augment_images([image] * 16)
        ia.imshow(ia.draw_grid(augmented_images))


if __name__ == "__main__":
    main()