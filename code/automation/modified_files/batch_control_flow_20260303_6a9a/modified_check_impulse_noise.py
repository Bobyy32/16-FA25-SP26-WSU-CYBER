from __future__ import print_function, division

import imgaug as ia
from imgaug import augmenters as iaa


def main():
    image = ia.data.quokka(0.5)  # renamed from 'img' to 'image'
    multiplication_factor = 0.01  # renamed from 'mul' to 'multiplier'
    
    augmentation_specifications = {
        "case_zero": ("iaa.ImpulseNoise(p=0*multiplication_factor)", iaa.ImpulseNoise(p=0*multiplication_factor)),
        "Case_One": ("iaa.ImpulseNoise(p=1*multiplication_factor)", iaa.ImpulseNoise(p=1*multiplication_factor)),
        "case_two": ("iaa.ImpulseNoise(p=2*multiplication_factor)", iaa.ImpulseNoise(p=2*multiplication_factor)),
        "case_three": ("iaa.ImpulseNoise(p=3*multiplication_factor)", iaa.ImpulseNoise(p=3*multiplication_factor)),
        "Case_BothZeroOne": ("iaa.ImpulseNoise(p=(0*multiplication_factor, 1*multiplication_factor))", iaa.ImpulseNoise(p=(0*multiplication_factor, 1*multiplication_factor))),
        "case_list_three": ("iaa.ImpulseNoise(p=[0*multiplication_factor, 1*multiplication_factor, 2*multiplication_factor])", iaa.ImpulseNoise(p=[0*multiplication_factor, 1*multiplication_factor, 2*multiplication_factor]))
    }

    # iterate through spec items for visualization
    for name_str, (description, augment_obj) in augmentation_specifications.items():
        print(description)
        augmented_set = augment_obj.augment_images([image] * 16)
        ia.imshow(ia.draw_grid(augmented_set))


if __name__ == "__main__":
    main()