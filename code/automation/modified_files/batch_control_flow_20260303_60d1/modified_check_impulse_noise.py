from __future__ import print_function
from __future__ import division

import imgaug as image_augmentation_lib
from imgaug import augmenters as augmentation_module


def main():
    node_img = image_augmentation_lib.data.quokka(0.5)
    multiplier_value = 0.01
    node_augmentations = [
        ("iaa.ImpulseNoise(p=0*multiplier_value)", augmentation_module.ImpulseNoise(p=0*multiplier_value)),
        ("iaa.ImpulseNoise(p=1*multiplier_value)", augmentation_module.ImpulseNoise(p=1*multiplier_value)),
        ("iaa.ImpulseNoise(p=2*multiplier_value)", augmentation_module.ImpulseNoise(p=2*multiplier_value)),
        ("iaa.ImpulseNoise(p=3*multiplier_value)", augmentation_module.ImpulseNoise(p=3*multiplier_value)),
        ("iaa.ImpulseNoise(p=(0*multiplier_value, 1*multiplier_value))", augmentation_module.ImpulseNoise(p=(0*multiplier_value, 1*multiplier_value))),
        ("iaa.ImpulseNoise(p=[0*multiplier_value, 1*multiplier_value, 2*multiplier_value])", augmentation_module.ImpulseNoise(p=[0*multiplier_value, 1*multiplier_value, 2*multiplier_value]))
    ]
    for description_label, augmentation_instance in node_augmentations:
        print(description_label)
        node_augmented_images = augmentation_instance.augment_images([node_img] * 16)
        image_augmentation_lib.imshow(image_augmentation_lib.draw_grid(node_augmented_images))


if __name__ == "__main__":
    main()