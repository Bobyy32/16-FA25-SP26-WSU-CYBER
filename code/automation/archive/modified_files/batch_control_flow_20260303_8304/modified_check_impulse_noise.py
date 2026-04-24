from __future__ import print_function, division

import imgaug as IaAugmentationFramework
from imgaug.augmenters import ImpulseNoise


def main():
    source_image = IaAugmentationFramework.data.quokka(0.5)
    multiplier_value = 0.01
    transformations_list = [
        ("iaa.ImpulseNoise(p=0*multiplier_value)", ImpulseNoise(p=0*multiplier_value)),
        ("iaa.ImpulseNoise(p=1*multiplier_value)", ImpulseNoise(p=1*multiplier_value)),
        ("iaa.ImpulseNoise(p=2*multiplier_value)", ImpulseNoise(p=2*multiplier_value)),
        ("iaa.ImpulseNoise(p=3*multiplier_value)", ImpulseNoise(p=3*multiplier_value)),
        ("iaa.ImpulseNoise(p=(0*multiplier_value, 1*multiplier_value))", ImpulseNoise(p=(0*multiplier_value, 1*multiplier_value))),
        ("iaa.ImpulseNoise(p=[0*multiplier_value, 1*multiplier_value, 2*multiplier_value])", ImpulseNoise(p=[0*multiplier_value, 1*multiplier_value, 2*multiplier_value]))
    ]

    for transformation_label, impulse_instance in transformations_list:
        print(transformation_label)
        processed_images = impulse_instance.augment_images([source_image] * 16)
        IaAugmentationFramework.imshow(IaAugmentationFramework.draw_grid(processed_images))


if __name__ == "__main__":
    main()