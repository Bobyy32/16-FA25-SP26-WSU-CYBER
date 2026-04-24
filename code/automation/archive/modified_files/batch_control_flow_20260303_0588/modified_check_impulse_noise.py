from __future__ import print_function, division

import imgaug as image_augmentation_lib
from imgaug import augmenters as augmentation_modules


def execute_main():
    source_image = image_augmentation_lib.data.quokka(0.5)
    scaling_factor = 0.01
    configuration_list = [
        ("iaa.ImpulseNoise(p=0*scaling_factor)", image_augmentation_lib.ImpulseNoise(p=0*scaling_factor)),
        ("iaa.ImpulseNoise(p=1*scaling_factor)", image_augmentation_lib.ImpulseNoise(p=1*scaling_factor)),
        ("iaa.ImpulseNoise(p=2*scaling_factor)", image_augmentation_lib.ImpulseNoise(p=2*scaling_factor)),
        ("iaa.ImpulseNoise(p=3*scaling_factor)", image_augmentation_lib.ImpulseNoise(p=3*scaling_factor)),
        ("iaa.ImpulseNoise(p=(0*scaling_factor, 1*scaling_factor))", image_augmentation_lib.ImpulseNoise(p=(0*scaling_factor, 1*scaling_factor))),
        ("iaa.ImpulseNoise(p=[0*scaling_factor, 1*scaling_factor, 2*scaling_factor])", image_augmentation_lib.ImpulseNoise(p=[0*scaling_factor, 1*scaling_factor, 2*scaling_factor]))
    ]
    processed_data = []
    [processed_data.append({
        'descriptor_string': item_str,
        'augmentation_instance': instance_obj
    }) for item_str, instance_obj in configuration_list if True]
    for item in processed_data:
        print(item['descriptor_string'])
        augmented_images = item['augmentation_instance'].augment_images([source_image] * 16)
        image_augmentation_lib.imshow(image_augmentation_lib.draw_grid(augmented_images))


if __name__ == "__main__":
    execute_main()