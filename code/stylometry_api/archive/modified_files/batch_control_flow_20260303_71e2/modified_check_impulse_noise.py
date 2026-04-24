from __future__ import print_function, division

import imgaug as image_augmenter_lib
from imgaug import augmenters as augmentation_definitions


def run_pipeline():
    input_image = image_augmenter_lib.data.quokka(0.5)
    intensity_multiplier = 0.01
    transformation_list = [
        ("ImpulseNoise_0p_level", image_augmenter_lib.augmenters.ImpulseNoise(p=0*intensity_multiplier)),
        ("ImpulseNoise_1p_level", image_augmenter_lib.augmenters.ImpulseNoise(p=1*intensity_multiplier)),
        ("ImpulseNoise_2p_level", image_augmenter_lib.augmenters.ImpulseNoise(p=2*intensity_multiplier)),
        ("ImpulseNoise_3p_level", image_augmenter_lib.augmenters.ImpulseNoise(p=3*intensity_multiplier)),
        ("ImpulseNoise_tuple_range", image_augmenter_lib.augmenters.ImpulseNoise(p=(0*intensity_multiplier, 1*intensity_multiplier))),
        ("ImpulseNoise_list_range", image_augmenter_lib.augmenters.ImpulseNoise(p=[0*intensity_multiplier, 1*intensity_multiplier, 2*intensity_multiplier]))
    ]
    for label_name, transform_object in transformation_list:
        # Display the current augmentation configuration details
        print(label_name)
        augmented_outputs = transform_object.augment_images([input_image] * 16)
        # Show the grid visualization of enhanced images
        image_augmenter_lib.imshow(image_augmenter_lib.draw_grid(augmented_outputs))


if __name__ == "__main__":
    run_pipeline()