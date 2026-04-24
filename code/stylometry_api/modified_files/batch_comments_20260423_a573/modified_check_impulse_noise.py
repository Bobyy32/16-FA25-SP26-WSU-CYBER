from __future__ import print_function, division

import imgaug as ia
from imgaug import augmenters as iaa


def main():
    target_image = ia.data.quokka(0.5)
    noise_threshold = 0.01
    augmentation_configurations = [
        ("iaa.ImpulseNoise(p=0*noise_threshold)", iaa.ImpulseNoise(p=0*noise_threshold)),
        ("iaa.ImpulseNoise(p=1*noise_threshold)", iaa.ImpulseNoise(p=1*noise_threshold)),
        ("iaa.ImpulseNoise(p=2*noise_threshold)", iaa.ImpulseNoise(p=2*noise_threshold)),
        ("iaa.ImpulseNoise(p=3*noise_threshold)", iaa.ImpulseNoise(p=3*noise_threshold)),
        ("iaa.ImpulseNoise(p=(0*noise_threshold, 1*noise_threshold))", iaa.ImpulseNoise(p=(0*noise_threshold, 1*noise_threshold))),
        ("iaa.ImpulseNoise(p=[0*noise_threshold, 1*noise_threshold, 2*noise_threshold])", iaa.ImpulseNoise(p=[0*noise_threshold, 1*noise_threshold, 2*noise_threshold]))
    ]
    for descriptor, augmented_opener in augmentation_configurations:
        print(descriptor)
        noised_images = augmented_opener.augment_images([target_image] * 16)
        ia.imshow(ia.draw_grid(noised_images))


if __name__ == "__main__":
    main()