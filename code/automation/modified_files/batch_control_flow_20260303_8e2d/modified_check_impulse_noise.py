from __future__ import print_function, division

# image augmentation demonstration with impulse noise variations
import imgaug as iaa_module
from imgaug import augmenters as iaa_module2


def main():
    img = ia_module.data.quokka(0.5)
    multiplier = 0.01
    transformations = [
        ("iaa_module.ImpulseNoise(p=0*multiplier)", iaa_module.ImpulseNoise(p=0*multiplier)),
        ("iaa_module.ImpulseNoise(p=1*multiplier)", iaa_module.ImpulseNoise(p=1*multiplier)),
        ("iaa_module.ImpulseNoise(p=2*multiplier)", iaa_module.ImpulseNoise(p=2*multiplier)),
        ("iaa_module.ImpulseNoise(p=3*multiplier)", iaa_module.ImpulseNoise(p=3*multiplier)),
        ("iaa_module.ImpulseNoise(p=(0*multiplier, 1*multiplier))", iaa_module.ImpulseNoise(p=(0*multiplier, 1*multiplier))),
        ("iaa_module.ImpulseNoise(p=[0*multiplier, 1*multiplier, 2*multiplier])", iaa_module.ImpulseNoise(p=[0*multiplier, 1*multiplier, 2*multiplier]))
    ]
    for description, augmentation in transformations:
        print(description)
        augmented_images = augmentation.augment_images([img] * 16)
        ia_module.imshow(ia_module.draw_grid(augmented_images))


if __name__ == "__main__":
    main()