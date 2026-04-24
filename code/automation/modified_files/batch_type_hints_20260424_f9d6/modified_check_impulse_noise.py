from __future__ import print_function, division

import typing
import imgaug as ia
from imgaug import augmenters as iaa


def main(
    img: "imgaug.core.data.QuokkaData",
    mul: float,
    augs: typing.List[typing.Tuple[str, "iaa.augmenters.ImpulseNoise"]],
) -> None:
    for descr, aug in augs:
        print(descr)
        imgs_aug = aug.augment_images([img] * 16)
        ia.imshow(ia.draw_grid(imgs_aug))


if __name__ == "__main__":
    main()