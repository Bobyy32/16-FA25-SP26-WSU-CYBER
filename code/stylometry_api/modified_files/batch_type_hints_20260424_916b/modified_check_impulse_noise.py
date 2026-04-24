from __future__ import print_function, division

import imgaug as ia
from imgaug import augmenters as iaa
from typing import Any


def main(img: Any, mul: Any, augs: Any) -> Any:
    imgs_aug = [img] * 16
    for descr, aug in augs:
        print(descr)
        imgs_aug = aug.augment_images(imgs_aug)
        ia.imshow(ia.draw_grid(imgs_aug))


if __name__ == "__main__":
    main()