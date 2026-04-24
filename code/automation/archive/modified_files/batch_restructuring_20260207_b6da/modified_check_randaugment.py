from __future__ import print_function, division, absolute_import

import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa


def _run_experiment_with_n_values(image, n_values):
    """Run augmentation experiment with different N values."""
    for n in n_values:
        print("N=%d" % (n,))

        images_aug = []
        for m in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
            images_aug.extend(
                iaa.RandAugment(n=n, m=m, random_state=1)(images=[image] * 10)
            )
        ia.imshow(ia.draw_grid(images_aug, cols=10))


def _run_experiment_with_m_values(image, m_values):
    """Run augmentation experiment with different M values."""
    for m in m_values:
        print("M=%d" % (m,))
        aug = iaa.RandAugment(m=m, random_state=1)

        images_aug = []
        for _ in np.arange(6):
            images_aug.extend(aug(images=[image] * 16))

        ia.imshow(ia.draw_grid(images_aug, cols=16, rows=6))


def main():
    image = ia.data.quokka(0.25)

    _run_experiment_with_n_values(image, [1, 2])
    _run_experiment_with_m_values(image, [0, 1, 2, 4, 8, 10])


if __name__ == "__main__":
    main()