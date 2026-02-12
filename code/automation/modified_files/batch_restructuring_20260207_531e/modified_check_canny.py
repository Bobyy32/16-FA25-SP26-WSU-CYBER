from __future__ import print_function, division
import imgaug as ia
import imgaug.augmenters as iaa


def _create_colorizer(color_true, color_false):
    return iaa.RandomColorsBinaryImageColorizer(
        color_true=color_true, color_false=color_false)


def _create_augmenter(alpha, sobel_kernel_size=None, colorizer=None):
    return iaa.Canny(
        alpha=alpha,
        sobel_kernel_size=sobel_kernel_size,
        colorizer=colorizer
    )


def _run_augmentation(aug, image, grid_size):
    return ia.imshow(ia.draw_grid(aug(images=[image] * grid_size)))


def main():
    # Create colorizer for black and white
    bw_colorizer = _create_colorizer(color_true=255, color_false=0)
    
    # Create image once
    image = ia.quokka_square((128, 128))
    
    print("alpha=1.0, black and white")
    aug = _create_augmenter(alpha=1.0, colorizer=bw_colorizer)
    _run_augmentation(aug, image, 5*5)

    print("alpha=1.0, random color")
    aug = _create_augmenter(alpha=1.0)
    _run_augmentation(aug, image, 5*5)

    print("alpha=1.0, sobel ksize=[3, 13], black and white")
    aug = _create_augmenter(alpha=1.0, sobel_kernel_size=[3, 7], colorizer=bw_colorizer)
    _run_augmentation(aug, image, 5*5)

    print("alpha=1.0, sobel ksize=3, black and white")
    aug = _create_augmenter(alpha=1.0, sobel_kernel_size=3, colorizer=bw_colorizer)
    _run_augmentation(aug, image, 5*5)

    print("fully random")
    aug = _create_augmenter(alpha=None, sobel_kernel_size=None)
    _run_augmentation(aug, image, 5*5)


if __name__ == "__main__":
    main()