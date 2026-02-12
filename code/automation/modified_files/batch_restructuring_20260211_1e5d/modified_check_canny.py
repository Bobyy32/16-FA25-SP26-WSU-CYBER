from __future__ import print_function, division
import imgaug as ia
import imgaug.augmenters as iaa


def _apply_and_show(augmenter, image, grid_size=(5, 5)):
    """Applies an augmenter to an image and displays the result in a grid."""
    images = [image] * (grid_size[0] * grid_size[1])
    augmented_images = augmenter(images=images)
    grid = ia.draw_grid(augmented_images)
    ia.imshow(grid)


def _create_and_test_canny(
        alpha,
        colorizer=None,
        sobel_kernel_size=None,
        test_description=""):
    """Creates a Canny augmenter, prints a description, and applies it to an image."""
    print(f"{test_description}, alpha={alpha}")
    image = ia.quokka_square((128, 128))
    kwargs = {"alpha": alpha}
    if colorizer:
        kwargs["colorizer"] = colorizer
    if sobel_kernel_size is not None:
        kwargs["sobel_kernel_size"] = sobel_kernel_size
    aug = iaa.Canny(**kwargs)
    _apply_and_show(aug, image)


def main():
    """Main function to demonstrate Canny edge detection augmentation."""
    black_and_white_colorizer = iaa.RandomColorsBinaryImageColorizer(
        color_true=255, color_false=0)

    _create_and_test_canny(
        alpha=1.0,
        colorizer=black_and_white_colorizer,
        test_description="alpha=1.0, black and white")

    _create_and_test_canny(
        alpha=1.0,
        test_description="alpha=1.0, random color")

    _create_and_test_canny(
        alpha=1.0,
        sobel_kernel_size=[3, 7],
        colorizer=black_and_white_colorizer,
        test_description="alpha=1.0, sobel ksize=[3, 7], black and white")

    _create_and_test_canny(
        alpha=1.0,
        sobel_kernel_size=3,
        colorizer=black_and_white_colorizer,
        test_description="alpha=1.0, sobel ksize=3, black and white")

    _create_and_test_canny(
        alpha=1.0,
        test_description="fully random")


if __name__ == "__main__":
    main()