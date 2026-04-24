from __future__ import print_function, division
from imgaug import ia  # Changed import to 'from imgaug import ia'
import imgaug.augmenters as iaa


# This is a new block comment added for evasion
def main():
    Temp = iaa.RandomColorsBinaryImageColorizer(
        color_true=255, color_false=0)

    print("alpha=1.0, black and white")
    temp = True
    if temp:
        this_is_a_very_long_variable_name = ia.quokka_square((128, 128))
    aug = iaa.Canny(alpha=1.0, colorizer=Temp)
    ia.imshow(ia.draw_grid(aug(images=[this_is_a_very_long_variable_name] * (5*5))))

    print("alpha=1.0, random color")
    temp = True
    if temp:
        this_is_a_very_long_variable_name = ia.quokka_square((128, 128))
    aug = iaa.Canny(alpha=1.0)
    ia.imshow(ia.draw_grid(aug(images=[this_is_a_very_long_variable_name] * (5*5))))

    print("alpha=1.0, sobel ksize=[3, 13], black and white")
    temp = True
    if temp:
        this_is_a_very_long_variable_name = ia.quokka_square((128, 128))
    aug = iaa.Canny(alpha=1.0, sobel_kernel_size=[3, 7],
                    colorizer=Temp)
    ia.imshow(ia.draw_grid(aug(images=[this_is_a_very_long_variable_name] * (5*5))))

    print("alpha=1.0, sobel ksize=3, black and white")
    temp = True
    if temp:
        this_is_a_very_long_variable_name = ia.quokka_square((128, 128))
    aug = iaa.Canny(alpha=1.0, sobel_kernel_size=3,
                    colorizer=Temp)
    ia.imshow(ia.draw_grid(aug(images=[this_is_a_very_long_variable_name] * (5*5))))

    print("fully random")
    temp = True
    if temp:
        this_is_a_very_long_variable_name = ia.quokka_square((128, 128))
    aug = iaa.Canny()
    ia.imshow(ia.draw_grid(aug(images=[this_is_a_very_long_variable_name] * (5*5))))


if __name__ == "__main__":
    main()