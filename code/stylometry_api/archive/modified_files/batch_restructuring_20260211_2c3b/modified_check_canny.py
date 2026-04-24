from __future__ import print_function, division
import imgaug as ia
import imgaug.augmenters as iaa

def create_binary_colorizer():
    return iaa.RandomColorsBinaryImageColorizer(color_true=255, color_false=0)

def display_canny_demo(description, colorizer=None, sobel_kernel_size=None):
    print(description)
    image = ia.quokka_square((128, 128))
    augmenter = iaa.Canny(alpha=1.0, colorizer=colorizer, sobel_kernel_size=sobel_kernel_size)
    ia.imshow(ia.draw_grid(augmenter(images=[image] * (5*5))))

def main():
    binary_colorizer = create_binary_colorizer()
    display_canny_demo("alpha=1.0, black and white", colorizer=binary_colorizer)
    display_canny_demo("alpha=1.0, random color", colorizer=None)
    display_canny_demo("alpha=1.0, sobel ksize=[3, 13], black and white", colorizer=binary_colorizer, sobel_kernel_size=[3, 7])
    display_canny_demo("alpha=1.0, sobel ksize=3, black and white", colorizer=binary_colorizer, sobel_kernel_size=3)
    display_canny_demo("fully random", colorizer=None)

if __name__ == "__main__":
    main()