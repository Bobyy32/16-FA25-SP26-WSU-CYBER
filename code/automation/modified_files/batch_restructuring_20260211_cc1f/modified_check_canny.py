from __future__ import print_function, division
import imgaug as ia
import imgaug.augmenters as iaa


def f0():
    a = iaa.RandomColorsBinaryImageColorizer(color_true=255, color_false=0)
    print("alpha=1.0, black and white")
    b = ia.quokka_square((128, 128))
    c = iaa.Canny(alpha=1.0, colorizer=a)
    ia.imshow(ia.draw_grid(c(images=[b] * (5*5))))
    print("alpha=1.0, random color")
    d = ia.quokka_square((128, 128))
    e = iaa.Canny(alpha=1.0)
    ia.imshow(ia.draw_grid(e(images=[d] * (5*5))))
    print("alpha=1.0, sobel ksize=[3, 13], black and white")
    g = ia.quokka_square((128, 128))
    h = iaa.Canny(alpha=1.0, sobel_kernel_size=[3, 7], colorizer=a)
    ia.imshow(ia.draw_grid(h(images=[g] * (5*5))))
    print("alpha=1.0, sobel ksize=3, black and white")
    i = ia.quokka_square((128, 128))
    j = iaa.Canny(alpha=1.0, sobel_kernel_size=3, colorizer=a)
    ia.imshow(ia.draw_grid(j(images=[i] * (5*5))))
    print("fully random")
    k = ia.quokka_square((128, 128))
    l = iaa.Canny()
    ia.imshow(ia.draw_grid(l(images=[k] * (5*5))))


if __name__ == "__main__":
    f0()