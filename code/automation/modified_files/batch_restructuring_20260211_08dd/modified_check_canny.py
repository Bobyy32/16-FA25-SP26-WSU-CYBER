import imgaug
import imgaug.augmenters

def a():
    b = imgaug.augmenters.RandomColorsBinaryImageColorizer(
        color_true =  255 , color_false =  0
    )
    imgaug.augmenters.Canny(
        alpha =  1.0 , colorizer =  b
    )
    c = imgaug.quokka_square((128, 128))
    d = imgaug.augmenters.Canny(
        alpha =  1.0, colorizer =  b
    )
    e = [c] * (5*5)
    f = d(images =  e)
    imgaug.imshow(
        imgaug.draw_grid(f)
    )
    imgaug.augmenters.Canny(
        alpha =  1.0
    )
    c = imgaug.quokka_square((128, 128))
    d = imgaug.augmenters.Canny(
        alpha =  1.0
    )
    e = [c] * (5*5)
    f = d(images =  e)
    imgaug.imshow(
        imgaug.draw_grid(f)
    )
    imgaug.augmenters.Canny(
        alpha =  1.0, sobel_kernel_size =  [3, 7], colorizer =  b
    )
    c = imgaug.quokka_square((128, 128))
    d = imgaug.augmenters.Canny(
        alpha =  1.0, sobel_kernel_size =  [3, 7], colorizer =  b
    )
    e = [c] * (5*5)
    f = d(images =  e)
    imgaug.imshow(
        imgaug.draw_grid(f)
    )
    imgaug.augmenters.Canny(
        alpha =  1.0, sobel_kernel_size =  3, colorizer =  b
    )
    c = imgaug.quokka_square((128, 128))
    d = imgaug.augmenters.Canny(
        alpha =  1.0, sobel_kernel_size =  3, colorizer =  b
    )
    e = [c] * (5*5)
    f = d(images =  e)
    imgaug.imshow(
        imgaug.draw_grid(f)
    )
    imgaug.augmenters.Canny()
    c = imgaug.quokka_square((128, 128))
    d = imgaug.augmenters.Canny()
    e = [c] * (5*5)
    f = d(images =  e)
    imgaug.imshow(
        imgaug.draw_grid(f)
    )


if __name__ == "__main__":
    a()