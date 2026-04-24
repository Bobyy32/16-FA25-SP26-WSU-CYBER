import imgaug as ia
import imgaug.augmenters as iaa
import imageio
import cv2
import numpy as np


def load_image(url):
    return imageio.imread(url)


def apply_cartoon(image):
    augs = [image] + iaa.Cartoon()(images=[image] * 15)
    return augs


def display_images(augs):
    ia.imshow(ia.draw_grid(augs, 4, 4))


def main():
    image = load_image(urls_medium[1])
    augs = apply_cartoon(image)
    display_images(augs)


if __name__ == "__main__":
    main()