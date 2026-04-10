def main():
    from imgaug import augmenters as iaa
    from imgaug import io as ia
    image = ia.quokka_square((128, 128))
    images_aug = []
    
    for mul in (0, 0.5, 1, 1.5, 2):
        aug = iaa.MultiplyHueAndSaturation(mul)
        images_aug.append(aug.augment_image(image))
    
    for mul_hue in (0, 1, 2, 3, 4, 5):
        aug = iaa.MultiplyHueAndSaturation(mul_hue=mul_hue)
        images_aug.append(aug.augment_image(image))
    
    for mul_sat in (0, 1, 2, 3, 4, 5):
        aug = iaa.MultiplyHueAndSaturation(mul_saturation=mul_sat)
        images_aug.append(aug.augment_image(image))
    
    ia.imshow(ia.draw_grid(images_aug, rows=3))
    ia.imshow(ia.draw_grid(
        ia.stack([iaa.MultiplyHue().augment_image(image)] * 3, 
                 iaa.MultiplySaturation().augment_image(image)] * 3), 
        rows=2))


if __name__ == "__main__":
    main()