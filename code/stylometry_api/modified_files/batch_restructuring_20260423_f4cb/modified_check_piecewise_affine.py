from __future__ import print_function, division

import numpy as np
import imgaug as ia
import imgaug.random as iarandom
from imgaug import augmenters as iaa


class ImageDataProcessor():
    def __init__(self):
        self.image = None
        self.image_shape = None
    
    def generate_quokka_image(self, size=0.5):
        self.image = ia.data.quokka(size=size)
        return self.image
    
    def get_image_dimensions(self, image):
        self.image_shape = (image.shape[0], image.shape[1])
        return self.image_shape


class AugmentationPipeline():
    def __init__(self):
        self.augmentations = []
        self.scale = 0.05
    
    def add_augmentation(self, scale):
        aug = iaa.PiecewiseAffine(scale=scale)
        self.augmentations.append(aug)
    
    def apply_augmentations(self, image, keypoints):
        image_aug = []
        for aug in self.augmentations:
            aug_det = aug.to_deterministic()
            img_aug = aug_det.augment_image(image)
            kps_aug = aug_det.augment_keypoints(keypoints)[0]
            img_aug_kps = ia.draw_grid(kps_aug, img_aug)
            img_aug_kps = np.pad(img_aug_kps, ((1, 1), (1, 1), (0, 0)), mode="constant", constant_values=255)
            image_aug.append(img_aug_kps)
        return image_aug


class RandomAugmentPipeline(AugmentationPipeline):
    def __init__(self):
        super().__init__()
        self.augmentations = []
        self.scale = 0.05
        self.scale_1 = 0.1
        self.scale_2 = 0.2
    
    def initialize_augmentations(self):
        self.augmentations = [
            iaa.PiecewiseAffine(scale=self.scale),
            iaa.PiecewiseAffine(scale=self.scale_1),
            iaa.PiecewiseAffine(scale=self.scale_2)
        ]


class KeypointManager():
    def __init__(self):
        self.keypoints = None
    
    def create_initial_keypoints(self, image_shape):
        kps_list = [
            ia.Keypoint(x=123, y=102),
            ia.Keypoint(x=182, y=98),
            ia.Keypoint(x=155, y=134),
            ia.Keypoint(x=-20, y=20)
        ]
        self.keypoints = ia.KeypointsOnImage(kps_list, shape=image_shape)
        return self.keypoints
    
    def augment_keypoints(self, keypoints, augmentation):
        aug_det = augmentation.to_deterministic()
        kps_aug = aug_det.augment_keypoints(keypoints)[0]
        return kps_aug


class ImagePaddingHelper():
    def __init__(self):
        self.border = 50
        self.padding_width = 0
    
    def apply_image_padding(self, image):
        image = ia.draw_grid(image)
        image = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode="constant", constant_values=255)
        return image


def keypoints_draw_on_image(kps, image, color=[0, 255, 0], size=3, copy=True, raise_if_out_of_image=False, border=50):
    if copy:
        image = np.copy(image)

    image = np.pad(
        image,
        ((border, border), (border, border), (0, 0)),
        mode="constant",
        constant_values=0
    )

    height, width = image.shape[0:2]

    for keypoint in kps.keypoints:
        y, x = keypoint.y + border, keypoint.x + border
        if 0 <= y < height and 0 <= x < width:
            x1 = max(x - size//2, 0)
            x2 = min(x + 1 + size//2, width - 1)
            y1 = max(y - size//2, 0)
            y2 = min(y + 1 + size//2, height - 1)
            image[y1:y2, x1:x2] = color
        else:
            if raise_if_out_of_image:
                raise Exception("Cannot draw keypoint x=%d, y=%d on image with shape %s." % (y, x, image.shape))

    return image


def render_keypoint_images(keypoint_images):
    result = ia.draw_grid(keypoint_images)
    return result


def visualize_keypoints_on_image(image, keypoints, size=3, color=[0, 255, 0], border=50):
    image = np.pad(image, ((border, border), (border, border), (0, 0)), mode="constant", constant_values=0)
    height, width = image.shape[0:2]
    
    for keypoint in keypoints:
        y, x = keypoint.y, keypoint.x
        if 0 <= y < height and 0 <= x < width:
            x1 = max(x - size//2, 0)
            x2 = min(x + 1 + size//2, width - 1)
            y1 = max(y - size//2, 0)
            y2 = min(y + 1 + size//2, height - 1)
            image[y1:y2, x1:x2] = color

    return image


def main():
    ia.random.seed(3)
    image = ia.data.quokka(size=0.5)
    print(image.shape)
    kps = ia.KeypointsOnImage(
        [
            ia.Keypoint(x=123, y=102),
            ia.Keypoint(x=182, y=98),
            ia.Keypoint(x=155, y=134),
            ia.Keypoint(x=-20, y=20)
        ],
        shape=(image.shape[0], image.shape[1])
    )
    print("image shape:", image.shape)

    augs = [
        iaa.PiecewiseAffine(scale=0.05),
        iaa.PiecewiseAffine(scale=0.1),
        iaa.PiecewiseAffine(scale=0.2)
    ]

    ia.imshow(kps.draw_on_image(image))
    print("-----------------")
    print("Random aug per image")
    print("-----------------")
    
    for aug in augs:
        images_aug = []
        for _ in range(16):
            aug_det = aug.to_deterministic()
            img_aug = aug_det.augment_image(image)
            kps_aug = aug_det.augment_keypoints(kps)[0]
            img_aug_kps = keypoints_draw_on_image(kps_aug, img_aug)
            img_aug_kps = np.pad(img_aug_kps, ((1, 1), (1, 1), (0, 0)), mode="constant", constant_values=255)
            images_aug.append(img_aug_kps)
        print(aug.name)
        ia.imshow(render_keypoint_images(images_aug))


if __name__ == "__main__":
    main()