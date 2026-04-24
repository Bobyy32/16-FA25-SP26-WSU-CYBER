from __future__ import print_function, division

import numpy as np
import imgaug as ia
import imgaug.random as iarandom
from imgaug import augmenters as iaa

iarandom.seed(3)


def main():
    # Load and prepare base image
    image = ia.data.quokka(size=0.5)
    print(image.shape)

    # Create initial keypoints
    kps = [
        ia.KeypointsOnImage(
            [
                ia.Keypoint(x=123, y=102),
                ia.Keypoint(x=182, y=98),
                ia.Keypoint(x=155, y=134),
                ia.Keypoint(x=-20, y=20)
            ],
            shape=(image.shape[0], image.shape[1])
        )
    ]
    print("image shape:", image.shape)

    # Define augmentations to apply
    augs = [
        iaa.PiecewiseAffine(scale=0.05),
        iaa.PiecewiseAffine(scale=0.1),
        iaa.PiecewiseAffine(scale=0.2)
    ]

    # Display original keypoints on image
    ia.imshow(kps[0].draw_on_image(image))

    print("-----------------")
    print("Random aug per image")
    print("-----------------")

    # Process each augmentation
    for aug in augs:
        # Collect augmented images for this augmentation
        images_aug = []
        # Apply augmentation 16 times
        for _ in range(16):
            # Make augmentation deterministic for reproducibility
            aug_det = aug.to_deterministic()
            
            # Augment image and keypoints
            img_aug = aug_det.augment_image(image)
            kps_aug = aug_det.augment_keypoints(kps)[0]
            
            # Draw keypoints on augmented image
            img_aug_kps = keypoints_draw_on_image(kps_aug, img_aug)
            
            # Pad the image to ensure consistent output dimensions
            img_aug_kps = np.pad(
                img_aug_kps,
                ((1, 1), (1, 1), (0, 0)),
                mode="constant",
                constant_values=255
            )
            images_aug.append(img_aug_kps)

        # Display results for this augmentation
        print(aug.name)
        ia.imshow(ia.draw_grid(images_aug))


def keypoints_draw_on_image(kps, image, color=[0, 255, 0], size=3, copy=True, raise_if_out_of_image=False, border=50):
    """
    Draw keypoints as colored circles on the provided image.

    Args:
        kps: KeypointsOnImage object
        image: Numpy array image
        color: RGB color for keypoints
        size: Size of the circle
        copy: Whether to copy the image
        raise_if_out_of_image: If True, raise exception for out of bounds keypoints
        border: Padding border for image

    Returns:
        Modified image with keypoints drawn
    """
    # Copy image if requested
    if copy:
        image = np.copy(image)

    # Apply padding to image
    image = np.pad(
        image,
        ((border, border), (border, border), (0, 0)),
        mode="constant",
        constant_values=0
    )

    height, width = image.shape[0:2]

    # Draw each keypoint
    for keypoint in kps.keypoints:
        # Calculate coordinates after padding
        y, x = keypoint.y + border, keypoint.x + border
        
        # Check if keypoint is within image bounds
        if 0 <= y < height and 0 <= x < width:
            # Define circle boundaries
            x1 = max(x - size//2, 0)
            x2 = min(x + 1 + size//2, width - 1)
            y1 = max(y - size//2, 0)
            y2 = min(y + 1 + size//2, height - 1)
            image[y1:y2, x1:x2] = color
        else:
            if raise_if_out_of_image:
                raise Exception("Cannot draw keypoint x=%d, y=%d on image with shape %s." % (y, x, image.shape))

    return image


if __name__ == "__main__":
    main()