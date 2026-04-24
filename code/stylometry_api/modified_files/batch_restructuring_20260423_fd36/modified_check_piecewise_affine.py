import numpy as np

import imgaug as ia
import imgaug.random as iarandom
from imgaug import augmenters as iaa

iarandom.seed(3)


def main():
    """Main function to process quokka image with keypoints and augmentations."""
    # Generate a quokka image
    image = ia.data.quokka(size=0.5)
    print(f"Image shape: {image.shape}")
    
    # Define keypoints for quokka detection
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
    
    # Define augmenters for image processing
    augs = [
        iaa.PiecewiseAffine(scale=0.05),
        iaa.PiecewiseAffine(scale=0.1),
        iaa.PiecewiseAffine(scale=0.2)
    ]
    
    # Display original image with keypoints
    print("Original image with keypoints:")
    ia.imshow(kps[0].draw_on_image(image))
    
    print("\n--- Augmented Images Per Image ---")
    
    for aug in augs:
        augmented_images = []
        for _ in range(16):
            # Make augmentation deterministic for consistency
            aug_det = aug.to_deterministic()
            img_aug = aug_det.augment_image(image)
            kps_aug = aug_det.augment_keypoints(kps)[0]
            
            # Draw augmented keypoints on augmented image
            img_aug_kps = draw_keypoints_on_image(kps_aug, img_aug)
            img_aug_kps = pad_image(img_aug_kps, padding=1, constant_values=255)
            
            augmented_images.append(img_aug_kps)
        
        print(f"Augmentation: {aug.name}")
        ia.imshow(draw_grid_from_images(augmented_images))


def draw_keypoints_on_image(kps, image, color=[0, 255, 0], size=3, copy=True, raise_if_out_of_image=False, border=50):
    """
    Helper function to draw keypoints on image with proper padding.
    
    Args:
        kps: KeyPointsOnImage object
        image: Input image
        color: Color to draw keypoints (default: green)
        size: Marker size (default: 3)
        copy: Whether to make a copy of the image (default: True)
        raise_if_out_of_image: Whether to raise exception for out of bounds
        border: Border size to pad image (default: 50)
    
    Returns:
        Processed image with keypoints drawn
    """
    # Make a copy if needed
    if copy:
        image = np.copy(image)
    
    # Pad image with border to handle edge keypoints
    image = np.pad(
        image,
        ((border, border), (border, border), (0, 0)),
        mode="constant",
        constant_values=0
    )
    
    height, width = image.shape[:2]
    
    for keypoint in kps.keypoints:
        y, x = keypoint.y + border, keypoint.x + border
        
        # Validate keypoint position
        if 0 <= y < height and 0 <= x < width:
            # Determine marker boundaries
            x1 = max(x - size//2, 0)
            x2 = min(x + 1 + size//2, width - 1)
            y1 = max(y - size//2, 0)
            y2 = min(y + 1 + size//2, height - 1)
            image[y1:y2, x1:x2] = color
        elif raise_if_out_of_image:
            raise Exception(
                f"Cannot draw keypoint x={x}, y={y} on image with shape {image.shape}."
            )
    
    return image


def pad_image(image, padding=1, constant_values=255):
    """
    Pad image with specified border and constant values.
    
    Args:
        image: Input image
        padding: Padding size (default: 1)
        constant_values: Value for padded border (default: 255)
    
    Returns:
        Padded image
    """
    return np.pad(image, ((padding, padding), (padding, padding), (0, 0)),
                  mode="constant", constant_values=constant_values)


def draw_grid_from_images(images):
    """
    Create a grid visualization of multiple images.
    
    Args:
        images: List of images to display
    
    Returns:
        Grid visualization
    """
    return ia.draw_grid(images)


if __name__ == "__main__":
    main()