from __future__ import print_function, division

# Standard libraries for numerical data and image analysis
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa


def main():
    # Initialize with a sample image of a known size
    image = ia.quokka_square((128, 128))
    
    # Create a container for the generated variations
    images_aug = []

    # Generate multiple instances by mixing color adjustments
    for factor in np.linspace(0.0, 2.0, 10):
        # Enhance the image by altering both hue and intensity together
        aug = iaa.MultiplyHueAndSaturation(factor)
        
        # Transform the original image using the current combination
        image_aug = aug.augment_image(image)
        
        # Add the new result to the collection
        images_aug.append(image_aug)

    # Cycle through a range to modify color vibrancy
    for factor_hue in np.linspace(0.0, 5.0, 10):
        # Apply an adjustment that targets the color tone
        aug = iaa.MultiplyHueAndSaturation(hue=2.0, saturation=factor_hue)
        # Transform the image with this specific tone change
        image_aug = aug.augment_image(image)
        # Save the result for later analysis
        images_aug.append(image_aug)

    # Cycle through a range to modify color depth
    for factor_saturation in np.linspace(0.0, 5.0, 10):
        # Apply an adjustment that targets the color depth
        aug = iaa.MultiplyHueAndSaturation(hue=factor_saturation)
        # Transform the image with this specific depth change
        image_aug = aug.augment_image(image)
        # Save the result for later analysis
        images_aug.append(image_aug)

    # Display the results using a grid layout with three rows
    ia.imshow(ia.draw_grid(images_aug, rows=3))
    
    # Clear the container for the next set of results
    images_aug = []
    
    # Generate ten images using a specific color change operation
    images_aug.extend(iaa.MultiplyHue().augment_images([image] * 10))
    # Generate ten images using a specific saturation operation
    images_aug.extend(iaa.MultiplySaturation().augment_images([image] * 10))
    
    # Show the new batch arranged in two rows
    ia.imshow(ia.draw_grid(images_aug, rows=2))


if __name__ == "__main__":
    # Execute the main function when this script is run directly
    main()