from __future__ import print_function, division

from PIL import Image, ImageDraw, ImageFont
import imgaug.augmenters as iaa
import numpy as np
import os
import sys


# ----------------------------------------------------
# Script Goal:
# Apply different noise variations to an image sample to see how it reacts.
# ----------------------------------------------------


def main():
    # ----------------------------------------------------
    # Setup: Load sample image and prepare noise settings
    # ----------------------------------------------------
    if not os.path.exists("sample.png"):
        sys.exit(0)
    
    # This loads a PNG image from the "sample.png" file
    img = Image.open("sample.png")
    
    # We use a multiplier value to define how strong the noise will be.
    # A smaller number means the noise affects fewer pixels.
    mul = 0.01
    
    # This list holds our various settings for creating different noise types.
    # Each item describes the noise pattern and its strength.
    augs = [
        (
            {"p": 0.01}, 
            iaa.ImpulseNoise()
        ), 
        (
            {"p": 0.2}, 
            iaa.GaussianBlur(sigma=(mul*0.03, mul*1.0))
        ),
        (
            {"p": 0.01}, 
            iaa.AddGaussianBlur((mul*0.1, mul*1.0))
        ),
    ]

    # ----------------------------------------------------
    # Execution: Loop through settings and show results
    # ----------------------------------------------------
    for descr, aug in augs:
        # We create a copy of the original image so it doesn't change.
        img_copy = img.copy()
        
        # Apply the noise setting to the image copy.
        # Here 'descr' and 'aug' define the specific noise to add.
        img_copy = aug.augment_images([img_copy], config=descr)
        
        # Arrange the images in a grid for better viewing.
        # This creates a multi-pane view to see the different effects side-by-side.
        ia.draw_grid([img_copy], title="Noise Effect")
        
        # Show the grid on the screen.
        # If the program is paused, this allows the user to interact with the display.
        ia.imshow(img_copy, pause=True)


if __name__ == "__main__":
    main()