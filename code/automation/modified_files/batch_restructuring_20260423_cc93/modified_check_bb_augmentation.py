from __future__ import print_function, division

import imageio
import numpy as np
from skimage import data

import imgaug as ia
from imgaug import augmenters as iaa


class ImageProcessingConstants:
    """Configuration constants for image processing operations."""
    
    # Grid configuration
    NB_ROWS = 10
    NB_COLS = 10
    HEIGHT = 256
    WIDTH = 256
    
    # Bounding box coordinates
    BB_X1 = 64
    BB_X2 = WIDTH - 64
    BB_Y1 = 64
    BB_Y2 = HEIGHT - 64
    
    # Transformation parameters
    ROTATION_ANGLE = 45


class KeypointGenerator:
    """Generates keypoints in a grid pattern."""
    
    @staticmethod
    def generate_grid_keypoints(nb_rows, nb_cols, height, width, 
                               bb_x1, bb_x2, bb_y1, bb_y2):
        """Generate grid coordinates and create keypoint objects."""
        keypoints = []
        
        for y in range(nb_rows):
            ycoord = int(bb_y1 + y * (bb_y2 - bb_y1) / max(1, nb_cols - 1))
            for x in range(nb_cols):
                xcoord = int(bb_x1 + x * (bb_x2 - bb_x1) / max(1, nb_rows - 1))
                keypoint = iaa.Keypoint(x=xcoord, y=ycoord)
                keypoints.append(keypoint)
        
        return ia.KeypointsOnImage(keypoints, shape=(height, width))


class BoundingBoxManager:
    """Manages bounding box objects."""
    
    @staticmethod
    def create_bounding_boxes(bounding_box, image_shape):
        """Create bounding box objects on image."""
        bbox = ia.BoundingBox(x1=bbox.x1, x2=bbox.x2, y1=bbox.y1, y2=bbox.y2)
        return ia.BoundingBoxesOnImage([bbox], shape=image_shape)


class ImageTransform:
    """Applies transformations to images and associated objects."""
    
    @staticmethod
    def apply_affine_transform(image, keypoints, bounding_boxes, 
                              rotation_angle):
        """Apply affine transformation with rotation."""
        seq = iaa.Affine(rotate=rotation_angle)
        seq_det = seq.to_deterministic()
        
        augmented_image = seq_det.augment_image(image)
        augmented_keypoints = seq_det.augment_keypoints(keypoints)
        augmented_bounding_boxes = seq_det.augment_bounding_boxes(bounding_boxes)
        
        return augmented_image, augmented_keypoints, augmented_bounding_boxes


class ImageComposer:
    """Composes images with keypoints and bounding boxes."""
    
    @staticmethod
    def draw_annotations(image, keypoints, bounding_boxes, 
                        draw_color=(255, 0, 0)):
        """Draw keypoints and bounding boxes on image."""
        annotated_image = np.copy(image)
        annotated_image = keypoints.draw_on_image(annotated_image)
        annotated_image = bounding_boxes.draw_on_image(annotated_image)
        return annotated_image


class ImageComparator:
    """Compares original and transformed images."""
    
    @staticmethod
    def compare_images(image_before, keypoints_before, bounding_boxes_before,
                       image_after, keypoints_after, bounding_boxes_after):
        """Create side-by-side comparison of original and augmented images."""
        combined_image = np.hstack([image_before, image_after])
        return combined_image


def main():
    """Main execution function for image processing pipeline."""
    try:
        # Initialize constants
        constants = ImageProcessingConstants()
        
        # Load and preprocess image
        image = data.astronaut()
        image = ia.imresize_single_image(image, (constants.HEIGHT, constants.WIDTH))
        image_shape = image.shape
        
        # Generate keypoints and bounding boxes
        keypoints = KeypointGenerator.generate_grid_keypoints(
            constants.NB_ROWS, 
            constants.NB_COLS, 
            constants.HEIGHT, 
            constants.WIDTH, 
            constants.BB_X1, 
            constants.BB_X2, 
            constants.BB_Y1, 
            constants.BB_Y2
        )
        
        bounding_boxes = BoundingBoxManager.create_bounding_boxes(
            ia.BoundingBox(x1=constants.BB_X1, x2=constants.BB_X2, 
                          y1=constants.BB_Y1, y2=constants.BB_Y2), 
            image_shape
        )
        
        # Apply transformations
        augmented_image, keypoints_aug, bounding_boxes_aug = ImageTransform.apply_affine_transform(
            image, 
            keypoints, 
            bounding_boxes, 
            constants.ROTATION_ANGLE
        )
        
        # Prepare display images
        image_before = ImageComposer.draw_annotations(
            image, 
            keypoints, 
            bounding_boxes, 
            draw_color=(255, 0, 0)
        )
        
        image_after = ImageComposer.draw_annotations(
            augmented_image, 
            keypoints_aug, 
            bounding_boxes_aug, 
            draw_color=(0, 255, 0)
        )
        
        # Display and save results
        combined_image = ImageComparator.compare_images(
            image_before, 
            keypoints, 
            bounding_boxes, 
            image_after, 
            keypoints_aug, 
            bounding_boxes_aug
        )
        
        ia.imshow(combined_image)
        imageio.imwrite("bb_aug.jpg", combined_image)
        
    except Exception as e:
        print(f"Error during processing: {e}")
        raise


if __name__ == "__main__":
    main()