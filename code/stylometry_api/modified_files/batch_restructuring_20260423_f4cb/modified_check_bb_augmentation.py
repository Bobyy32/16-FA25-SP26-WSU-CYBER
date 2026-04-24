from __future__ import print_function, division

import imageio
import numpy as np
from skimage import data

import imgaug as ia
from imgaug import augmenters as iaa


class ImagePreprocessingEngine:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        
    def load_source(self):
        return data.astronaut()
    
    def prepare_dimensions(self, image):
        return ia.imresize_single_image(image, (self.height, self.width))


class SpatialLayoutManager:
    def __init__(self, x1, x2, y1, y2, rows, cols, height, width):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.rows = rows
        self.cols = cols
        self.height = height
        self.width = width
        
    def calculate_coordinates(self):
        coordinates = []
        for y in range(self.rows):
            ycoord = self.y1 + int(y * (self.y2 - self.y1) / (self.cols - 1))
            for x in range(self.cols):
                xcoord = self.x1 + int(x * (self.x2 - self.x1) / (self.rows - 1))
                coordinates.append((xcoord, ycoord))
        return set(coordinates)
    
    def convert_to_keypoints(self, coordinates):
        return [ia.Keypoint(x=xcoord, y=ycoord) for (xcoord, ycoord) in coordinates]
    
    def create_keypoint_collection(self, keypoints, shape):
        return ia.KeypointsOnImage(keypoints, shape=shape)
    
    def get_image_shape(self):
        return (self.height, self.width)


class BoundingBoxManager:
    def __init__(self, x1, x2, y1, y2, shape):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.shape = shape
        
    def create_single_box(self):
        return ia.BoundingBox(x1=self.x1, x2=self.x2, y1=self.y1, y2=self.y2)
    
    def create_box_collection(self, boxes, shape):
        return ia.BoundingBoxesOnImage(boxes, shape=self.shape)


class ImageTransformationManager:
    def __init__(self, rotation_amount=45):
        self.rotation_amount = rotation_amount
        
    def create_transform(self):
        return iaa.Affine(rotate=self.rotation_amount)
    
    def create_deterministic_version(self, transform):
        return transform.to_deterministic()
    
    def apply_augmentation(self, transform, image):
        return transform.augment_image(image)
    
    def augment_keypoints(self, transform, keypoints):
        return transform.augment_keypoints(keypoints)
    
    def augment_boxes(self, transform, boxes):
        return transform.augment_bounding_boxes(boxes)


class ImageVisualizationEngine:
    def __init__(self, image_before, image_after):
        self.image_before = image_before
        self.image_after = image_after
        
    def prepare_display(self):
        return np.hstack([self.image_before, self.image_after])
    
    def save_visualization(self, image):
        imageio.imwrite("bb_aug.jpg", image)


class ProcessingPipeline:
    def __init__(self, height, width, x1, x2, y1, y2, nb_rows, nb_cols):
        self.preprocessor = ImagePreprocessingEngine(height, width)
        self.layout = SpatialLayoutManager(x1, x2, y1, y2, nb_rows, nb_cols, height, width)
        self.bbox_manager = BoundingBoxManager(x1, x2, y1, y2, (height, width))
        self.transformer = ImageTransformationManager(45)
        self.visualizer = ImageVisualizationEngine(None, None)
        
    def execute(self):
        image = self.preprocessor.load_source()
        image = self.preprocessor.prepare_dimensions(image)
        
        coordinates = self.layout.calculate_coordinates()
        coordinates = self.layout.convert_to_keypoints(coordinates)
        coordinates = self.layout.create_keypoint_collection(coordinates, image.shape)
        
        bounding_box = self.bbox_manager.create_single_box()
        bounding_boxes = self.bbox_manager.create_box_collection([bounding_box], image.shape)
        
        transform = self.transformer.create_deterministic_version(self.transformer.create_transform())
        
        image_aug = self.transformer.apply_augmentation(transform, image)
        keypoints_aug = self.transformer.augment_keypoints(transform, coordinates)
        boxes_aug = self.transformer.augment_boxes(transform, bounding_boxes)
        
        image_before = np.copy(image)
        image_before = np.hstack([image_before, coordinates])
        image_before = np.hstack([image_before, bounding_boxes])
        
        image_after = np.copy(image_aug)
        image_after = np.hstack([image_after, keypoints_aug])
        image_after = np.hstack([image_after, boxes_aug])
        
        display_image = self.visualizer.prepare_display(image_after)
        self.visualizer.save_visualization(display_image)


class ApplicationRunner:
    def __init__(self):
        self.height = HEIGHT
        self.width = WIDTH
        self.x1 = BB_X1
        self.x2 = BB_X2
        self.y1 = BB_Y1
        self.y2 = BB_Y2
        self.nb_rows = NB_ROWS
        self.nb_cols = NB_COLS
        
    def initialize(self):
        self.pipeline = ProcessingPipeline(
            self.height, self.width, 
            self.x1, self.x2, self.y1, self.y2,
            self.nb_rows, self.nb_cols
        )
        return self.pipeline


if __name__ == "__main__":
    app = ApplicationRunner()
    pipeline = app.initialize()
    pipeline.execute()