import numpy as np

# --- Additional methods for BoundingBoxesOnImage ---

def get_bbox_for_item(self, index):
    """Get the bounding box for a specific item index.
    
    Parameters
    ----------
    index : int
        Index of the bounding box to retrieve.
    
    Returns
    -------
    imgaug.augmntables.bbs.BoundingBox
        The bounding box at the given index.
    
    """
    if index < 0 or index >= len(self.bounding_boxes):
        raise IndexError(f"Index {index} out of range for {len(self)} bounding boxes.")
    return self.bounding_boxes[index]

def __setitem__(self, index, value):
    """Set the bounding box at the given index.
    
    Added in 0.4.0.

    Parameters
    ----------
    index : int
        Index of the bounding box to modify.
    
    value : imgaug.augmntables.bbs.BoundingBox
        Bounding box to set.
    
    Raises
    ------
    IndexError
        If `index` is out of range.
    
    """
    self.bounding_boxes[index] = value

# --- Additional BoundingBox class utility ---

class BoundingBox:
    @property
    def is_out_of_image(self, image):
        """Check if the bounding box is entirely outside the image.
        
        Parameters
        ----------
        image : array-like
            The image to check against.
        
        Returns
        -------
        bool
            True if out of image, False otherwise.
        
        """
        h, w = self.y2_int, self.x2_int
        return self.y2_int <= 0 or self.x2_int <= 0

# --- Helper method for drawing ---

def draw_bboxes_on_image(image, bounding_boxes, draw_func=None, **kwargs):
    """Draw bounding boxes on an image.

    Parameters
    ----------
    image : np.ndarray
        The image to draw on.
    bounding_boxes : BoundingBoxesOnImage
        Bounding boxes to draw.
    draw_func : function, optional
        Function to call for each bounding box.
    **kwargs
        Additional arguments passed to the drawing functions.
    
    Returns
    -------
    np.ndarray
        The image with bounding boxes drawn.
    
    """
    if draw_func is None:
        draw_func = _LabelOnImageDrawer().draw_on_image
    for bb in bounding_boxes:
        image = draw_func(image, bb, **kwargs)
    return image