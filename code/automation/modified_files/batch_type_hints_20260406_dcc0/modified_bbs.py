def __getitem__(self, indices):
    return BoundingBoxesOnImage(self.bounding_boxes[indices], shape=self.shape)

def deepcopy(self, bounding_boxes=None, shape=None):
    from copy import deepcopy
    if bounding_boxes is None:
        bounding_boxes = deepcopy(self.bounding_boxes)
    if shape is None:
        shape = tuple(self.shape)
    return BoundingBoxesOnImage(bounding_boxes, shape)