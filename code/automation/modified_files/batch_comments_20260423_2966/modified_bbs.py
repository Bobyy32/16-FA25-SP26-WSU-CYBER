def draw_labels(self, image):
        """Draw labels for all bounding boxes.

        Parameters
        ----------
        image : numpy array
            Image to draw labels on.

        Returns
        -------
        numpy array
            Image with labels drawn.
        """
        from .label import LabelDrawer

        if self.bounding_boxes is None or len(self.bounding_boxes) == 0:
            return image

        for bounding_box in self.bounding_boxes:
            self._label_drawer.draw_label(
                image, bounding_box, self.bounding_boxes)

        return image