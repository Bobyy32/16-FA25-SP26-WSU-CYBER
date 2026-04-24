import numpy as np
from PIL import Image
from imgaug.augmentables.ats import AttributesOnImage
from imgaug.augmentables.bbs import BoundingBoxesOnImage
from imgaug import augs
from imgaug.augmentables.polygons import PolygonsOnImage

def draw_polygons_with_labels(image, polygons, labels, font_size=20, background_color=(255, 0, 0), text_color=(0, 0, 0)):
    """
    Draws polygons with text labels on the image.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    polygons : PolygonsOnImage
        Polygons to draw.
    labels : list or str
        Labels to draw on polygons.
    font_size : int
        Text size.
    background_color : tuple
        Background color for the label.
    text_color : tuple
        Color for the text.

    Returns
    -------
    image : np.ndarray
        Image with drawn polygons and labels.
    """
    image = image.copy()
    
    # Initialize drawing attributes for polygonal labels
    label_drawer = PolygonLabelDrawer(
        size=3,
        color_bg=background_color,
        color_text=text_color,
        size_text=font_size,
        alpha=1.0,
        raise_if_out_of_image=False
    )

    # For each polygon, draw it with its corresponding label
    for polygon, label in zip(polygons, labels):
        x1, y1, x2, y2 = polygon.bbox
        image = label_drawer.draw_on_image(image, BoundingBoxOnImage(
            x1=x1, y1=y1, x2=x2, y2=y2,
            label=str(label),
            bbox=polygon.to_bbox()
        ))
    
    return image


class PolygonLabelDrawer:
    """
    A class that draws polygons with labels similar to LabelOnImageDrawer.
    """
    
    def __init__(self, size=3, color_bg=(0, 255, 0), color_text=None,
                 size_text=20, alpha=1.0, raise_if_out_of_image=False):
        self.size = size
        self.color_bg = color_bg
        self.color_text = color_text
        self.size_text = size_text
        self.alpha = alpha
        self.raise_if_out_of_image = raise_if_out_of_image

    def draw_on_image(self, image, bounding_box):
        image = image.copy()
        self.draw_on_image_(image, bounding_box)
        return image
    
    def draw_on_image_(self, image, bounding_box):
        x1, y1, x2, y2 = bounding_box.to_array()
        x1, y1, x2, y2 = np.clip([x1, y1, x2, y2], 0, max(image.shape) - 1)

        label_arr = self._draw_label_arr(
            label=str(bounding_box.label),
            height=x2 - x1 + 1,
            width=y2 - y1 + 1,
            nb_channels=image.shape[-1],
            dtype=image.dtype,
            color_text=self.color_text,
            color_bg=self.color_bg,
            size_text=self.size_text
        )

        image = np.zeros_like(image)
        image[y1:y2, x1:x2, :] = label_arr
        return image

    @classmethod
    def _do_raise_if_out_of_image(cls, image, bounding_box):
        if bounding_box.is_out_of_image(image):
            raise Exception(
                "Cannot draw label for polygon outside image bounds. "
                f"Bounding box: {bounding_box}, Image shape: {image.shape}."
            )

    def _preprocess_colors(self):
        color_bg = self.color_bg
        if self.color_bg is not None:
            color_bg = np.uint8(color_bg)

        color_text = self.color_text
        if self.color_text is not None:
            color_text = np.uint8(color_text)
        else:
            gray = (0.299 * color_bg[0]
                    + 0.587 * color_bg[1]
                    + 0.114 * color_bg[2])
            color_text = np.full((3,),
                                 0 if gray > 128 else 255,
                                 dtype=np.uint8)

        return color_text, color_bg

    @staticmethod
    def _draw_label_arr(label, height, width, nb_channels, dtype,
                        color_text, color_bg, size_text):
        label_arr = np.zeros((height, width, nb_channels), dtype=dtype)
        label_arr[...] = color_bg.reshape((1, 1, -1))
        label_arr = augs.draw_text(label_arr, x=2, y=2,
                                    text=str(label),
                                    color=color_text,
                                    size=size_text)
        return label_arr

    def _blend_label_arr_with_image_(self, image, label_arr, x1, y1, x2, y2):
        alpha = self.alpha
        if alpha >= 0.99:
            image[y1:y2, x1:x2, :] = label_arr
        else:
            input_dtype = image.dtype
            foreground = label_arr.astype(np.float64)
            background = image[y1:y2, x1:x2, :].astype(np.float64)
            blend = (1 - alpha) * background + alpha * foreground
            blend = np.clip(blend, 0, 255).astype(input_dtype)
            image[y1:y2, x1:x2, :] = blend
        return image


def convert_polygon_to_bbox(polygon):
    """
    Converts a polygon to its corresponding bounding box.
    """
    return BoundingBoxOnImage(
        x1=polygon.x1_int,
        y1=polygon.y1_int,
        x2=polygon.x2_int,
        y2=polygon.y2_int,
        label=str(polygon),
        bbox=polygon
    )