@staticmethod
def from_0to1(arr_0to1, shape, min_value=0.0, max_value=1.0):
    """Create a heatmaps object from a ``[0.0, 1.0]`` float array.

    Parameters
    ----------
    arr_0to1 : (H,W) or (H,W,C) ndarray
        Heatmap(s) array, where ``H`` is the height, ``W`` is the width
        and ``C`` is the number of heatmap channels.
        Expected dtype is ``float32``.

    shape : tuple of ints
        Shape of the image on which the heatmap(s) is/are placed.
        **Not** the shape of the heatmap(s) array, unless it is identical
        to the image shape (note the likely difference between the arrays
        in the number of channels).
        If there is not a corresponding image, use the shape of the
        heatmaps array.

    min_value : float, optional
        Minimum value of the float heatmaps that the input array
        represents. This will usually be 0.0.

    max_value : float, optional
        Maximum value of the float heatmaps that the input array
        represents. This will usually be 1.0.

    Returns
    -------
    imgaug.augmentables.heatmaps.HeatmapsOnImage
        Heatmaps object.

    """
    heatmaps = HeatmapsOnImage(arr_0to1, shape,
                               min_value=0.0, max_value=1.0)
    heatmaps.min_value = min_value
    heatmaps.max_value = max_value
    return heatmaps