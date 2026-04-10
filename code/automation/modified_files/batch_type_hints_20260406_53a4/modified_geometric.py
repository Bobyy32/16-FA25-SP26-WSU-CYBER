class Jigsaw(IABaseAugmenter):
    """
    Augmentation to split an image into ``nb_rows`` x ``nb_cols`` cells
    and shuffle the cells around (Jigsaw puzzle).

    This augmenter can be used for data augmentation in training pipelines.
    It randomly splits images into a grid of cells and then rearranges them
    randomly. This can be used to make the model more robust to different
    spatial arrangements of the same objects.

    .. versionadded:: 0.4.0

    Parameters
    ----------
    nb_rows : tuple of int or int, optional
        The number of rows to split the image into. If a tuple is provided,
        a uniform distribution is used between the values. Default is
        ``(3, 10)``.
    nb_cols : tuple of int or int, optional
        The number of columns to split the image into. If a tuple is provided,
        a uniform distribution is used between the values. Default is
        ``(3, 10)``.
    max_steps : tuple of int or int, optional
        The maximum number of steps the cells are moved by in each direction.
        If a tuple is provided, a uniform distribution is used between the
        values. Default is ``1``.
    allow_pad : bool, optional
        Whether to allow padding the image to the size of the cells.
        If ``True``, the image will be padded to fit the cells.
        If ``False``, the image will not be padded and the cells may be
        partially cut off. Default is ``True``.

    Attributes
    ----------
    random_state : int, optional
        A seed for reproducibility of the random augmentations.
    deterministic : bool, optional
        If ``True``, the augmentation will be deterministic based on the
        ``random_state``.
    seed : int, optional
        Deprecated. Use ``random_state`` instead.
    name : str, optional
        The name of the augmentation for logging and debugging purposes.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.Jigsaw(nb_rows=10, nb_cols=10)

    Create a jigsaw augmenter that splits images into ``10x10`` cells
    and shifts them around by ``0`` to ``2`` steps (default setting).

    >>> aug = iaa.Jigsaw(nb_rows=(1, 4), nb_cols=(1, 4))

    Create a jigsaw augmenter that splits each image into ``1`` to ``4``
    cells along each axis.

    >>> aug = iaa.Jigsaw(nb_rows=10, nb_cols=10, max_steps=(1, 5))

    Create a jigsaw augmenter that moves the cells in each image by a random
    amount between ``1`` and ``5`` times (decided per image). Some images will
    be barely changed, some will be fairly distorted.

    Note
    ----
    The ``Jigsaw`` augmenter can also work on other batch data types like
    heatmaps, segmentation maps, and keypoints. It cannot work with
    bounding boxes, polygons, or line strings yet.
    """