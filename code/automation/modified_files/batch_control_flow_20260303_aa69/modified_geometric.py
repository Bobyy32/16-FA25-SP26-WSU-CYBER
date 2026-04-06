import numpy as np
import imgaug.augmenters as iaa


def elastic_transform_image(image, alpha=1.0, sigma=20.0, nb_samples=15):
    """
    Apply Elastic Transformation to an image.
    
    Parameters:
    -----------
    image : numpy.ndarray or PIL.Image
        Input image (grayscale or RGB)
    alpha : float
        Standard deviation for the random sampling of the displacement map.
        Larger values create more dramatic distortions.
    sigma : float
        The size of the local window in pixel units used for Gaussian smoothing
        of the displacement field.
    nb_samples : int
        The number of pixels to draw samples at (the larger this is, the smoother
        the distortion will be).

    Returns:
    --------
    numpy.ndarray
        Transformed image
    """
    
    # Convert PIL Image to numpy array if needed
    if isinstance(image, PIL.Image):
        image = np.array(image)
    
    # Validate input dimensions
    if image.ndim == 2:
        image = image[np.newaxis, ...]
    elif image.shape[-1] not in (3, 4):
        raise ValueError(
            f"Input image must have shape (H, W) or (H, W, C) with C=3/4. "
            f"Got shape: {image.shape}"
        )
    
    # Handle different dtypes appropriately
    if not np.issubdtype(image.dtype, np.floating):
        image = image.astype(np.float32)
    
    # Create augmenter
    elastic_aug = iaa.ElasticTransformation(
        alpha=alpha,
        sigma=sigma,
        nb_samples=nb_samples
    )
    
    try:
        # Apply the augmentation
        augmented = elastic_aug.augment_image(image)
        
        # Ensure output is normalized back to [0, 1] range if needed
        if augmented.max() > 255 or (augmented.min() < 0 and augmented.max() < 1):
            image_max = np.iinfo(image.dtype).max if np.issubdtype(
                image.dtype, np.integer) else 1.0
            augmented = normalized_img(augmented, image_max)
            
        return augmented
        
    except Exception as e:
        # Handle specific errors that may occur during augmentation
        raise ElasticTransformationError(
            f"Error applying elastic transformation: {str(e)}",
            input_shape=image.shape,
            input_dtype=image.dtype,
            parameters={"alpha": alpha, "sigma": sigma, "nb_samples": nb_samples}
        )


def process_image_batch(image_batch, params):
    """
    Process a batch of images with consistent elastic transformation.

    Parameters:
    -----------
    image_batch : numpy.ndarray
        Batch of images with shape (N, H, W) or (N, H, W, C)
    params : dict
        Dictionary containing 'alpha', 'sigma', 'nb_samples'

    Returns:
    --------
    numpy.ndarray
        Augmented batch of images
    """
    
    alpha = params['alpha']
    sigma = params['sigma']
    nb_samples = params['nb_samples']
    
    # Create augmenter with fixed parameters
    elastic_aug = iaa.ElasticTransformation(
        alpha=alpha,
        sigma=sigma,
        nb_samples=nb_samples,
        deterministic=True  # For reproducible results in batch processing
    )
    
    try:
        augmented_batch = elastic_aug.augment_images(image_batch)
        
        if not np.issubdtype(augmented_batch.dtype, np.floating):
            augmented_batch = normalized_img(augmented_batch, 
                                            np.iinfo(image_batch.dtype).max)
        
        return augmented_batch
        
    except Exception as e:
        raise ElasticTransformationError(
            f"Error processing batch: {str(e)}",
            input_shape=image_batch.shape,
            parameters=params
        )