from __future__ import print_function, division
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
from skimage import data
import cv2

# --- Global Constants and Configuration ---
TIME_PER_STEP = 5000
NB_AUGS_PER_IMAGE = 10
IMAGE_TARGET_SIZE = (128, 128)
DISPLAY_WINDOW_NAME = "Augmentation Demo"
DISPLAY_WIDTH = IMAGE_TARGET_SIZE[0] * NB_AUGS_PER_IMAGE
DISPLAY_HEIGHT = IMAGE_TARGET_SIZE[1]

# --- Helper Functions ---

def load_and_prepare_source_content(target_size_tuple):
    """
    Loads a source image and resizes it to the specified target dimensions.
    
    Args:
        target_size_tuple (tuple): A tuple (width, height) for resizing.
        
    Returns:
        np.ndarray: The prepared source image.
    """
    original_content = data.astronaut()
    prepared_content = ia.imresize_single_image(original_content, target_size_tuple)
    print("Source content shape:", prepared_content.shape)
    return prepared_content

def setup_visual_display(window_title, window_width, window_height, initial_message):
    """
    Configures the OpenCV display window and prints an initial message.
    
    Args:
        window_title (str): The title for the display window.
        window_width (int): The width of the display window.
        window_height (int): The height of the display window.
        initial_message (str): A message to print to the console.
    """
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, window_width, window_height)
    print(initial_message)

def create_bilateral_blur_augmenter(d_param, sigma_color_param, sigma_space_param):
    """
    Instantiates an iaa.BilateralBlur augmenter with specific parameters.
    
    Args:
        d_param: Diameter of the pixel neighborhood.
        sigma_color_param: Filter sigma in the color space.
        sigma_space_param: Filter sigma in the coordinate space.
        
    Returns:
        iaa.BilateralBlur: Configured augmenter instance.
    """
    return iaa.BilateralBlur(d=d_param, sigma_color=sigma_color_param, sigma_space=sigma_space_param)

def generate_augmented_batch(source_content_array, augmenter_instance, num_samples_per_batch):
    """
    Applies a given augmenter multiple times to a source content array
    and stacks the results horizontally.
    
    Args:
        source_content_array (np.ndarray): The base image for augmentation.
        augmenter_instance (iaa.Augmenter): The augmenter to apply.
        num_samples_per_batch (int): How many augmented samples to generate.
        
    Returns:
        np.ndarray: Horizontally stacked array of augmented samples.
    """
    augmented_samples_list = [augmenter_instance.augment_image(source_content_array) for _ in range(num_samples_per_batch)]
    combined_samples = np.hstack(augmented_samples_list)
    return combined_samples

def generate_config_info_title(d_param, sigma_color_param, sigma_space_param):
    """
    Creates a formatted title string representing the current augmentation configuration.
    
    Args:
        d_param: Diameter parameter.
        sigma_color_param: Sigma color parameter.
        sigma_space_param: Sigma space parameter.
        
    Returns:
        str: Formatted title string.
    """
    return "d=%s, sc=%s, ss=%s" % (str(d_param), str(sigma_color_param), str(sigma_space_param))

def show_content_and_wait(window_title, content_for_display, delay_milliseconds):
    """
    Displays the given content in the specified OpenCV window and waits.
    
    Args:
        window_title (str): The title of the display window.
        content_for_display (np.ndarray): The image content to show.
        delay_milliseconds (int): Time to wait in milliseconds (0 for infinite).
    """
    # OpenCV expects BGR, while imgaug typically works with RGB.
    cv2.imshow(window_title, content_for_display[..., ::-1]) 
    cv2.waitKey(delay_milliseconds)

# --- Main Program Logic ---

def main():
    # 1. Prepare the base image content
    source_image_data = load_and_prepare_source_content(IMAGE_TARGET_SIZE)

    # 2. Define the list of augmentation configurations to test
    augmentation_configurations = [
        (1, 75, 75), (3, 75, 75), (5, 75, 75), (10, 75, 75), (10, 25, 25),
        (10, 250, 150), (15, 75, 75), (15, 150, 150), (15, 250, 150),
        (20, 75, 75), (40, 150, 150), ((1, 5), 75, 75), (5, (10, 250), 75),
        (5, 75, (10, 250)), (5, (10, 250), (10, 250)), (10, (10, 250), (10, 250)),
    ]

    # 3. Setup the OpenCV display window
    initial_console_prompt = "Press any key or wait %d ms to proceed to the next image." % (TIME_PER_STEP,)
    setup_visual_display(DISPLAY_WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT, initial_console_prompt)

    # 4. Iterate through each configuration, apply augmentations, and display results
    for current_config_params in augmentation_configurations:
        d_value, sigma_color_value, sigma_space_value = current_config_params
        
        # Create an augmenter instance for the current configuration
        active_augmenter = create_bilateral_blur_augmenter(d_value, sigma_color_value, sigma_space_value)

        # Generate a batch of augmented samples
        augmented_content_batch = generate_augmented_batch(source_image_data, active_augmenter, NB_AUGS_PER_IMAGE)
        
        # Print informational statistics about the augmented content
        print("dtype", augmented_content_batch.dtype, "averages", np.average(augmented_content_batch, axis=tuple(range(0, augmented_content_batch.ndim-1))))

        # Generate and draw a title on the content for display
        display_text_title = generate_config_info_title(d_value, sigma_color_value, sigma_space_value)
        final_content_for_presentation = ia.draw_text(augmented_content_batch, x=5, y=5, text=display_text_title)

        # Display the final content and wait for user interaction or timeout
        show_content_and_wait(DISPLAY_WINDOW_NAME, final_content_for_presentation, TIME_PER_STEP)

if __name__ == "__main__":
    main()