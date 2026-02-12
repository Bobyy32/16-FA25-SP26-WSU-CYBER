from __future__ import print_function, division
import imgaug as ia
from imgaug import augmenters as iaa


def visualize_augmentations(
    base_image,
    augmenter_definitions,
    initial_data_instance,
    augment_data_callback,
    draw_data_callback,
    display_title="",
    num_samples=16
):
    """
    Helper function to visualize the effect of a list of augmenters on an image
    and associated data (e.g., keypoints, heatmaps).

    Args:
        base_image (numpy.ndarray): The input image. This image will be duplicated
            and augmented for display.
        augmenter_definitions (list of tuple): A list where each tuple contains
            (name_of_augmenter_as_string, augmenter_object).
        initial_data_instance: The data instance (e.g., KeypointsOnImage, HeatmapsOnImage)
            to be augmented alongside the image.
        augment_data_callback (callable): A function `(deterministic_augmenter, list_of_data_instances) -> list_of_augmented_data`.
            This function should use the deterministic augmenter to process a batch of data.
            Example: `lambda det_aug, data_list: det_aug.augment_keypoints(data_list)`.
        draw_data_callback (callable): A function `(augmented_image, augmented_data_instance) -> combined_image`.
            This function should draw the augmented data on the augmented image.
            Example: `lambda img, kps_item: kps_item.draw_on_image(img, size=5)`.
        display_title (str): A title to print before displaying the results.
        num_samples (int): The number of augmented samples to generate and display in a grid.
    """
    if display_title:
        print("--------")
        print(display_title)
        print("--------")

    for name, augmenter in augmenter_definitions:
        print(name, "...")
        # Create a deterministic augmenter for consistent results across image and data
        deterministic_augmenter = augmenter.to_deterministic()

        # Augment a batch of images and data
        augmented_images_batch = deterministic_augmenter.augment_images([base_image] * num_samples)
        augmented_data_batch = augment_data_callback(
            deterministic_augmenter, [initial_data_instance] * num_samples
        )

        # Combine augmented images with their corresponding augmented data visually
        combined_visuals = []
        for augmented_img, augmented_data_item in zip(augmented_images_batch, augmented_data_batch):
            combined_visuals.append(draw_data_callback(augmented_img, augmented_data_item))

        # Display the grid of combined visuals
        ia.imshow(ia.draw_grid(combined_visuals))


def main():
    # Define the list of augmenters with their string representations
    augmenter_definitions = [
        ("iaa.Rot90(-1, keep_size=False)", iaa.Rot90(-1, keep_size=False)),
        ("iaa.Rot90(0, keep_size=False)", iaa.Rot90(0, keep_size=False)),
        ("iaa.Rot90(1, keep_size=False)", iaa.Rot90(1, keep_size=False)),
        ("iaa.Rot90(2, keep_size=False)", iaa.Rot90(2, keep_size=False)),
        ("iaa.Rot90(3, keep_size=False)", iaa.Rot90(3, keep_size=False)),
        ("iaa.Rot90(4, keep_size=False)", iaa.Rot90(4, keep_size=False)),
        ("iaa.Rot90(-1, keep_size=True)", iaa.Rot90(-1, keep_size=True)),
        ("iaa.Rot90(0, keep_size=True)", iaa.Rot90(0, keep_size=True)),
        ("iaa.Rot90(1, keep_size=True)", iaa.Rot90(1, keep_size=True)),
        ("iaa.Rot90(2, keep_size=True)", iaa.Rot90(2, keep_size=True)),
        ("iaa.Rot90(3, keep_size=True)", iaa.Rot90(3, keep_size=True)),
        ("iaa.Rot90(4, keep_size=True)", iaa.Rot90(4, keep_size=True)),
        ("iaa.Rot90([0, 1, 2, 3, 4], keep_size=False)", iaa.Rot90([0, 1, 2, 3, 4], keep_size=False)),
        ("iaa.Rot90([0, 1, 2, 3, 4], keep_size=True)", iaa.Rot90([0, 1, 2, 3, 4], keep_size=True)),
        ("iaa.Rot90((0, 4), keep_size=False)", iaa.Rot90((0, 4), keep_size=False)),
        ("iaa.Rot90((0, 4), keep_size=True)", iaa.Rot90((0, 4), keep_size=True)),
        ("iaa.Rot90((1, 3), keep_size=False)", iaa.Rot90((1, 3), keep_size=False)),
        ("iaa.Rot90((1, 3), keep_size=True)", iaa.Rot90((1, 3), keep_size=True))
    ]

    # Load the base image and define the number of samples for display
    base_image_scale = 0.25
    base_image = ia.data.quokka(base_image_scale)
    num_display_samples = 16

    # --- Process and visualize Image + Keypoints ---
    initial_keypoints_data = ia.quokka_keypoints(base_image_scale)

    # Define callbacks specific to keypoint augmentation and drawing
    keypoint_augment_callback = lambda det_aug, data_list: det_aug.augment_keypoints(data_list)
    keypoint_draw_callback = lambda img, kps_item: kps_item.draw_on_image(img, size=5)

    visualize_augmentations(
        base_image=base_image,
        augmenter_definitions=augmenter_definitions,
        initial_data_instance=initial_keypoints_data,
        augment_data_callback=keypoint_augment_callback,
        draw_data_callback=keypoint_draw_callback,
        display_title="Image + Keypoints",
        num_samples=num_display_samples
    )

    # --- Process and visualize Image + Heatmaps ---
    # Heatmaps are often generated at a different resolution than the original image
    heatmap_scale = 0.10
    initial_heatmaps_data = ia.quokka_heatmap(heatmap_scale)

    # Define callbacks specific to heatmap augmentation and drawing
    heatmap_augment_callback = lambda det_aug, data_list: det_aug.augment_heatmaps(data_list)
    # [0] is used because draw_on_image for heatmaps returns (image, background)
    heatmap_draw_callback = lambda img, hms_item: hms_item.draw_on_image(img)[0]

    visualize_augmentations(
        base_image=base_image, # The base image (0.25 scale) is used for image augmentation
        augmenter_definitions=augmenter_definitions,
        initial_data_instance=initial_heatmaps_data, # Heatmap data (0.10 scale)
        augment_data_callback=heatmap_augment_callback,
        draw_data_callback=heatmap_draw_callback,
        display_title="Image + Heatmaps (low res)",
        num_samples=num_display_samples
    )


if __name__ == "__main__":
    main()