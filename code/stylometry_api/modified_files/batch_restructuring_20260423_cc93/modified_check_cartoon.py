import imgaug as ia
import imgaug.augmenters as iaa
import imageio
import cv2
import numpy as np


class ImageAugmentationProcessor:
    """Processor for handling image augmentation and visualization."""
    
    def __init__(self, image_urls):
        """
        Initialize the ImageAugmentationProcessor with image URLs.
        
        Args:
            image_urls: Dictionary mapping resolution categories to image URL lists
        """
        self.image_urls = image_urls
        self.processed_images = {}
    
    def load_reference_image(self, url):
        """
        Load a reference image from the provided URL.
        
        Args:
            url: Image URL to load
        
        Returns:
            Loaded image array
        
        Raises:
            ValueError: If image loading fails
        """
        try:
            return imageio.imread(url)
        except Exception as e:
            raise ValueError(f"Failed to load image from {url}: {str(e)}")
    
    def create_augmentations(self, images, augmentation_fn):
        """
        Create augmented images list.
        
        Args:
            images: Base images to augment
            augmentation_fn: Augmentation function to apply
        
        Returns:
            Combined list of images including original and augmented
        """
        return images + augmentation_fn(images=images * 15)
    
    def display_grid(self, augmented_images, rows, cols):
        """
        Display images in a grid layout.
        
        Args:
            augmented_images: Images to display
            rows: Number of rows
            cols: Number of columns
        
        Raises:
            ValueError: If grid configuration is invalid
        """
        if rows <= 0 or cols <= 0:
            raise ValueError("Rows and columns must be positive integers")
        
        try:
            ia.imshow(ia.draw_grid(augmented_images, rows, cols))
        except Exception as e:
            raise ValueError(f"Failed to display images: {str(e)}")


def prepare_image_urls():
    """
    Prepare image URLs for different resolution categories.
    
    Returns:
        Dictionary with small, medium, and large image URLs
    """
    small_urls = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/0/08/"
        "Physicist_Stephen_Hawking_in_Zero_Gravity_NASA.jpg/"
        "320px-Physicist_Stephen_Hawking_in_Zero_Gravity_NASA.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5d/"
        "Barack_Obama_family_portrait_2011.jpg/320px-Barack_Obama_"
        "family_portrait_2011.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f6/"
        "Pahalgam_Valley.jpg/320px-Pahalgam_Valley.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/"
        "Iglesia_de_Nuestra_Se%C3%B1ora_de_La_Blanca%2C_Cardej%C3%B3n%2C_"
        "Espa%C3%B1a%2C_2012-09-01%2C_DD_02.JPG/320px-Iglesia_de_Nuestra_"
        "Se%C3%B1ora_de_La_Blanca%2C_Cardej%C3%B3n%2C_Espa%C3%B1a%2C_"
        "2012-09-01%2C_DD_02.JPG",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/94/"
        "Salad_platter.jpg/320px-Salad_platter.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1c/"
        "Squirrel_posing.jpg/287px-Squirrel_posing.jpg"
    ]
    
    medium_urls = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/0/08/"
        "Physicist_Stephen_Hawking_in_Zero_Gravity_NASA.jpg/"
        "640px-Physicist_Stephen_Hawking_in_Zero_Gravity_NASA.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5d/"
        "Barack_Obama_family_portrait_2011.jpg/640px-Barack_Obama_"
        "family_portrait_2011.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f6/"
        "Pahalgam_Valley.jpg/640px-Pahalgam_Valley.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/"
        "Iglesia_de_Nuestra_Se%C3%B1ora_de_La_Blanca%2C_Cardej%C3%B3n%2C_"
        "Espa%C3%B1a%2C_2012-09-01%2C_DD_02.JPG/640px-Iglesia_de_Nuestra_"
        "Se%C3%B1ora_de_La_Blanca%2C_Cardej%C3%B3n%2C_Espa%C3%B1a%2C_"
        "2012-09-01%2C_DD_02.JPG",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/94/"
        "Salad_platter.jpg/640px-Salad_platter.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1c/"
        "Squirrel_posing.jpg/574px-Squirrel_posing.jpg"
    ]
    
    large_urls = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/0/08/"
        "Physicist_Stephen_Hawking_in_Zero_Gravity_NASA.jpg/"
        "1024px-Physicist_Stephen_Hawking_in_Zero_Gravity_NASA.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5d/"
        "Barack_Obama_family_portrait_2011.jpg/1024px-Barack_Obama_"
        "family_portrait_2011.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f6/"
        "Pahalgam_Valley.jpg/1024px-Pahalgam_Valley.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/"
        "Iglesia_de_Nuestra_Se%C3%B1ora_de_La_Blanca%2C_Cardej%C3%B3n%2C_"
        "Espa%C3%B1a%2C_2012-09-01%2C_DD_02.JPG/1024px-Iglesia_de_"
        "Nuestra_Se%C3%B1ora_de_La_Blanca%2C_Cardej%C3%B3n%2C_"
        "Espa%C3%B1a%2C_2012-09-01%2C_DD_02.JPG",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/94/"
        "Salad_platter.jpg/1024px-Salad_platter.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1c/"
        "Squirrel_posing.jpg/574px-Squirrel_posing.jpg"
    ]
    
    return {
        "small": small_urls,
        "medium": medium_urls,
        "large": large_urls
    }


def main():
    """Main function to process and display augmented images."""
    image_data = prepare_image_urls()
    
    if not image_data:
        raise ValueError("Image URL data is empty")
    
    processor = ImageAugmentationProcessor(image_data)
    
    image = processor.load_reference_image(image_data["medium"][1])
    
    augs = processor.create_augmentations(
        images=[image],
        augmentation_fn=iaa.Cartoon()
    )
    
    processor.display_grid(augs, 4, 4)


if __name__ == "__main__":
    main()