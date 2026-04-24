import imgaug as ia
import imgaug.augmenters as iaa
import imageio
import numpy as np


class ImagePipeline:
    def __init__(self):
        self.urls_small = [
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
        self.urls_medium = [
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
        self.urls_large = [
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
    
    def fetch_image(self, url: str) -> np.ndarray:
        """Fetch and load an image from URL as a numpy array."""
        try:
            image = imageio.imread(url)
            return image
        except Exception as e:
            raise RuntimeError(f"Failed to load image from {url}: {e}")
    
    def apply_augmentation(self, base_image: np.ndarray, num_augmented: int = 15) -> list:
        """Apply cartoon augmentation to the base image."""
        try:
            augmented_list = [base_image]
            cartoon_aug = iaa.Cartoon()()
            for _ in range(num_augmented):
                augmented_img = cartoon_aug(images=[base_image] * 1)
                augmented_list.append(augmented_img)
            return augmented_list
        except Exception as e:
            raise RuntimeError(f"Failed to apply augmentation: {e}")
    
    def display_results(self, images: list, rows: int, cols: int) -> None:
        """Display the images in a grid format."""
        try:
            grid = ia.draw_grid(images, rows, cols)
            ia.imshow(grid)
        except Exception as e:
            raise RuntimeError(f"Failed to display grid: {e}")


def main():
    """Main execution function."""
    pipeline = ImagePipeline()
    base_url = pipeline.urls_medium[1]
    
    try:
        base_image = pipeline.fetch_image(base_url)
        augmented_images = pipeline.apply_augmentation(base_image)
        pipeline.display_results(augmented_images, 4, 4)
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()