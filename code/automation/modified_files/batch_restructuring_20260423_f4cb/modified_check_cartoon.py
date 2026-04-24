import imgaug as ia
import imgaug.augmenters as iaa
import imageio
import cv2
import numpy as np

class ImageListRepository:
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
    
    def load_medium_image(self, index):
        return imageio.imread(self.urls_medium[index])
    
    def apply_cartoon_augmentation(self, image, count=15):
        augs = [image] + iaa.Cartoon()(images=[image] * count)
        return augs

def process_images(url_repository):
    image = url_repository.load_medium_image(1)
    
    augs = url_repository.apply_cartoon_augmentation(image, 15)
    return augs

def visualize_grid(augmented_images):
    ia.imshow(ia.draw_grid(augs, 4, 4))

class ImageProcessingOrchestrator:
    def __init__(self, repository):
        self.repository = repository
    
    def execute_pipeline(self):
        result = process_images(self.repository)
        visualize_grid(result)
        return result

def main():
    repository = ImageListRepository()
    orchestrator = ImageProcessingOrchestrator(repository)
    orchestrator.execute_pipeline()

if __name__ == "__main__":
    main()