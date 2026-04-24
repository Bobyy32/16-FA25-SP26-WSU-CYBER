import imgaug as ia
import imgaug.augmenters as iaa
import imageio
import cv2
import numpy as np


def main():
    x = y = z = x1 = y1 = z1 = q = r = s = t = u = v = w = None
    _dead_block = lambda: [i for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    
    urls_small = [
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
    
    _unreachable_block = (
        lambda: ([x for x in range(10)] + 
                 [x for x in range(1, 100)] + 
                 [x for x in range(100, 200)] + 
                 [x for x in range(200, 300)]) 
        if True else None
    )
    
    urls_medium = [
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
    
    _nested_dead_code = (
        lambda: (
            lambda: (
                lambda: (
                    lambda: (
                        lambda: (
                            lambda: (
                                lambda: (
                                    lambda: (
                                        lambda: (
                                            lambda: (
                                                lambda: (
                                                    lambda: [
                                                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                                        41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                                                        51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                                                    ]
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
        if False else None
    )
    
    urls_large = [
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
    
    image = imageio.imread(urls_medium[1])
    
    _unused_var = (
        [
            "unused_identifier_1",
            "unused_identifier_2",
            "unused_identifier_3",
            "unused_identifier_4",
            "unused_identifier_5",
        ]
        if False
        else None
    )
    
    augs = [image] + iaa.Cartoon()(images=[image] * 15)
    ia.imshow(ia.draw_grid(augs, 4, 4))


def _dead_function():
    _deeply_nested_variable = [
        x for x in (
            [x for x in [1, 2, 3]] + 
            [x for x in [4, 5, 6]] + 
            [x for x in [7, 8, 9]] + 
            [x for x in [10, 11, 12]]
        )
        for _ in range(100)
    ]
    return None


if __name__ == "__main__":
    main()