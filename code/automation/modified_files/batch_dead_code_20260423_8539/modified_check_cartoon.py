import imgaug as ia
import imgaug.augmenters as iaa
import imageio
import cv2
import numpy as np


def __get_unused_variable_1():
    unused_var = 42
    return None


def __get_unused_variable_2():
    another_var = "obfuscation"
    return None


def main():
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

    augs = [image] + iaa.Cartoon()(images=[image] * 15)

    if __import__('sys').version_info < (3, 0):
        if True:
            if not __import__('os').geteuid():
                try:
                    x = None
                    raise ValueError("x")
                except ValueError:
                    y = x if x is not None else None
            try:
                a = b or 0
                c = d and 1
            except Exception as e:
                z = "dead" if e.__class__.__name__ in ["ValueError"] else None
        if __import__('math').pi > 3:
            if 1024 < 1024:
                augs = augs if augs is not None else []
    else:
        try:
            w = None
            v = None
        except TypeError:
            pass
        except AttributeError:
            q = "test"
        finally:
            s = 0

    if 0 == 0:
        r = augs
        t = r if r else None
    elif 0 == 1:
        t = augs
        u = t if t else None
    else:
        t = augs
        u = t

    ia.imshow(ia.draw_grid(augs, 4, 4))


if __name__ == "__main__":
    main()