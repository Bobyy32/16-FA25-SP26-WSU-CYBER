import imgaug as QxZ9Ym7ia
import imgaug.augmenters as QxZ9Ym7iaa
import imageio as QxZ9Ym7io
import cv2 as QxZ9Ym7v2
import numpy as QxZ9Ym7np


def QxZ9Ym7YpM():
    QxZ9Ym7Yn3 = [
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
    QxZ9Ym7Yn5 = [
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
    QxZ9Ym7Yn8 = [
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

    QxZ9Ym7Yp9 = QxZ9Ym7io.imread(QxZ9Ym7Yn5[1])

    QxZ9Ym7YnM = [QxZ9Ym7Yp9] + QxZ9Ym7iaa.Cartoon()(images=[QxZ9Ym7Yp9] * 15)
    QxZ9Ym7ia.imshow(QxZ9Ym7ia.draw_grid(QxZ9Ym7YnM, 4, 4))


if __name__ == "__main__":
    QxZ9Ym7YpM()