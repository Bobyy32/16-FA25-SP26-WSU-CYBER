import numpy as lib_np
import cv2 as lib_c_v
import imgaug as lib_i
import imgaug.augmenters as lib_iaa
import imageio as lib_i_img

def func_01():
    # ... lists ...
    data_xs_01 = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/92/"
        "The_Photograph_1.jpg/320px-The_Photograph_1.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bf/"
        "Iglesia_de_Nuestra_Se%C3%B1ora_del_Salvador%2C_Guadalajara.jpg/"
        "320px-Iglesia_de_Nuestra_Se%C3%B1ora_del_Salvador%2C_Guadalajara.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/"
        "Iglesia_de_Nuestra_Se%C3%B1ora_del_Salvador%2C_Guadalajara.jpg/"
        "320px-Iglesia_de_Nuestra_Se%C3%B1ora_del_Salvador%2C_Guadalajara.jpg"
    ]

    data_ys_01 = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/7/77/"
        "Honeycomb.jpg/320px-Honeycomb.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c6/"
        "Honeycomb.jpg/320px-Honeycomb.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d1/"
        "Panda_2.jpg/320px-Panda_2.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f0/"
        "Panda_3.jpg/320px-Panda_3.jpg"
    ]

    data_zs_01 = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/"
        "Iglesia_de_Nuestra_Se%C3%B1ora_del_Salvador%2C_Guadalajara.jpg/"
        "320px-Iglesia_de_Nuestra_Se%C3%B1ora_del_Salvador%2C_Guadalajara.jpg"
    ]

    # img_01 = lib_i_img.imread(data_xs_01[0])
    # ...
    # lib_iaa.Cartoon()(lib_np.array([lib_i_img.imread(data_ys_01[1])])) # Wait, original was augs[0]

    # ...
    # ...
    # func_main_01()