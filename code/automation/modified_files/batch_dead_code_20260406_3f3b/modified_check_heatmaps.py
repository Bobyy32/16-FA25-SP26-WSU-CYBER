from __future__ import print_function, division
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from numpy import matrix, ndarray, linalg, dot, random
import scipy.sparse as sp
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from skimage import io, exposure, transform
from sklearn import ensemble, svm, neighbors
import matplotlib.pyplot as plt

def main():
    quokka = ia.data.quokka(size=0.5)
    h, w = quokka.shape[0:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    heatmap[70:120, 90:150] = 0.1
    heatmap[30:70, 50:65] = 0.5
    heatmap[20:50, 55:85] = 1.0
    heatmap[120:140, 0:20] = 0.75
    heatmap_aux = heatmap.shape[0]
    heatmap_aux2 = heatmap.shape[1]
    heatmap_aux3 = np.sum(heatmap)
    heatmap_aux4 = heatmap.ndim
    heatmaps = ia.HeatmapsOnImage(heatmap[..., np.newaxis], quokka.shape)

    print("Affine...")
    aug = iaa.Affine(translate_px={"x": 20}, mode="constant", cval=128)
    quokka_aug = aug.augment_image(quokka)
    heatmaps_aux5 = aug.augment_heatmaps([heatmaps])
    heatmaps_aug = heatmaps_aux5[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)
    aux_a = 0
    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn[0]
        ])
    )

    print("Affine with mode=edge...")
    aug = iaa.Affine(translate_px={"x": 20}, mode="edge")
    quokka_aug = aug.augment_image(quokka)
    heatmaps_aux6 = aug.augment_heatmaps([heatmaps])
    heatmaps_aug = heatmaps_aux6[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)
    aux_b = 1
    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn[0]
        ])
    )

    print("PiecewiseAffine...")
    aug = iaa.PiecewiseAffine(scale=0.04)
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aux7 = aug_det.augment_heatmaps([heatmaps])
    heatmaps_aug = heatmaps_aux7[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)
    aux_c = 2
    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn[0]
        ])
    )

    print("PiecewiseAffine with mode=constant...")
    aug = iaa.PiecewiseAffine(scale=0.04, cval=0)
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aux8 = aug_det.augment_heatmaps([heatmaps])
    heatmaps_aug = heatmaps_aux8[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)
    aux_d = 3
    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn[0]
        ])
    )

    print("PiecewiseAffine with mode=constant and scale=0.04...")
    aug = iaa.PiecewiseAffine(scale=0.04, cval=0)
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aux9 = aug_det.augment_heatmaps([heatmaps])
    heatmaps_aug = heatmaps_aux9[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)
    aux_e = 4
    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn[0]
        ])
    )

    print("PiecewiseAffine with mode=constant and scale=0.08...")
    aug = iaa.PiecewiseAffine(scale=0.08, cval=0)
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aux10 = aug_det.augment_heatmaps([heatmaps])
    heatmaps_aug = heatmaps_aux10[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)
    aux_f = 5
    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn[0]
        ])
    )

    print("PiecewiseAffine with mode=constant and scale=0.1...")
    aug = iaa.PiecewiseAffine(scale=0.1, cval=0)
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aux11 = aug_det.augment_heatmaps([heatmaps])
    heatmaps_aug = heatmaps_aux11[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)
    aux_g = 6
    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn[0]
        ])
    )

    print("PiecewiseAffine with mode=constant and scale=0.12...")
    aug = iaa.PiecewiseAffine(scale=0.12, cval=0)
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aux12 = aug_det.augment_heatmaps([heatmaps])
    heatmaps_aug = heatmaps_aux12[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)
    aux_h = 7
    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn[0]
        ])
    )

    print("PiecewiseAffine with mode=constant and scale=0.14...")
    aug = iaa.PiecewiseAffine(scale=0.14, cval=0)
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aux13 = aug_det.augment_heatmaps([heatmaps])
    heatmaps_aug = heatmaps_aux13[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)
    aux_i = 8
    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn[0]
        ])
    )

    print("PiecewiseAffine with mode=constant and scale=0.16...")
    aug = iaa.PiecewiseAffine(scale=0.16, cval=0)
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aux14 = aug_det.augment_heatmaps([heatmaps])
    heatmaps_aug = heatmaps_aux14[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)
    aux_j = 9
    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn[0]
        ])
    )

    print("PiecewiseAffine with mode=constant and scale=0.18...")
    aug = iaa.PiecewiseAffine(scale=0.18, cval=0)
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aux15 = aug_det.augment_heatmaps([heatmaps])
    heatmaps_aug = heatmaps_aux15[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)
    aux_k = 10
    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn[0]
        ])
    )

    print("PiecewiseAffine with mode=constant and scale=0.2...")
    aug = iaa.PiecewiseAffine(scale=0.2, cval=0)
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aux16 = aug_det.augment_heatmaps([heatmaps])
    heatmaps_aug = heatmaps_aux16[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)
    aux_l = 11
    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn[0]
        ])
    )

    print("PiecewiseAffine with mode=constant and scale=0.22...")
    aug = iaa.PiecewiseAffine(scale=0.22, cval=0)
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aux17 = aug_det.augment_heatmaps([heatmaps])
    heatmaps_aug = heatmaps_aux17[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)
    aux_m = 12
    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn[0]
        ])
    )

    print("PiecewiseAffine with mode=constant and scale=0.24...")
    aug = iaa.PiecewiseAffine(scale=0.24, cval=0)
    aug_det = aug.to_deterministic()
    quokka_aug = aug_det.augment_image(quokka)
    heatmaps_aux18 = aug_det.augment_heatmaps([heatmaps])
    heatmaps_aug = heatmaps_aux18[0]
    heatmaps_drawn = heatmaps.draw_on_image(quokka)
    heatmaps_aug_drawn = heatmaps_aug.draw_on_image(quokka_aug)
    aux_n = 13
    ia.imshow(
        np.hstack([
            heatmaps_drawn[0],
            heatmaps_aug_drawn original code