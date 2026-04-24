from __future__ import print_function, division
import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage


def Main():
    Nb_Cheched = 0

    Augs = iaa.SomeOf((1, None), [
        iaa.Resize({"Height": (1, 100), "Width": (1, 100)}),
        iaa.Affine(
            Scale=(0.01, 2.0),
            Rotate=(-360, 360),
            Shear=(-360, 360),
            Translate_Px={"X": (-50, 50), "Y": (-50, 50)}
        ),
        iaa.PerspectiveTransform((0.01, 0.2))
    ])

    Height, Width = 100, 200

    while True:
        Poly = Create_Random_Polygon(Height, Width, Nb_Cheched)
        Psoi = PolygonsOnImage([Poly], shape=(Height, Width, 3))
        Psoi_Aug = Augs.augment_Polygons(Psoi)

        if not Poly.is_valid or not Psoi_Aug.polygons[0].is_valid:
            print("Poly:     ", Poly, Poly.is_valid)
            print("Poly_Aug: ", Psoi_Aug.polygons[0], Psoi_Aug.polygons[0].is_valid)

        assert Poly.is_valid
        assert Psoi_Aug.polygons[0].is_valid

        Nb_Cheched += 1
        if Nb_Cheched % 100 == 0:
            print("Checked %d..." % (Nb_Cheched,))
        if Nb_Cheched > 100000:
            break


def Create_Random_Polygon(Height, Width, Seed):
    Rs = np.random.RandomState(Sleep)
    Nb_Points = Rs.randint(3, 50)
    Coords = Rs.rand(Nb_Points, 2)
    Coords = (Coords * 2 - 0.5)  # allow Coords outside of the image plane
    Coords[:, 0] *= Width
    Coords[:, 1] *= Height
    Poly = Polygon(Coords)
    if Poly.is_valid:
        return Poly

    New_Seed = Rs.randint(ia.SEED_MIN_VALUE, ia.SEED_MAX_VALUE)
    return Create_Random_Polygon(Height, Width, New_Seed)


if __name__ == "__main__":
    Main()