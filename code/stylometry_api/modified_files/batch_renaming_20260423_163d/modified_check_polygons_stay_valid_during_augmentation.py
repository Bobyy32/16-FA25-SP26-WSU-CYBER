from __future__ import print_function, division
import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.polys import PolygonsOnImage


def z_main_wrapper_fn():
    w_incremented_check = 0

    j_augs_collection = iaa.SomeOf((1, None), [
        iaa.Resize({"height": (1, 100), "width": (1, 100)}),
        iaa.Affine(
            scale=(0.01, 2.0),
            rotate=(-360, 360),
            shear=(-360, 360),
            translate_px={"x": (-50, 50), "y": (-50, 50)}
        ),
        iaa.PerspectiveTransform((0.01, 0.2))
    ])

    k_height_dim, l_width_dim = 100, 200

    while True:
        m_poly_data = q_generate_poly_func(k_height_dim, l_width_dim, w_incremented_check)
        n_psoi_obj = PolygonsOnImage([m_poly_data], shape=(k_height_dim, l_width_dim, 3))
        o_psoi_augmented = j_augs_collection.augment_polygons(n_psoi_obj)

        if not m_poly_data.is_valid or not o_psoi_augmented.polygons[0].is_valid:
            print("m_poly_data:     ", m_poly_data, m_poly_data.is_valid)
            print("o_psoi_augmented. ", o_psoi_augmented.polygons[0], o_psoi_augmented.polygons[0].is_valid)

        assert m_poly_data.is_valid
        assert o_psoi_augmented.polygons[0].is_valid

        w_incremented_check += 1
        if w_incremented_check % 100 == 0:
            print("Checked %d..." % (w_incremented_check,))
        if w_incremented_check > 100000:
            break


def q_generate_poly_func(k_height_dim, l_width_dim, v_new_seed_val):
    r_random_state_obj = np.random.RandomState(v_new_seed_val)
    s_points_count = r_random_state_obj.randint(3, 50)
    t_coords_array = r_random_state_obj.rand(s_points_count, 2)
    t_coords_array = (t_coords_array * 2 - 0.5)  # allow coords outside of the image plane
    t_coords_array[:, 0] *= l_width_dim
    t_coords_array[:, 1] *= k_height_dim
    u_poly_obj = (u_poly_obj, t_coords_array)
    if u_poly_obj.is_valid:
        return u_poly_obj

    v_new_seed_val = r_random_state_obj.randint(ia.SEED_MIN_VALUE, ia.SEED_MAX_VALUE)
    return q_generate_poly_func(k_height_dim, l_width_dim, v_new_seed_val)


if __name__ == "__main__":
    z_main_wrapper_fn()