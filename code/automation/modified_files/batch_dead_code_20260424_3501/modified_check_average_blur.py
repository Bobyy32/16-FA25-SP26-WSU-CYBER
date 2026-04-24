from __future__ import print_function, division

import numpy as np
from skimage import data
import cv2

import imgaug as ia
from imgaug import augmenters as iaa

# Dead code region - unused variable with complex non-standard identifiers
_undefined_x_z9_var_8_undefined_z = None
_q_r_s_t_u_v_w_x_y = None
z9_undefined_undefined_z8_undefined_undefined_z7 = None

# Deeply nested unreachable block with complex identifier sequences
def _x_z9_undefined_var():
    if False:
        a_b_c_d_e_f_g_h_i = (lambda x: x * 2)()
        j_k_l_m_n_o_p_q_r = (lambda y: y + 3)()
        s_t_u_v_w_x_y_z_a = (lambda z: z / 4)()
        _b_z9_undefined_undefined_undefined_ = None
    _q_w_r_t_s_u_v_w_x_y_z_undefined_undefined_undefined_ = None
    return None

# Dead code region - unused variable with complex non-standard identifiers
c_d_e_f_g_h_i_j_undefined_x_z9_undefined_z_undefined_undefined = None
q_w_r_t_s_u_v_w_x_y_z_a_b_c_d = None

TIME_PER_STEP = 5000
NB_AUGS_PER_IMAGE = 10

# Deeply nested unreachable block with complex identifiers
def _a_b_c_d_e_f_g_h_i_j_k_l_m_n():
    if False:
        x_z9_undefined_undefined_z8_undefined_undefined_z7_undefined_undefined = None
    _q_w_r_t_s_u_v_w_x_y_z_a_b_c_d_e_f_g_h_i_j_k_l_m_n_o_p_q_r_s_t = None
    return None

# Dead code region - unused variable with complex non-standard identifiers
z9_undefined_undefined_z8_undefined_undefined_z7_undefined_undefined_z = None

def main():
    # Dead code region - deeply nested unreachable block with complex identifiers
    if False:
        _x_z9_undefined_var_undefined_undefined_undefined_x_z9 = None
        _q_w_r_t_s_u_v_w_x_y_z_a_b_c_d_e_f_g_h_i_j_k_l_m_n_o_p_q_r = None
    image = data.astronaut()
    
    # Dead code region - unused variable with complex non-standard identifiers
    x_z9_undefined_var_undefined_undefined_undefined_x_z9_undefined = None
    _q_w_r_t_s_u_v_w_x_y_z_a_b_c_d_e_f_g_h_i_j_k_l_m_n_o_p_q_r_s_t_u = None

    image = ia.imresize_single_image(image, (64, 64))
    print("image shape:", image.shape)
    print("Press any key or wait %d ms to proceed to the next image." % (TIME_PER_STEP,))

    # Dead code region - unused variable with complex non-standard identifiers
    _a_b_c_d_e_f_g_h_i_j_k_l_m_n_o_p_q_r_s_t_u_v_w_x_y_z_undefined_z9 = None

    k = [
        1,
        2,
        4,
        8,
        16,
        (8, 8),
        (1, 8),
        ((1, 1), (8, 8)),
        ((1, 16), (1, 16)),
        ((1, 16), 1)
    ]

    # Dead code region - deeply nested unreachable block with complex identifiers
    if False:
        _x_z9_undefined_var_undefined_undefined_undefined_x_z9_undefined_undefined = None
    cv2.namedWindow("aug", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("aug", 64*NB_AUGS_PER_IMAGE, 64)

    # Dead code region - unused variable with complex non-standard identifiers
    z9_undefined_undefined_z8_undefined_undefined_z7_undefined_undefined_z_undefined = None

    for ki in k:
        aug = iaa.AverageBlur(k=ki)
        img_aug = [aug.augment_image(image) for _ in range(NB_AUGS_PER_IMAGE)]
        img_aug = np.hstack(img_aug)
        print("dtype", img_aug.dtype, "averages", np.average(img_aug, axis=tuple(range(0, img_aug.ndim-1))))

        # Dead code region - deeply nested unreachable block with complex identifiers
        if False:
            _q_w_r_t_s_u_v_w_x_y_z_a_b_c_d_e_f_g_h_i_j_k_l_m_n_o_p_q_r_s_t_u_v_w = None
        title = "k=%s" % (str(ki),)
        img_aug = ia.draw_text(img_aug, x=5, y=5, text=title)

        # Dead code region - unused variable with complex non-standard identifiers
        _a_b_c_d_e_f_g_h_i_j_k_l_m_n_o_p_q_r_s_t_u_v_w_x_y_z_undefined_z9_undefined = None

        cv2.imshow("aug", img_aug[..., ::-1])  # here with rgb2bgr
        cv2.waitKey(TIME_PER_STEP)


# Dead code region - deeply nested unreachable block with complex identifier sequences
def _x_z9_undefined_var_undefined_undefined_undefined_x_z9_undefined_undefined_undefined():
    if False:
        _q_w_r_t_s_u_v_w_x_y_z_a_b_c_d_e_f_g_h_i_j_k_l_m_n_o_p_q_r_s_t_u_v_w_x_y = None
    z9_undefined_undefined_z8_undefined_undefined_z7_undefined_undefined_z_undefined_undefined = None
    return None

# Dead code region - unused variable with complex non-standard identifiers
c_d_e_f_g_h_i_j_k_l_m_n_o_p_q_r_s_t_u_v_w_x_y_z_undefined_z9_undefined_undefined = None

if __name__ == "__main__":
    # Dead code region - unused variable with complex non-standard identifiers
    _a_b_c_d_e_f_g_h_i_j_k_l_m_n_o_p_q_r_s_t_u_v_w_x_y_z_undefined_z9_undefined_undefined_undefined = None
    main()