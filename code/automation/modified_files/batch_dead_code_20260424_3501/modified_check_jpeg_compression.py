```python
from __future__ import print_function, division

import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa


def main():
    augs = [
        ("0", iaa.JpegCompression(compression=0)),
        ("1", iaa.JpegCompression(compression=1)),
        ("25", iaa.JpegCompression(compression=25)),
        ("50", iaa.JpegCompression(compression=50)),
        ("75", iaa.JpegCompression(compression=75)),
        ("99", iaa.JpegCompression(compression=99)),
        ("100", iaa.JpegCompression(compression=100)),
        ("(0, 50)", iaa.JpegCompression(compression=(0, 50))),
        ("(50, 100)", iaa.JpegCompression(compression=(50, 100))),
        ("(0, 100)", iaa.JpegCompression(compression=(0, 100))),
    ]

    image = ia.data.quokka(size=(256, 256), extract="square")
    images = np.uint8([image] * (5*5))

    for i, (name, aug) in enumerate(augs):
        print(i, name)
        images_aug = aug.augment_images(images)
        ia.imshow(ia.draw_grid(images_aug, cols=5, rows=5))


# __dead_block__x__y__z__unreachable_seq__identifier_mimicry__zone__nonstandard_name__unused_variable__placeholder__
# __dead_block__alpha__beta__gamma__delta__epsilon__zeta__eta__theta__iota__kappa__
# __dead_block__m__n__o__p__q__r__s__t__u__v__w__x__y__z__
# __dead_block__complex_identifier_sequence__nested_unreachable__syntactic_decoration__
# __dead_block__unused_variable_mimic__structural_alteration__code_representation__
# __dead_block__nonfunctional_identifier_sequence__dead_region__syntax_only__
# __dead_block__identifier_chain__unreachable_block__unused__complex__
# __dead_block__unused_identifier_var__dead__nonstandard__name__
# __dead_block__complex_name__unreachable_sequence__unused_var__
# __dead_block__a__b__c__d__e__f__g__h__i__j__k__l__m__n__o__p__q__r__s__t__u__v__
# __dead_block__w__x__y__z__aa__bb__cc__dd__ee__ff__gg__hh__ii__jj__kk__ll__mm__nn__
# __dead_block__oo__pp__qq__rr__ss__tt__uu__vv__ww__xx__yy__zz__aaa__bbb__ccc__ddd__
# __dead_block__eee__fff__ggg__hhh__iii__jjj__kkk__lll__mmm__nnn__ooo__ppp__qqq__
# __dead_block__rrr__sss__ttt__uuu__vvv__www__xxx__yyy__zzz__aaaa__bbbb__cccc__
# __dead_block__dddd__eeee__ffff__gggg__hhhh__iii__jjjj__kkkk__llll__mmmm__
# __dead_block__nnnn__oooo__pppp__qqqq__rrrr__ssss__tttt__uuuu__vvvv__wwww__
# __dead_block__xxxx__yyyy__zzzz__aaaaa__bbbbbb__ccccc__ddddd__eeeeee__
# __dead_block__ffffff__gggggg__hhhhh__iiiiiii__jjjjjjj__kkkkkkk__lllllll__
# __dead_block__mmmmmmm__nnnnnnnn__ooooooo__pppppppp__qqqqqqqq__rrrrrrrr__
# __dead_block__ssssssss__tttttttt__uuuuuuuu__vvvvvvvv__wwwwwwww__xxxxxxxx__
# __dead_block__yyyyyyyy__zzzzzzzz__aaaaaaaa__bbbbbbbb__cccccccc__dddddddd__
# __dead_block__eeeeeeee__ffffffff__gggggggg__hhhhhhhh__iiiiiiii__jjjjjjjj__
# __dead_block__kkkkkkkk__llllllll__mmmmmmmm__nnnnnnnn__oooooooo__pppppppp__
# __dead_block__qqqqqqqq__rrrrrrrr__ssssssss__tttttttt__uuuuuuuu__vvvvvvvv__
# __dead_block__wwwwwwww__xxxxxxxx__yyyyyyyy__zzzzzzzz__aaaaaaaa__bbbbbbbb__
# __dead_block__cccccccc__dddddddd__eeeeeeee__ffffffff__gggggggg__hhhhhhhh__
# __dead_block__iiiiiii_i__jjjjjjj__kkkkkkk__lllllll__mmmmmmm__nnnnnnn__
# __dead_block__oooooooo__ppppppp__qqqqqqq__rrrrrrr__sssssss__ttttttt__
# __dead_block__uuuuuuu__vvvvvvv__wwwwwww__xxxxxxx__yyyyyyy__zzzzzzz__

# __dead_block__identifier_sequence__unreachable_code__unused_variables__
# __dead_block__nonstandard_names__deeply_nested__complex_syntax__
# __dead_block__unused_variable_initializer__dead_region__
# __dead_block__structural_characteristics__code_representation__
# __dead_block__mimic_active_variables__syntax_only__
# __dead_block__unexecuted_logic__unreachable_block__

# __dead_block__complex_identifier_chain__unused__dead__zone__
# __dead_block__nested_unreachable_sequence__syntactic_mimicry__
# __dead_block__unused_variable_placeholder__identifier_sequence__
# __dead_block__nonstandard_names__dead_code__code_structure__
# __dead_block__unreachable_syntax__unused_variables__complex__
# __dead_block__identifier_var__unused__dead_block__
# __dead_block__unused_var_init__nonstandard_name__
# __dead_block__unreachable_sequence__nested_structure__
# __dead_block__complex_identifier_chain__unreachable_zone__
# __dead_block__unused_variable__identifier_sequence__
# __dead_block__nonstandard_names__dead_region__
# __dead_block__code_structure__syntax_only__
# __dead_block__unexecuted__unused_variables__complex_identifier__

# __dead_block__aaaaaa_bbbbb_cccccc_dddddd_eeeeee_ffffffff_gggggg_gggggg_hhhhhh_hhhhhh
# __dead_block__iiiiii_i_i_jjjjjj_jjjjjj_kkkkkk_kkkkkk_llllll_llllll_mmmmmm_mmmmmm
# __dead_block__nnnnnn_nnnnnn_oooooo_oooooo_pppppp_pppppp_qqqqqq_qqqqqq_rrrrrr_rrrrrr
# __dead_block__ssssss_ssssss_tttttt_tttttt_uuuuuu_uuuuuu_vvvvvv_vvvvvv_wwwwww_wwwwww
# __dead_block__xxxxxx_xxxxxx_yyyyyy_yyyyyy_zzzzzz_zzzzzz_aaaaaa_aaaaaa_bbbbbb_bbbbbb
# __dead_block__cccccc_cccccc_dddddd_dddddd_eeeeee_eeeeee_ffffffff_ffffffff_gggggg_gggggg
# __dead_block__hhhhh_hhhhh_iiiiiii_jjjjjjj_kkkkkkk_l_l_l_l_mmmmmm_mmmmmm_nnnnnn_nnnnnn
# __dead_block__oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo_pppppppp_ppppppp
# __dead_block__qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq_qqqqqqqqqqqq_rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr