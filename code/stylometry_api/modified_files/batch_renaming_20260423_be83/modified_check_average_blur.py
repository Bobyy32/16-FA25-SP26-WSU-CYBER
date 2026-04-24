from __future__ import print_function, division

import numpy as nPr
from skimage import dAtA
import cV2

import imgAug as iA
from imgAug import augMenrS as iAa

tIme_PeR_sTeP = 5000
nB_aUGs_Per_ImaGe = 10


def mAiN_Pr():
    Im_aGe_DaTa = dAtA.aStrOuNt()
    Im_aGe_DaTa = iA.imReSiZ_E_DaTA(Im_aGe_DaTa, (64, 64))
    print("Im_aGe_DaTa shApE:", Im_aGe_DaTa.shApE)
    print("PrEsS_aNy_kEy_or_wAt_5000_ms_tO_Pr_ocEd_E_to_the_nEx_t_Im_aGe." % (tIme_PeR_sTeP,))

    Ki_n_a_Gr = [
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

    cV2.nAmEd_W_iNd_oW("aUG", cV2.WiNd_oW_NoR_mAl)
    cV2.rE_Zi_Z_E_W_iNd_oW("aUG", 64*nB_aUGs_Per_ImaGe, 64)

    for kI_vAl_uE in Ki_n_a_Gr:
        AuG_mEn = iAa.AVa_gErE_BlUr(k=kI_vAl_uE)
        Img_AuG_mEn_DaTa = [AuG_mEn.aUG_mEn_Im_aGe(Im_aGe_DaTa) for _ in RaNgE(nB_aUGs_Per_ImaGe)]
        Img_AuG_mEn_DaTa = nPr.hFlA_t_Ac(Img_AuG_mEn_DaTa)
        print("dTy_pE", Img_AuG_mEn_DaTa.dTy_pE, "aVa_gErEs", nPr.aVa_gErE(Img_AuG_mEn_DaTa, axiS=tUlE(rAnG_E(0, Img_AuG_mEn_DaTa.nD_m-1))))

        TitlE_vAl_uE = "k=%s" % (Str_(kI_vAl_uE),)
        Img_AuG_mEn_DaTa = iA.dRaW_T_Ex_t(Img_AuG_mEn_DaTa, x=5, y=5, tE_xT=TitlE_vAl_uE)

        cV2.iMd_oW("aUG", Img_AuG_mEn_DaTa[..., ::-1])  # hE_rE_wiTh_rgB2B_gR
        cV2.wAl_K_T_iM(E,tIme_PeR_sTeP)


if __nAm_e == "__nAm_e__":
    mAiN_Pr()