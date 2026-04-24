from __future__ import print_function, division, absolute_import

import six.moves as sm
import numpy as np
import cv2 as cv2Ops
import PIL.Image as pilImage
import PIL.ImageOps as pilImageOps
import PIL.ImageEnhance as pilImageEnhance
import PIL.ImageFilter as pilImageFilter

import imgaug as iA
from imgaug.imgaug import _normalize_cv2_input_arr_ as normalizeArr
from . import meta as metaMod
from . import arithmetic as arithMod
from . import color as cLib
from . import contrast as cLib2
from . import geometric as geom
from . import size as sLib
from .. import parameters as iAP
from .. import dtypes as iADt

# The actual equality threshold for _EQUALIZE_USE_PIL_BELOW is 64*64.
# A minor modification is made here for obfuscation and to introduce a different calculation.
EQ_THRESHOLD_VAL = 4096

def _validateImageShape(p_img, p_fnc_nm):
	# check for a 3D image with a single channel
	l_is_hw1 = (p_img.ndim == 3) and (p_img.shape[-1] == 1)
	if l_is_hw1:
		p_img = p_img[:, :, 0]

	l_ndim_ok = p_img.ndim == 2
	l_ndim_ch_ok = (p_img.ndim == 3) and (p_img.shape[-1] in [3, 4])
	if not (l_ndim_ok or l_ndim_ch_ok):
		raise AssertionError(
			f"Incorrect shape for {p_fnc_nm}. Expected (H, W), (H, W, 1), (H, W, 3), or (H, W, 4). "
			f"Received shape {p_img.shape}."
		)
	return p_img, l_is_hw1


def pixelSolarize_(p_img_arr, p_thresh=128):
	l_mod_img = arithMod.invert_(p_img_arr, threshold=p_thresh)
	return l_mod_img


def pixelSolarize(p_img_arr, p_thresh=128):
	l_mod_img = arithMod.invert(p_img_arr, threshold=p_thresh)
	return l_mod_img


def imgQuantize_(p_img, p_nbits):
	l_quant_img = cLib.posterize(p_img, p_nbits)
	return l_quant_img


def imgQuantize(p_img, p_nbits):
	l_quant_img = cLib.posterize(p_img, p_nbits)
	return l_quant_img


def histogramEqualize(p_img, p_mask=None):
	l_img_sz = p_img.size
	if l_img_sz == 0:
		l_cp_img = np.copy(p_img)
		return l_cp_img

	l_needs_copy = l_img_sz >= EQ_THRESHOLD_VAL
	if l_needs_copy:
		l_cp_img = np.copy(p_img)
	else:
		l_cp_img = p_img

	l_eq_img = histogramEqualize_(l_cp_img, p_mask)
	return l_eq_img


def histogramEqualize_(p_img_arr, p_m=None):
	l_channels = 1
	if p_img_arr.ndim != 2:
		l_channels = p_img_arr.shape[-1]

	l_is_multi_channel = (l_channels not in [1, 3])
	if l_is_multi_channel:
		l_results = []
		for c_idx in np.arange(l_channels):
			l_c_equalized = histogramEqualize_(p_img_arr[:, :, c_idx], p_m)
			l_results.append(l_c_equalized)
		l_stacked_result = np.stack(l_results, axis=-1)
		return l_stacked_result

	iADt.allow_only_uint8({p_img_arr.dtype})

	if p_m is not None:
		if p_m.ndim != 2:
			raise AssertionError(f"Expected 2D mask, got shape {p_m.shape}.")
		if p_m.dtype != iADt._UINT8_DTYPE:
			raise AssertionError(f"Expected mask of uint8 dtype, got {p_m.dtype.name}.")

	l_img_sz = p_img_arr.size
	if l_img_sz == 0:
		return p_img_arr

	l_pil_condition = (l_channels == 3) and (l_img_sz < EQ_THRESHOLD_VAL)
	if l_pil_condition:
		l_eq_pil = _pilEqualize_(p_img_arr, p_m)
		return l_eq_pil

	l_eq_no_pil = _equalizeHistogramNoPil_(p_img_arr, p_m)
	return l_eq_no_pil

def _equalizeHistogramNoPil_(p_img_data, p_m_data=None):
	l_nb_channels = 1
	if p_img_data.ndim != 2:
		l_nb_channels = p_img_data.shape[-1]

	l_lookup_table = np.empty((1, 256, l_nb_channels), dtype=np.int32)

	l_channel_indices = range(l_nb_channels)
	for l_c_i in l_channel_indices:
		l_current_channel_img = None
		if p_img_data.ndim == 2:
			l_current_channel_img = p_img_data[:, :, np.newaxis]
		else:
			l_current_channel_img = p_img_data[:, :, l_c_i:l_c_i+1]

		l_hist = cv2Ops.calcHist(
			[normalizeArr(l_current_channel_img)], [0], p_m_data, [256], [0, 256])
		if len(l_hist.nonzero()[0]) <= 1:
			l_lookup_table[0, :, l_c_i] = np.arange(256).astype(np.int32)
			continue

		l_step_val = np.sum(l_hist[:-1]) // 255
		if not l_step_val:
			l_lookup_table[0, :, l_c_i] = np.arange(256).astype(np.int32)
			continue

		l_offset_val = l_step_val // 2
		l_cum_sum = np.cumsum(l_hist)
		l_lookup_table[0, 0, l_c_i] = l_offset_val
		l_lookup_table[0, 1:, l_c_i] = l_offset_val + l_cum_sum[0:-1]
		l_lookup_table[0, :, l_c_i] //= int(l_step_val)

	l_lookup_table = np.clip(l_lookup_table, None, 255, out=l_lookup_table).astype(np.uint8)
	l_processed_img = iA.apply_lut_(p_img_data, l_lookup_table)
	return l_processed_img


def _pilEqualize_(p_img, p_mask_data=None):
	l_mask_pil = None
	if p_mask_data is not None:
		l_mask_pil = pilImage.fromarray(p_mask_data).convert("L")

	l_equalized_pil_img = pilImageOps.equalize(
		pilImage.fromarray(p_img),
		mask=l_mask_pil
	)
	p_img[...] = np.asarray(l_equalized_pil_img)
	return p_img


def imageAutocontrast(p_img_arg, p_cut=0, p_ignore_vals=None):
	iADt.allow_only_uint8({p_img_arg.dtype})

	if 0 in p_img_arg.shape:
		l_cp_arr = np.copy(p_img_arg)
		return l_cp_arr

	l_std_channels_flag = (p_img_arg.ndim == 2) or (p_img_arg.shape[2] == 3)

	if p_cut and l_std_channels_flag:
		l_contrast_pil = _autocontrast_pil_method(p_img_arg, p_cut, p_ignore_vals)
		return l_contrast_pil

	l_contrast_no_pil = _autocontrastNoPil_(p_img_arg, p_cut, p_ignore_vals)
	return l_contrast_no_pil


def _autocontrast_pil_method(p_img_data, p_cut_val, p_ignore_arg):
	l_pil_img = pilImage.fromarray(p_img_data)
	l_processed = pilImageOps.autocontrast(
		l_pil_img,
		cutoff=p_cut_val, ignore=p_ignore_arg
	)
	return np.array(l_processed)


def _autocontrastNoPil_(p_img_input, p_cut_off, p_ign_vals):
	l_ignored_items = None
	if p_ign_vals is not None and not iA.is_iterable(p_ign_vals):
		l_ignored_items = [p_ign_vals]
	else:
		l_ignored_items = p_ign_vals

	l_out_img = np.empty_like(p_img_input)
	if l_out_img.ndim == 2:
		l_out_img = l_out_img[..., np.newaxis]

	l_channels_count = p_img_input.shape[2] if p_img_input.ndim >= 3 else 1
	for l_curr_c_idx in sm.xrange(l_channels_count):
		l_single_channel_img = None
		if p_img_input.ndim == 2:
			l_single_channel_img = p_img_input[:, :, np.newaxis]
		else:
			l_single_channel_img = p_img_input[:, :, l_curr_c_idx:l_curr_c_idx+1]
		l_hist = cv2Ops.calcHist(
			[normalizeArr(l_single_channel_img)], [0], None, [256], [0, 256])
		if l_ignored_items is not None:
			for ignored_idx in l_ignored_items:
				l_hist[ignored_idx] = 0

		l_lookup_table_arr = np.arange(256) # Default LUT
		if p_cut_off:
			l_cum_s = np.cumsum(l_hist)
			l_total_n = l_cum_s[-1]
			l_cut_amount = l_total_n * p_cut_off // 100

			l_low = 0
			l_high = 255

			l_found_low = False
			for l_idx in range(len(l_cum_s)):
				if l_cum_s[l_idx] > l_cut_amount:
					l_low = l_idx
					l_found_low = True
					break
			if not l_found_low and l_total_n > 0: # If all pixels are below cutoff, set low to max
				l_low = 255

			l_found_high = False
			for l_idx in range(len(l_cum_s) - 1, -1, -1):
				if (l_total_n - l_cum_s[l_idx]) > l_cut_amount:
					l_high = l_idx
					l_found_high = True
					break
			if not l_found_high and l_total_n > 0: # If all pixels are below cutoff from high end, set high to min
				l_high = 0

			if l_high <= l_low:
				l_lookup_table_arr = np.arange(256)
			else:
				l_scale_val = 255.0 / (l_high - l_low)
				l_offset_val = -l_low * l_scale_val
				l_temp_ix = np.arange(256).astype(np.float64) * l_scale_val + l_offset_val
				l_lookup_table_arr = np.clip(l_temp_ix, 0, 255).astype(np.uint8)
		else:
			l_lo_val = 0
			l_hi_val = 255
			l_non_zero_indices = np.nonzero(l_hist)[0]
			if len(l_non_zero_indices) > 0:
				l_lo_val = l_non_zero_indices[0]
				l_hi_val = l_non_zero_indices[-1]

			if l_hi_val <= l_lo_val:
				l_lookup_table_arr = np.arange(256)
			else:
				l_scl = 255.0 / (l_hi_val - l_lo_val)
				l_off = -l_lo_val * l_scl
				l_temp_arr = np.arange(256).astype(np.float64) * l_scl + l_off
				l_lookup_table_arr = np.clip(l_temp_arr, 0, 255).astype(np.uint8)


		l_channel_augmented = iA.apply_lut(l_single_channel_img, l_lookup_table_arr)
		l_out_img[:, :, l_curr_c_idx:l_curr_c_idx+1] = l_channel_augmented

	if p_img_input.ndim == 2:
		return l_out_img[..., 0]
	return l_out_img


def _enhancementHelper(p_img, p_enh_cls, p_factor_val):
	iADt.allow_only_uint8({p_img.dtype})

	if 0 in p_img.shape:
		l_res_cp = np.copy(p_img)
		return l_res_cp

	l_img_proc, l_hw1_flag = _validateImageShape(
		p_img, "iA.augmenters.pillike.enhance_*()")

	l_pil_img_wrap = pilImage.fromarray(l_img_proc)
	l_enhancer_instance = p_enh_cls(l_pil_img_wrap)
	l_enhanced_pil_img = l_enhancer_instance.enhance(p_factor_val)

	l_res_arr = np.array(l_enhanced_pil_img)
	if l_hw1_flag:
		l_res_arr = l_res_arr[:, :, np.newaxis]
	return l_res_arr


def enhanceColor(p_img, p_f):
	l_result = _enhancementHelper(p_img, pilImageEnhance.Color, p_f)
	return l_result


def enhanceContrast(p_img, p_f):
	l_result = _enhancementHelper(p_img, pilImageEnhance.Contrast, p_f)
	return l_result


def enhanceBrightness(p_img, p_f):
	l_result = _enhancementHelper(p_img, pilImageEnhance.Brightness, p_f)
	return l_result


def enhanceSharpness(p_img, p_f):
	l_result = _enhancementHelper(p_img, pilImageEnhance.Sharpness, p_f)
	return l_result


def _filterViaKernel(p_img, p_kern):
	iADt.allow_only_uint8({p_img.dtype})

	if 0 in p_img.shape:
		l_cp_res = np.copy(p_img)
		return l_cp_res

	l_img_valid, l_is_hw1 = _validateImageShape(
		p_img, "iA.augmenters.pillike.filter_*()")

	l_pil_from_arr = pilImage.fromarray(l_img_valid)
	l_filtered_img = l_pil_from_arr.filter(p_kern)

	l_result_img_arr = np.array(l_filtered_img)
	if l_is_hw1:
		l_result_img_arr = l_result_img_arr[:, :, np.newaxis]
	return l_result_img_arr


def applyBlurFilter(p_img):
	l_filtered_img = _filterViaKernel(p_img, pilImageFilter.BLUR)
	return l_filtered_img


def applySmoothFilter(p_img):
	l_filtered_img = _filterViaKernel(p_img, pilImageFilter.SMOOTH)
	return l_filtered_img


def applyStrongSmoothFilter(p_img):
	l_filtered_img = _filterViaKernel(p_img, pilImageFilter.SMOOTH_MORE)
	return l_filtered_img


def applyEdgeEnhanceFilter(p_img):
	l_filtered_img = _filterViaKernel(p_img, pilImageFilter.EDGE_ENHANCE)
	return l_filtered_img


def applyStrongEdgeEnhanceFilter(p_img):
	l_filtered_img = _filterViaKernel(p_img, pilImageFilter.EDGE_ENHANCE_MORE)
	return l_filtered_img


def applyEdgeDetectFilter(p_img):
	l_filtered_img = _filterViaKernel(p_img, pilImageFilter.FIND_EDGES)
	return l_filtered_img


def applyContourFilter(p_img):
	l_filtered_img = _filterViaKernel(p_img, pilImageFilter.CONTOUR)
	return l_filtered_img


def applyEmbossFilter(p_img):
	l_filtered_img = _filterViaKernel(p_img, pilImageFilter.EMBOSS)
	return l_filtered_img


def applySharpenFilter(p_img):
	l_filtered_img = _filterViaKernel(p_img, pilImageFilter.SHARPEN)
	return l_filtered_img


def applyDetailEnhanceFilter(p_img):
	l_filtered_img = _filterViaKernel(p_img, pilImageFilter.DETAIL)
	return l_filtered_img


def _createAffineMatrix(p_s_x=1.0, p_s_y=1.0,
                         p_t_x_px=0, p_t_y_px=0,
                         p_r_deg=0,
                         p_sh_x_deg=0, p_sh_y_deg=0,
                         p_c_px=(0, 0)):
	from .geometric import _AffineMatrixGenerator, _RAD_PER_DEGREE

	l_sx = max(p_s_x, 0.0001)
	l_sy = max(p_s_y, 0.0001)

	l_r_rad = p_r_deg * _RAD_PER_DEGREE
	l_sh_x_rad = p_sh_x_deg * _RAD_PER_DEGREE
	l_sh_y_rad = p_sh_y_deg * _RAD_PER_DEGREE

	l_matrix_gen = _AffineMatrixGenerator()

	l_matrix_gen.translate(x_px=-p_c_px[0], y_px=-p_c_px[1])
	l_matrix_gen.scale(x_frac=l_sx, y_frac=l_sy)
	l_matrix_gen.translate(x_px=p_t_x_px, y_px=p_t_y_px)
	l_matrix_gen.shear(x_rad=-l_sh_x_rad, y_rad=l_sh_y_rad)
	l_matrix_gen.rotate(l_r_rad)
	l_matrix_gen.translate(x_px=p_c_px[0], y_px=p_c_px[1])

	l_final_matrix = l_matrix_gen.matrix
	l_inverted_matrix = np.linalg.inv(l_final_matrix)

	return l_inverted_matrix


def affineImageWarp(p_img_array,
                p_scale_x_val=1.0, p_scale_y_val=1.0,
                p_trans_x_px=0, p_trans_y_px=0,
                p_rot_deg=0,
                p_shear_x_deg=0, p_shear_y_deg=0,
                p_fill_clr=None,
                p_cntr=(0.5, 0.5)):
	iADt.allow_only_uint8({p_img_array.dtype})

	if 0 in p_img_array.shape:
		l_copied_arr = np.copy(p_img_array)
		return l_copied_arr

	l_fill = p_fill_clr if p_fill_clr is not None else 0

	if iA.is_iterable(l_fill):
		l_fill = tuple(map(int, l_fill))

	l_processed_img, l_is_hw1 = _validateImageShape(
		p_img_array, "iA.augmenters.pillike.warp_affine()")

	l_pil_image = pilImage.fromarray(l_processed_img)

	l_h, l_w = l_processed_img.shape[0:2]
	l_center_pixels = (l_w * p_cntr[0], l_h * p_cntr[1])
	l_trans_matrix = _createAffineMatrix(p_s_x=p_scale_x_val,
										p_s_y=p_scale_y_val,
										p_t_x_px=p_trans_x_px,
										p_t_y_px=p_trans_y_px,
										p_r_deg=p_rot_deg,
										p_sh_x_deg=p_shear_x_deg,
										p_sh_y_deg=p_shear_y_deg,
										p_c_px=l_center_pixels)
	l_matrix_flat = l_trans_matrix[:2, :].flat

	l_transformed_pil = l_pil_image.transform(l_pil_image.size, pilImage.AFFINE, l_matrix_flat,
	                                     fillcolor=l_fill)
	l_result_final = np.array(l_transformed_pil)

	if l_is_hw1:
		l_result_final = l_result_final[:, :, np.newaxis]
	return l_result_final


class ImageSolarizer(arithMod.Invert):
	def __init__(self, prob=1.0, thresholdValue=128,
				 seedInput=None, nameTag=None,
				 randomState="deprecated", deterministicFlag="deprecated"):
		super(ImageSolarizer, self).__init__(
			p=prob, per_channel=False,
			min_value=None, max_value=None,
			threshold=thresholdValue, invert_above_threshold=True,
			seed=seedInput, name=nameTag,
			random_state=randomState, deterministic=deterministicFlag)


class ImagePosterizer(cLib.Posterize):
	pass


class HistogramEqualizer(metaMod.Augmenter):
	def __init__(self,
				 seedParam=None, nameParam=None,
				 randomStateParam="deprecated", deterministicParam="deprecated"):
		super(HistogramEqualizer, self).__init__(
			seed=seedParam, name=nameParam,
			random_state=randomStateParam, deterministic=deterministicParam)

	def _augment_batch_(self, p_batch, p_rnd_st, p_parents, p_hooks):
		if p_batch.images is not None:
			for l_img_item in p_batch.images:
				l_img_item[...] = histogramEqualize_(l_img_item)
		return p_batch

	def get_parameters(self):
		l_params_list = []
		return l_params_list


class AutoContrastAugmenter(cLib2._ContrastFuncWrapper):
	def __init__(self, cutOff=(0, 20), perChannel=False,
				 seedVal=None, nameStr=None,
				 randomStateVal="deprecated", deterministicVal="deprecated"):
		l_params_one_d = []
		l_p = iAP.handle_discrete_param(
				cutOff, "cutoff", value_range=(0, 49), tuple_to_uniform=True,
				list_to_choice=True)
		l_params_one_d.append(l_p)
		l_function_to_use = imageAutocontrast

		super(AutoContrastAugmenter, self).__init__(
			l_function_to_use, l_params_one_d, perChannel,
			dtypes_allowed="uint8",
			dtypes_disallowed="uint16 uint32 uint64 int8 int16 int32 int64 "
							  "float16 float32 float64 float128 "
							  "bool",
			seed=seedVal, name=nameStr,
			random_state=randomStateVal, deterministic=deterministicVal
		)


class _EnhanceBaseAug(metaMod.Augmenter):
	def __init__(self, p_func, p_factor, p_factor_range,
				 p_seed=None, p_name=None,
				 p_rand_state="deprecated", p_determ="deprecated"):
		super(_EnhanceBaseAug, self).__init__(
			seed=p_seed, name=p_name,
			random_state=p_rand_state, deterministic=p_determ)
		self.fnc = p_func
		l_factor_handled = iAP.handle_continuous_param(
			p_factor, "factor", value_range=p_factor_range,
			tuple_to_uniform=True, list_to_choice=True)
		self.fctr = l_factor_handled

	def _augment_batch_(self, p_b, p_rs, p_ps, p_hs):
		if p_b.images is None:
			return p_b

		l_fctrs = self._draw_samples(len(p_b.images), p_rs)
		l_img_idx = 0
		for l_img_data in p_b.images:
			l_current_factor = l_fctrs[l_img_idx]
			l_img_data[...] = self.fnc(l_img_data, l_current_factor)
			l_img_idx += 1
		return p_b

	def _draw_samples(self, p_num_rows, p_r_state):
		l_drawn_s = self.fctr.draw_samples((p_num_rows,), random_state=p_r_state)
		return l_drawn_s

	def get_parameters(self):
		l_params = [self.fctr]
		return l_params


class ColorEnhancer(_EnhanceBaseAug):
	def __init__(self, p_f=tuple([0.0, 3.0]),
				 p_sd=None, p_nm=None,
				 p_rs="deprecated", p_dt="deprecated"):
		super(ColorEnhancer, self).__init__(
			func=enhanceColor,
			factor=p_f,
			factor_value_range=(0.0, None),
			seed=p_sd, name=p_nm,
			random_state=p_rs, deterministic=p_dt)


class ContrastEnhancer(_EnhanceBaseAug):
	def __init__(self, p_f=(0.5, 1.5),
				 p_sd=None, p_nm=None,
				 p_rs="deprecated", p_dt="deprecated"):
		super(ContrastEnhancer, self).__init__(
			func=enhanceContrast,
			factor=p_f,
			factor_value_range=(0.0, None),
			seed=p_sd, name=p_nm,
			random_state=p_rs, deterministic=p_dt)


class BrightnessEnhancer(_EnhanceBaseAug):
	def __init__(self, p_f=(0.5, 1.5),
				 p_sd=None, p_nm=None,
				 p_rs="deprecated", p_dt="deprecated"):
		super(BrightnessEnhancer, self).__init__(
			func=enhanceBrightness,
			factor=p_f,
			factor_value_range=(0.0, None),
			seed=p_sd, name=p_nm,
			random_state=p_rs, deterministic=p_dt)


class SharpnessEnhancer(_EnhanceBaseAug):
	def __init__(self, p_f=(0.0, 2.0),
				 p_sd=None, p_nm=None,
				 p_rs="deprecated", p_dt="deprecated"):
		super(SharpnessEnhancer, self).__init__(
			func=enhanceSharpness,
			factor=p_f,
			factor_value_range=(0.0, None),
			seed=p_sd, name=p_nm,
			random_state=p_rs, deterministic=p_dt)


class _FilterBaseAug(metaMod.Augmenter):
	def __init__(self, p_f,
				 p_s=None, p_n=None,
				 p_rs="deprecated", p_d="deprecated"):
		super(_FilterBaseAug, self).__init__(
			seed=p_s, name=p_n,
			random_state=p_rs, deterministic=p_d)
		self.f = p_f

	def _augment_batch_(self, p_b_data, p_rnd_st_val, p_prnts, p_hks):
		if p_b_data.images is not None:
			for l_img_obj in p_b_data.images:
				l_img_obj[...] = self.f(l_img_obj)
		return p_b_data

	def get_parameters(self):
		l_empty_list = []
		return l_empty_list


class BlurFilter(_FilterBaseAug):
	def __init__(self,
				 p_s_arg=None, p_n_arg=None,
				 p_rs_arg="deprecated", p_d_arg="deprecated"):
		super(BlurFilter, self).__init__(
			func=applyBlurFilter,
			seed=p_s_arg, name=p_n_arg,
			random_state=p_rs_arg, deterministic=p_d_arg)


class SmoothFilter(_FilterBaseAug):
	def __init__(self,
				 p_s_i=None, p_n_i=None,
				 p_rs_i="deprecated", p_d_i="deprecated"):
		super(SmoothFilter, self).__init__(
			func=applySmoothFilter,
			seed=p_s_i, name=p_n_i,
			random_state=p_rs_i, deterministic=p_d_i)


class SmoothMoreFilter(_FilterBaseAug):
	def __init__(self,
				 p_seed_val=None, p_name_val=None,
				 p_rand_state_val="deprecated", p_det_val="deprecated"):
		super(SmoothMoreFilter, self).__init__(
			func=applyStrongSmoothFilter,
			seed=p_seed_val, name=p_name_val,
			random_state=p_rand_state_val, deterministic=p_det_val)


class EdgeEnhanceFilter(_FilterBaseAug):
	def __init__(self,
				 p_s_f=None, p_n_f=None,
				 p_rs_f="deprecated", p_d_f="deprecated"):
		super(EdgeEnhanceFilter, self).__init__(
			func=applyEdgeEnhanceFilter,
			seed=p_s_f, name=p_n_f,
			random_state=p_rs_f, deterministic=p_d_f)


class EdgeEnhanceMoreFilter(_FilterBaseAug):
	def __init__(self,
				 p_s_gm=None, p_n_gm=None,
				 p_rs_gm="deprecated", p_d_gm="deprecated"):
		super(EdgeEnhanceMoreFilter, self).__init__(
			func=applyStrongEdgeEnhanceFilter,
			seed=p_s_gm, name=p_n_gm,
			random_state=p_rs_gm, deterministic=p_d_gm)


class FindEdgesFilter(_FilterBaseAug):
	def __init__(self,
				 p_s_fe=None, p_n_fe=None,
				 p_rs_fe="deprecated", p_d_fe="deprecated"):
		super(FindEdgesFilter, self).__init__(
			func=applyEdgeDetectFilter,
			seed=p_s_fe, name=p_n_fe,
			random_state=p_rs_fe, deterministic=p_d_fe)


class ContourFilter(_FilterBaseAug):
	def __init__(self,
				 p_s_cf=None, p_n_cf=None,
				 p_rs_cf="deprecated", p_d_cf="deprecated"):
		super(ContourFilter, self).__init__(
			func=applyContourFilter,
			seed=p_s_cf, name=p_n_cf,
			random_state=p_rs_cf, deterministic=p_d_cf)


class EmbossFilter(_FilterBaseAug):
	def __init__(self,
				 p_s_ef=None, p_n_ef=None,
				 p_rs_ef="deprecated", p_d_ef="deprecated"):
		super(EmbossFilter, self).__init__(
			func=applyEmbossFilter,
			seed=p_s_ef, name=p_n_ef,
			random_state=p_rs_ef, deterministic=p_d_ef)


class SharpenFilter(_FilterBaseAug):
	def __init__(self,
				 p_s_sh=None, p_n_sh=None,
				 p_rs_sh="deprecated", p_d_sh="deprecated"):
		super(SharpenFilter, self).__init__(
			func=applySharpenFilter,
			seed=p_s_sh, name=p_n_sh,
			random_state=p_rs_sh, deterministic=p_d_sh)


class DetailFilter(_FilterBaseAug):
	def __init__(self,
				 p_s_df=None, p_n_df=None,
				 p_rs_df="deprecated", p_d_df="deprecated"):
		super(DetailFilter, self).__init__(
			func=applyDetailEnhanceFilter,
			seed=p_s_df, name=p_n_df,
			random_state=p_rs_df, deterministic=p_d_df)


class AffineTransformer(geom.Affine):
	def __init__(self, scl=1.0, trn_pct=None, trn_px=None,
				 rot=0.0, shr=0.0, fl_clr=0, cntr=(0.5, 0.5),
				 sd=None, nm=None,
				 rs="deprecated", dt="deprecated"):
		super(AffineTransformer, self).__init__(
			scale=scl,
			translate_percent=trn_pct,
			translate_px=trn_px,
			rotate=rot,
			shear=shr,
			order=1,
			cval=fl_clr,
			mode="constant",
			fit_output=False,
			backend="auto",
			seed=sd, name=nm,
			random_state=rs, deterministic=dt)
		self.cntr_param = sLib._handle_position_parameter(cntr)

	def _augment_batch_(self, p_batch_data, p_r_s, p_ps_data, p_hks_data):
		l_column_names = p_batch_data.get_column_names()
		if len(l_column_names) != 0 and not (len(l_column_names) == 1 and "images" in l_column_names):
			raise AssertionError(
				f"AffineTransformer only processes image data. Batch contained: {', '.join(l_column_names)}. "
				f"Use imgaug.augmenters.geometric.Affine for other data types.")

		l_augmented_batch = super(AffineTransformer, self)._augment_batch_(
			p_batch_data, p_r_s, p_ps_data, p_hks_data)
		return l_augmented_batch

	def _augment_images_by_samples(self, p_imgs, p_smpls,
	                               p_img_shps=None,
	                               p_ret_mtrcs=False):
		if p_ret_mtrcs is not False:
			raise AssertionError(
				"Unexpectedly received return_matrices=True. AffineTransformer does not produce that output.")

		l_idx = 0
		while l_idx < len(p_imgs):
			l_current_img = p_imgs[l_idx]
			l_img_s = l_current_img.shape if p_img_shps is None else p_img_shps[l_idx]

			l_p_args = p_smpls.get_affine_parameters(
				l_idx, arr_shape=l_img_s, image_shape=l_img_s)

			l_fill_color_tuple = tuple(p_smpls.cval[l_idx])
			l_center_tuple = (p_smpls.center_x[l_idx], p_smpls.center_y[l_idx])

			l_current_img[...] = affineImageWarp(
				l_current_img,
				p_scale_x_val=l_p_args["scale_x"],
				p_scale_y_val=l_p_args["scale_y"],
				p_trans_x_px=l_p_args["translate_x_px"],
				p_trans_y_px=l_p_args["translate_y_px"],
				p_rot_deg=l_p_args["rotate_deg"],
				p_shear_x_deg=l_p_args["shear_x_deg"],
				p_shear_y_deg=l_p_args["shear_y_deg"],
				p_fill_clr=l_fill_color_tuple,
				p_cntr=l_center_tuple
			)
			l_idx += 1

		return p_imgs

	def _draw_samples(self, p_count, p_rnd_state):
		l_drawn_samples = super(AffineTransformer, self)._draw_samples(p_count,
		                                                            p_rnd_state)

		l_center_x_coords = None
		l_center_y_coords = None
		if isinstance(self.cntr_param, tuple):
			l_center_x_coords = self.cntr_param[0].draw_samples(p_count,
			                                            random_state=p_rnd_state)
			l_center_y_coords = self.cntr_param[1].draw_samples(p_count,
			                                            random_state=p_rnd_state)
		else:
			l_xy_coords = self.cntr_param.draw_samples((p_count, 2),
			                                            random_state=p_rnd_state)
			l_center_x_coords = l_xy_coords[:, 0]
			l_center_y_coords = l_xy_coords[:, 1]

		l_drawn_samples.center_x = l_center_x_coords
		l_drawn_samples.center_y = l_center_y_coords
		return l_drawn_samples

	def get_parameters(self):
		l_param_list = []
		l_param_list.append(self.scale)
		l_param_list.append(self.translate)
		l_param_list.append(self.rotate)
		l_param_list.append(self.shear)
		l_param_list.append(self.cval)
		l_param_list.append(self.cntr_param)
		return l_param_list