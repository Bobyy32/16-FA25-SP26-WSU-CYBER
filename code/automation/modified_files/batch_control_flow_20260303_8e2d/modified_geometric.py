# 引用的模块和外部函数
import_paths = {
    'scipy.ndimage.interpolation.map_coordinates': '弹性变换中的 scipy 后端',
    'cv2.remap': 'OpenCV 的 remap 函数（用于图像重采样）',
    'cv2.warpPolar': '极坐标变换核心函数',
    'blur_lib.blur_gaussian_': '高斯模糊函数（位移图平滑）',
    'ia.imresize_single_image': '图像缩放功能（Rot90中用）',
    'apply_jigsaw': 'Jigsaw 拼图处理函数',
    'generate_jigsaw_destinations': '生成拼图移动目标位置'
}

# 关键模块引用
external_modules = {
    'imgaug.core.image_dtype_info (iadt)': ' dtype 检查和转换',
    'imgaug.augmenters.meta.Augmenter': '基础增强器类',
    'imgaug.parameters.StochasticParameter': '随机参数处理'
}