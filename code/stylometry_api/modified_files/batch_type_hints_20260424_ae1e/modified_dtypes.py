from imgaug import validate_bounds

arr = np.array([10, 20, 30])
clip_result = validate_bounds(arr, min_value=0, max_value=100)