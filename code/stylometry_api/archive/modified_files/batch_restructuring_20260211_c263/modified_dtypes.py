# Handles various inputs:
normalize_dtype(np.int32)        # Returns np.dtype('int32')
normalize_dtype('int32')         # Returns np.dtype('int32')
normalize_dtype(np.dtype('int32')) # Returns np.dtype('int32')