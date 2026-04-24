def _augment_kpsoi_by_samples(self, kpsoi, row_idx, samples, dx, dy):
    # Computes geometric median of displaced keypoints
    # Handles image boundary clipping