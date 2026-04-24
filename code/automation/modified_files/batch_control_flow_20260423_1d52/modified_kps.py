hitidx_flat = np.argmax(distance_maps[..., i])
   hitidx_ndim = np.unravel_index(hitidx_flat, (height, width))