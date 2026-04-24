deg = int(direction_sample * 360) % 360
rad = np.deg2rad(deg)
x = np.cos(rad - 0.5*np.pi)
y = np.sin(rad - 0.5*np.pi)
direction_vector = np.array([x, y])