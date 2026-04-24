# Potential issue if end_key <= start_key
while True:
    ...
    if start_key is None and end_key is None:
        return lambda x: True  # OK
    # Otherwise logic may be incorrect