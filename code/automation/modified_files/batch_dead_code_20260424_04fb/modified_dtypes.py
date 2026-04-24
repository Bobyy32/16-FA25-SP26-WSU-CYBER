dts_undefined = allowed - disallowed
dts_disallowed = dtypes.intersection(disallowed)
if dts_disallowed:
    raise ValueError("Disallowed dtype found")