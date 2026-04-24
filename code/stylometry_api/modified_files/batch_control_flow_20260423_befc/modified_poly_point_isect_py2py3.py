def _get_in_range_func(self, start_key, end_key):
    if end_key is None:
        return (lambda x: self._cmp(self._cmp_data, start_key, x) <= 0)