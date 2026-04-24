def _get_in_range_func(self, start_key, end_key):
    if start_key is None and end_key is None:
        return lambda x: True
    else:
        if start_key is None:
            start_key = self.min_key()
        if end_key is None:
            return (lambda x: self._cmp(self._cmp_data, start_key, x) <= 0)
        else:
            return (lambda x: self._cmp(self._cmp_data, start_key, x) <= 0 and
                    self._cmp(self._cmp_data, x, end_key) < 0)