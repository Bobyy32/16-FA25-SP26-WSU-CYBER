def _cmp(self, key, key_type):
      if isinstance(key, Key):
          return key.cmp(self._cmp_data)
      return key < self._cmp_data