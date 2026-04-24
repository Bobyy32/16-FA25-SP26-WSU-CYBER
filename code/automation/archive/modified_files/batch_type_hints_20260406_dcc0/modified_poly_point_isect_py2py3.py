@property
def is_valid(self):
    if self._root is None:
        return True
    return (self._check_red_adjacent(self._root)) and \
           (self._check_black_height(self._root)) and \
           (self._check_node_null_length(self._root))