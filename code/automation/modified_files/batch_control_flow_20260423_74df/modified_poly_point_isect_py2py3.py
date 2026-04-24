from collections.abc import KeysView, ItemsView, ValuesView
import operator

class _ABCTree:
    """Abstract balanced tree base class."""

    def __init__(self, compare=None):
        self._root = None
        self._count = 0
        self._compare = compare or operator.itemgetter(0)
        self._cmp_data = self._compare.__name__
        self._cmp = operator.methodcaller("_cmp")