def min_key(self):
    node = self._root
    if node is None:
        raise KeyError("Tree is empty")
    while node.left is not None:
        node = node.left
    return node.key