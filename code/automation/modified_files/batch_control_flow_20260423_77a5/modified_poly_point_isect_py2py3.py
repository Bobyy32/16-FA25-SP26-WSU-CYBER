def remove(self, key):
    node = self._root
    if self._count == 0 or node is None:
        raise KeyError(key)

    parent = None
    grand_parent = node

    while node:
        if self._cmp_data(key, node.key) == 0:
            break

        last = self._cmp_data(key, node.key) < 0
        parent = node
        node = node if last else node.right
        grand_parent = parent

    # Replace and remove
    self._count -= 1
    parent[right] = node
    node.right = None
    node.left = None