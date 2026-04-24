def insert(self, key, value):
    if self._root is None:
        self._root = Node(key, value)
        self._root.red = False

    else:
        parent = None
        current = self._root
        while True:
            if self._cmp(key, current.key) < 0:
                parent, current = current, current.left
            else:
                parent, current = current, current.right

            if current is None:
                child = Node(key, value)
                child.red = True
                parent.right = child if key > current.key else child.left
                self._fixup(current)
                break

        self._count += 1