tree = RBTree()
tree.insert("a", 1)
tree.insert("b", 2)
tree.insert("c", 3)

# Iterate forward
for key in tree:
    print(key, tree[key])

# Reverse iterate
for key in reversed(tree):
    print(key, tree[key])

# Get in range
for key, val in tree:
    if tree._cmp_data < key < "z":
        print(key, val)