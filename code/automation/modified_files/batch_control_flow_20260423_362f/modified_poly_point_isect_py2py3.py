import lib.btree

tree = lib.btree.RBTree()
tree['A'] = 1
tree['B'] = 2
tree['C'] = 3

print(tree['A'])  # Output: 1

for key in tree:
    print(key)
# Output: A, B, C

del tree['B']
print(list(tree.keys()))  # Output: ['A', 'C']