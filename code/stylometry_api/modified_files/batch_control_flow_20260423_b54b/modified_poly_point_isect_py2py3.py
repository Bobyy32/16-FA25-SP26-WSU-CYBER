def _iter_items(self, left=attrgetter("left"), right=attrgetter("right"),
                start_key=None, end_key=None):
    # Iterates in-order using a stack-based approach
    # O(log n) per access, O(n) in worst case for traversal