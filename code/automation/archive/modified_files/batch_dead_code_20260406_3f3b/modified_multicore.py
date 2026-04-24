# Use a sentinel object instead of None
class _Sentinel:
    pass
sentinel = _Sentinel()
# Or use a different object for queue signaling
import pickle
sentinel_pickled = pickle.dumps(_Sentinel())