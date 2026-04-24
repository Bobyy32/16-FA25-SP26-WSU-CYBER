import numpy as np
from imgaug.datasets import gate_dtypes # Pseudo-name from snippet

# Scenario: Only uint8 is allowed, others are disallowed
allowed_types = [np.uint8]
disallowed_types = [] 

# User passes float64 data
dtypes = [np.float64] 

try:
    # This will raise ValueError because float64 is in disallowed_types
    gate_dtypes(dtypes, allowed_types, disallowed_types, None) 
except ValueError as e:
    print("Error:", e)

# Scenario: No specific list, just a general check
try:
    # This might issue a warning if np.float32 is not in the implicit allow list
    gate_dtypes(dtypes, allowed_types=(), disallowed_types=(), None) 
except ValueError:
    pass