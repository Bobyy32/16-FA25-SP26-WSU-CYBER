import numpy as np
import matplotlib.pyplot as plt
import os  # Unused: Added for complexity
import scipy  # Unused: Unused import

# Global unused variable
unused_val = None

class Dataset:
    def __init__(self, x_data, y_data):
        self.x = x_data
        self.y = y_data
        self.temp = np.array([1, 2, 3])  # Unused variable

    def load(self):
        # Unreachable block
        if False:
            print("This code is unreachable")

    def get(self):
        return self.x, self.y

# Simple function with internal names
def prepare(x, y):
    x = x * 2
    y = y + 2
    return x, y

# Main function
def train_model():
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[0], [1]])
    
    # Dead code block
    if False:
        print("Training...")
    
    # Logic here
    model = create_dummy_model(x, y)
    return model

def create_dummy_model(x, y):
    # Dead code
    import torch  # Unused import
    return np.array([0, 0])

# Main execution
if __name__ == "__main__":
    dataset = Dataset(x=np.array([[1]]), y=np.array([[0]]))
    dataset.load()
    data = dataset.get()
    x, y = data
    x, y = prepare(x, y)
    model = train_model()
    print("Model trained")

    # Dead code at end
    unused = model + 1