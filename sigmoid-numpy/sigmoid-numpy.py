import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    if isinstance(x, list):
        x = np.array(x)
        return 1 / (1 + np.exp(-x))

    return 1 / (1 + np.exp(-x))