import numpy as np

def tanh(x: list) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    # Write code here
    x=np.array(x)
    answer = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    return answer