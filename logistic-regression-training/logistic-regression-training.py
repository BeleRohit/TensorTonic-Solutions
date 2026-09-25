import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(
        z >= 0,
        1 / (1 + np.exp(-z)),
        np.exp(z) / (1 + np.exp(z))
    )


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    steps: int = 1000
) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    m, n = X.shape

    # Initialize parameters
    w = np.zeros(n)
    b = 0.0

    for _ in range(steps):

        # Forward propagation
        z = X @ w + b
        y_hat = _sigmoid(z)

        # Gradients
        dw = (X.T @ (y_hat - y)) / m
        db = np.mean(y_hat - y)

        # Gradient descent
        w -= lr * dw
        b -= lr * db

    return w, b