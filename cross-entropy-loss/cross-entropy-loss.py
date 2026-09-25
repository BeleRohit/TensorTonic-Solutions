import numpy as np

def cross_entropy_loss(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=float)
    row_indices = np.arange(len(y_true))
    correct_probabilities = y_pred[row_indices, y_true]
    return float(-np.mean(np.log(correct_probabilities)))
