import numpy as np

def impute_missing(X: list, strategy: str = "mean") -> np.ndarray:
    X = np.array(X, dtype=float)
    orig_1d = X.ndim == 1
    if orig_1d:
        X = X.reshape(-1, 1)
    out = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        mask = ~np.isnan(col)
        if mask.sum() == 0:
            fill = 0.0
        elif strategy == "mean":
            fill = np.mean(col[mask])
        else:
            fill = np.median(col[mask])
        out[~mask, j] = fill
    if orig_1d:
        out = out.reshape(-1)
    return out
