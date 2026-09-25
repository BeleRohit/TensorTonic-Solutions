import numpy as np

def stratified_split(X: list, y: list, test_size: float = 0.2, seed: int = 42) -> dict:
    X = np.asarray(X)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    train_idx = []
    test_idx = []
    for c in classes:
        idx = np.where(y == c)[0].copy()
        rng.shuffle(idx)
        n_c = len(idx)
        n_test = round(n_c * test_size)
        if n_c > 1:
            n_test = min(n_test, n_c - 1)
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())
    train_idx = sorted(train_idx)
    test_idx = sorted(test_idx)
    return {
        "X_train": X[train_idx],
        "X_test": X[test_idx],
        "y_train": y[train_idx],
        "y_test": y[test_idx],
    }
