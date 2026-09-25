import numpy as np

def info_nce_loss(Z1: list, Z2: list, temperature: float = 0.1) -> float:
    Z1 = np.asarray(Z1, dtype=float)
    Z2 = np.asarray(Z2, dtype=float)
    S = (Z1 @ Z2.T) / temperature
    row_max = np.max(S, axis=1, keepdims=True)
    S_stable = S - row_max
    log_sum_exp = np.log(np.sum(np.exp(S_stable), axis=1))
    diag = np.diag(S_stable)
    loss = -(diag - log_sum_exp)
    return float(np.mean(loss))
