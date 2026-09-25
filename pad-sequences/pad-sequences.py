import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    N = len(seqs)
    if N == 0:
        return np.zeros((0, 0), dtype=int)
    L = max_len if max_len is not None else max((len(s) for s in seqs), default=0)
    out = np.full((N, L), pad_value, dtype=int)
    for i, seq in enumerate(seqs):
        trunc = list(seq)[:L]
        if len(trunc) > 0:
            out[i, :len(trunc)] = trunc
    return out
