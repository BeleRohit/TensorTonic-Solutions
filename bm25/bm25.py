import math
from collections import Counter
import numpy as np

def bm25_score(query_tokens: list[str], docs: list[list[str]], k1: float = 1.2, b: float = 0.75) -> np.ndarray:
    N = len(docs)
    doc_lens = [len(d) for d in docs]
    avgdl = sum(doc_lens) / N if N > 0 else 0.0
    unique_query = list(dict.fromkeys(query_tokens))
    df = {t: sum(1 for d in docs if t in d) for t in unique_query}
    idf = {t: math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1) for t in unique_query}
    scores = np.zeros(N, dtype=float)
    for i, d in enumerate(docs):
        counts = Counter(d)
        D = doc_lens[i]
        s = 0.0
        for t in unique_query:
            tf = counts.get(t, 0)
            if tf == 0:
                continue
            denom = tf + k1 * (1 - b + b * D / avgdl)
            s += idf[t] * (tf * (k1 + 1)) / denom
        scores[i] = s
    return scores
