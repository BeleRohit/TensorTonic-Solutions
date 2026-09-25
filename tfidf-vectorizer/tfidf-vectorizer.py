import math
from collections import Counter
import numpy as np

def tfidf_vectorizer(documents: list[str]) -> dict:
    tokenized = [doc.lower().split() for doc in documents]
    vocab = sorted(set(tok for doc in tokenized for tok in doc))
    N = len(documents)
    V = len(vocab)
    vocab_index = {term: i for i, term in enumerate(vocab)}
    df = np.zeros(V, dtype=float)
    for doc in tokenized:
        for term in set(doc):
            df[vocab_index[term]] += 1
    idf = np.log(N / df)
    tfidf_matrix = np.zeros((N, V), dtype=float)
    for i, doc in enumerate(tokenized):
        counts = Counter(doc)
        doc_len = len(doc)
        for term, cnt in counts.items():
            j = vocab_index[term]
            tfidf_matrix[i, j] = (cnt / doc_len) * idf[j]
    return {"tfidf_matrix": tfidf_matrix, "vocabulary": vocab}
