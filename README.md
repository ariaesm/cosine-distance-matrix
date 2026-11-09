# Cosine Similarity & Distance Matrices

A notebook that computes pairwise cosine similarity and distance matrices from scratch with NumPy, checks the result against SciPy and scikit-learn, and then uses it to find similar documents with TF-IDF.

**[Open the notebook →](cosine_distance_matrix.ipynb)** (outputs included)

## Highlights

- Vectorized implementation: normalize the rows once, then a single matrix product gives every pairwise cosine. Zero vectors are handled.
- Cross-checked against `sklearn.metrics.pairwise.cosine_similarity`, `pairwise_distances(metric="cosine")` and `scipy.spatial.distance.cdist`.
- A vector plot showing why cosine ignores magnitude: rows `[1, 5]` and `[3, 8]` have a distance of only 0.013.
- A document-similarity example with a TF-IDF heatmap.

```python
def cosine_similarity_matrix(X):
    X = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    unit = X / np.where(norms == 0, 1, norms)
    return unit @ unit.T
```

## Run

```bash
pip install numpy scipy scikit-learn matplotlib jupyter
jupyter notebook cosine_distance_matrix.ipynb
```

## License

[MIT](LICENSE)
