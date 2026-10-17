"""Memory-bounded pairwise attribute distances, unchanged entrywise formula."""
import numpy as np

def mean_absolute_attribute_distance(left, right, chunk_size=64):
    left, right = np.asarray(left), np.asarray(right)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("Attribute matrices must have compatible columns")
    result = np.empty((len(left), len(right)), dtype=float)
    for start in range(0, len(left), chunk_size):
        stop = min(start+chunk_size, len(left))
        # Retain the original mean over attribute columns for each pair; only
        # the outer row dimension is chunked. This avoids an n*n*p temporary.
        result[start:stop] = np.mean(np.abs(left[start:stop, None, :] - right[None, :, :]), axis=2)
    return result
