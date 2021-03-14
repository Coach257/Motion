import numpy as np


def vector_norm(vec):
    return vec / np.sqrt(np.sum(vec ** 2, axis=-1, keepdims=True))


def vector_mod(vec, keepdims=False):
    return np.sqrt(np.sum(vec ** 2, axis=-1, keepdims=keepdims))