import numpy as np
from scipy.special import logsumexp


def log_matrix_mult(v, P) -> np.ndarray:
    # noinspection PyTypeChecker
    return logsumexp(np.expand_dims(v, axis=-1) + P, axis=-2, return_sign=False, keepdims=False)


def log_dot(v, w) -> np.ndarray:
    # noinspection PyTypeChecker
    return logsumexp(v + w, axis=-1, return_sign=False, keepdims=False)


def kahan_dot(v: np.ndarray, w: np.ndarray):
    accumulator = np.float64(0.0)
    compensator = np.float64(0.0)
    for idx in range(v.shape[-1]):
        y = np.squeeze(v[idx] * w[idx]) - compensator
        # print(f"{y=}")
        t = accumulator + y
        # print(f"{t=}")
        compensator = (t - accumulator) - y
        # print(f"{compensator=}")
        accumulator = t
    return accumulator
