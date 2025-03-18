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


def np_full_print(nparray):
    from shutil import get_terminal_size

    from numpy import inf, printoptions

    # noinspection PyTypeChecker
    with printoptions(
        threshold=inf,
        linewidth=get_terminal_size((80, 20)).columns,
        suppress=True,
    ):
        print(nparray)


class CallbackParam:
    num_func_evals: int = 0

    def __call__(self, x):
        self.num_func_evals += 1
        print(self.num_func_evals, flush=True)
        np_full_print(x)


class CallbackIR:
    num_func_evals: int = 0

    def __call__(self, intermediate_result):
        self.num_func_evals += 1
        print(self.num_func_evals, flush=True)
        print(intermediate_result, flush=True)
        np_full_print(intermediate_result.x)


def print_stats(*, s_est, pis_est, res, tree_distances, true_branch_lens):
    import numpy as np

    from pruning.matrices import make_A_GTR

    print(f"neg log likelihood: {res.fun}")
    print()

    for s, v in zip(["s_ac", "s_ag", "s_at", "s_cg", "s_ct", "s_gt"], s_est):
        print(f"{s}: {v}")
    print()

    for p, v in zip(["pi_a", "pi_c", "pi_g", "pi_t"], pis_est):
        print(f"{p}: {v}")
    print()

    # TODO: per model code
    print("Q:")
    A_GTR = make_A_GTR(pis_est)
    q_est = (A_GTR @ s_est).reshape(4, 4)
    for row in q_est:
        print(" [", end="")
        for val in row:
            print(f" {val:8.5f}", end="")
        print(" ]")
    print()

    print("tree dist stats:")
    print(f"min internal branch dist: {np.min(tree_distances[1:])}")
    print(f"max internal branch dist: {np.max(tree_distances[1:])}")
    print(f"mean internal branch dist: {np.mean(tree_distances[1:])}")
    print(f"stdev internal branch dist: {np.std(tree_distances[1:])}")
    print()

    print("tree dist error stats:")
    abs_error = np.abs(tree_distances[1:] - true_branch_lens[1:])
    print(f"min abs error: {np.min(abs_error)}")
    print(f"max abs error: {np.max(abs_error)}")
    print(f"mean abs error: {np.mean(abs_error)}")
    print(f"stdev abs error: {np.std(abs_error)}")
    rel_mask = (tree_distances > 0) & (true_branch_lens > 0)
    rel_error = (tree_distances[rel_mask] - true_branch_lens[rel_mask]) / true_branch_lens[rel_mask]
    print(f"min rel error: {np.min(rel_error) if len(rel_error) > 0 else float('nan')}")
    print(f"max rel error: {np.max(rel_error) if len(rel_error) > 0 else float('nan')}")
    print(f"mean rel error: {np.mean(rel_error) if len(rel_error) > 0 else float('nan')}")
    print(f"stdev rel error: {np.std(rel_error) if len(rel_error) > 0 else float('nan')}")
