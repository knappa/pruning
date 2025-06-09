from collections import defaultdict

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
        t = accumulator + y
        compensator = (t - accumulator) - y
        accumulator = t
    return accumulator


def np_full_print(nparray, flush=False):
    from shutil import get_terminal_size

    from numpy import inf, printoptions

    # noinspection PyTypeChecker
    with printoptions(
        threshold=inf,
        linewidth=get_terminal_size((80, 20)).columns,
        suppress=True,
    ):
        print(nparray, flush=flush)


class CallbackParam:
    num_func_evals: int = 0

    def __init__(self, print_period: int = 1):
        self.print_period = print_period

    def __call__(self, x):
        self.num_func_evals += 1
        if self.num_func_evals % self.print_period == 0:
            print(self.num_func_evals)
            np_full_print(x, flush=True)


# class CallbackIR:
#     num_func_evals: int = 0
#
#     def __call__(self, intermediate_result):
#         self.num_func_evals += 1
#         print(self.num_func_evals, flush=True)
#         print(intermediate_result, flush=True)
#         np_full_print(intermediate_result.x)


def print_dna_params(s_est, pis_est):
    from pruning.matrices import make_A_GTR

    for s, v in zip(["s_ac", "s_ag", "s_at", "s_cg", "s_ct", "s_gt"], s_est):
        print(f"{s}: {v}")
    print()

    for p, v in zip(["pi_a", "pi_c", "pi_g", "pi_t"], pis_est):
        print(f"{p}: {v}")
    print()

    print("Q:")
    A_GTR = make_A_GTR(pis_est)
    q_est = (A_GTR @ s_est).reshape(4, 4)
    for row in q_est:
        print(" [", end="")
        for val in row:
            print(f" {val:8.5f}", end="")
        print(" ]")
    print()


def print_gtr10_params(s_est, pis_est):
    from pruning.matrices import Qsym_gtr10

    for s, v in zip(
        [
            # fmt: off
            # @formatter:off
            "s_{aa|ac}", "s_{aa|ag}", "s_{aa|at}", "s_{aa|cc}", "s_{aa|cg}", "s_{aa|ct}", "s_{aa|gg}", "s_{aa|gt}",
            "s_{aa|tt}", "s_{ac|ag}", "s_{ac|at}", "s_{ac|cc}", "s_{ac|cg}", "s_{ac|ct}", "s_{ac|gg}", "s_{ac|gt}",
            "s_{ac|tt}", "s_{ag|at}", "s_{ag|cc}", "s_{ag|cg}", "s_{ag|ct}", "s_{ag|gg}", "s_{ag|gt}", "s_{ag|tt}",
            "s_{at|cc}", "s_{at|cg}", "s_{at|ct}", "s_{at|gg}", "s_{at|gt}", "s_{at|tt}", "s_{cc|cg}", "s_{cc|ct}",
            "s_{cc|gg}", "s_{cc|gt}", "s_{cc|tt}", "s_{cg|ct}", "s_{cg|gg}", "s_{cg|gt}", "s_{cg|tt}", "s_{ct|gg}",
            "s_{ct|gt}", "s_{ct|tt}", "s_{gg|gt}", "s_{gg|tt}", "s_{gt|tt}",
            # @formatter:on
            # fmt: on
        ],
        s_est,
    ):
        print(f"{s}: {v}")
    print()

    for p, v in zip(
        [
            # fmt: off
            # @formatter:off
            "pi_{aa}", "pi_{ac}", "pi_{ag}", "pi_{at}", "pi_{cc}",
            "pi_{cg}", "pi_{ct}", "pi_{gg}", "pi_{gt}", "pi_{tt}",
            # @formatter:on
            # fmt: on
        ],
        pis_est,
    ):
        print(f"{p}: {v}")
    print()

    q_sym_est = Qsym_gtr10(pis_est, s_est)
    # Q_gtr10 = np.diag([np.sqrt(x) for x in pi10s]) @ Qsym_gtr10 @ np.diag([1/np.sqrt(x) for x in pi10s])
    q_est = (
        np.diag([np.sqrt(x) for x in pis_est])
        @ q_sym_est
        @ np.diag([1 / np.sqrt(x) for x in pis_est])
    )
    print("Q:")
    for row in q_est:
        print(" [", end="")
        for val in row:
            print(f" {val:8.5f}", end="")
        print(" ]")
    print()


def print_gtr10z_params(s_est, pis_est):
    from pruning.matrices import Qsym_gtr10z

    for s, v in zip(
        [
            # fmt: off
            # @formatter:off
            "s_{aa|ac}", "s_{aa|ag}", "s_{aa|at}", "s_{ac|ag}", "s_{ac|at}", "s_{ac|cc}", "s_{ac|cg}", "s_{ac|ct}",
            "s_{ag|at}", "s_{ag|cg}", "s_{ag|gg}", "s_{ag|gt}", "s_{at|ct}", "s_{at|gt}", "s_{at|tt}", "s_{cc|cg}",
            "s_{cc|ct}", "s_{cg|ct}", "s_{cg|gg}", "s_{cg|gt}", "s_{ct|gt}", "s_{ct|tt}", "s_{gg|gt}", "s_{gt|tt}",
            # @formatter:on
            # fmt: on
        ],
        s_est,
    ):
        print(f"{s}: {v}")
    print()

    for p, v in zip(
        [
            # fmt: off
            # @formatter:off
            "pi_{aa}", "pi_{ac}", "pi_{ag}", "pi_{at}", "pi_{cc}",
            "pi_{cg}", "pi_{ct}", "pi_{gg}", "pi_{gt}", "pi_{tt}",
            # @formatter:on
            # fmt: on
        ],
        pis_est,
    ):
        print(f"{p}: {v}")
    print()

    q_sym_est = Qsym_gtr10z(pis_est, s_est)
    # Q_gtr10z = np.diag([1/np.sqrt(x) for x in pi10s]) @ Qsym_gtr10z @ np.diag([np.sqrt(x) for x in pi10s])
    q_est = (
        np.diag([1 / np.sqrt(x) for x in pis_est])
        @ q_sym_est
        @ np.diag([np.sqrt(x) for x in pis_est])
    )
    print("Q:")
    for row in q_est:
        print(" [", end="")
        for val in row:
            print(f" {val:8.5f}", end="")
        print(" ]")
    print()


def print_cellphy10_params(s_est, pis_est):
    from pruning.matrices import Qsym_cellphy10

    for s, v in zip(
        ["alpha", "beta", "gamma", "kappa", "lambda", "mu"],  # notation from supplementary note 2
        s_est,
    ):
        print(f"{s}: {v}")
    print()

    for p, v in zip(
        [
            # fmt: off
            # @formatter:off
            "pi_{aa}", "pi_{ac}", "pi_{ag}", "pi_{at}", "pi_{cc}",
            "pi_{cg}", "pi_{ct}", "pi_{gg}", "pi_{gt}", "pi_{tt}",
            # @formatter:on
            # fmt: on
        ],
        pis_est,
    ):
        print(f"{p}: {v}")
    print()

    q_sym_est = Qsym_cellphy10(pis_est, s_est)
    # Q_cellphy10 = np.diag([np.sqrt(x) for x in pi10s]) @ Qsym_cellphy10 @ np.diag([1/np.sqrt(x) for x in pi10s])
    q_est = (
        np.diag([np.sqrt(x) for x in pis_est])
        @ q_sym_est
        @ np.diag([1 / np.sqrt(x) for x in pis_est])
    )
    print("Q:")
    for row in q_est:
        print(" [", end="")
        for val in row:
            print(f" {val:8.5f}", end="")
        print(" ]")
    print()


def print_stats(*, rate_params, freq_params, neg_l, tree_distances, true_branch_lens, model):
    import numpy as np

    print(f"neg log likelihood: {neg_l}")
    print()

    # noinspection PyUnreachableCode
    match model:
        case "DNA" | "PHASED_DNA4":
            print_dna_params(rate_params, freq_params)
        case "CELLPHY":
            print_cellphy10_params(rate_params, freq_params)
        case "GTR10Z":
            print_gtr10z_params(rate_params, freq_params)
        case "GTR10":
            print_gtr10_params(rate_params, freq_params)
        # case "UNPHASED_DNA":
        #     pass
        # case "PHASED_DNA16":
        #     pass
        # case "PHASED_DNA16_MP":
        #     pass
        # case "SIEVE":
        #     # TODO: per model code
        #     pass
        case _:
            print(f"{rate_params=}")
            print()
            print(f"{freq_params=}")
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


def rate_param_scale(*, x, log_freq_params, ploidy, rate_constraint) -> float:
    rate_params = np.maximum(0.0, x)
    return ploidy / rate_constraint(np.exp(log_freq_params), rate_params)


def rate_param_cleanup(*, x, log_freq_params, ploidy, rate_constraint):
    rate_params = np.maximum(0.0, x)
    return rate_params * ploidy / rate_constraint(np.exp(log_freq_params), rate_params)


################################################################################
# solver options

solver_options = defaultdict(dict)
solver_options["L-BFGS-B"] = {"maxiter": 1000, "maxfun": 100_000, "ftol": 1e-8}
solver_options["L-BFGS-B-Lite"] = {"maxiter": 100, "maxfun": 10_000, "ftol": 1e-4}
solver_options["Powell"] = {"maxiter": 1000, "ftol": 1e-10}
solver_options["Nelder-Mead"] = {"adaptive": True, "fatol": 1e-8, "xatol": 1e-8}
