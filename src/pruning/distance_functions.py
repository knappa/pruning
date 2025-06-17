from typing import Callable, Tuple, Union

import numba
import numpy as np

####################################################################################################


@numba.njit
def haploid_dna_disagreement(seq1: np.ndarray, seq2: np.ndarray) -> float:
    disagreement = 0.0
    assert seq1.shape == seq2.shape
    for idx in range(len(seq1)):
        if seq1[idx] < 4 and seq2[idx] < 4:
            if seq1[idx] != seq2[idx]:
                disagreement += 1.0
        else:
            disagreement += 0.75
    return disagreement / len(seq1)


def haploid_dna_sequence_distance(
    seq1: np.ndarray, seq2: np.ndarray, *, pis: np.ndarray
) -> Tuple[float, float]:
    """
    F81 distance, a generalization of the JC69 distance, which takes into account nucleotide frequencies
    :param seq1: a sequence
    :param seq2: a sequence
    :param pis: ACGT frequencies
    :return: F81 distance, variance of distance * sequence length
    """
    #
    beta = 1 / (1 - float(np.sum(pis**2)))
    disagreement = haploid_dna_disagreement(seq1, seq2)
    return (
        -np.log(np.maximum(1e-10, 1 - beta * disagreement)) / beta,
        # np.maximum(1e-10, (1 - disagreement) * disagreement)
        # / np.maximum(1e-10, (beta * disagreement - 1) ** 2),
        np.clip(
            np.nan_to_num((1 - disagreement) * disagreement / (beta * disagreement - 1) ** 2),
            1,
            1_000,
        ),
    )


####################################################################################################


@numba.njit
def sequence_disagreement(seq1: np.ndarray, seq2: np.ndarray) -> float:
    disagreement = 0.0
    assert seq1.shape == seq2.shape
    for nuc_pair_a, nuc_pair_b in zip(seq1, seq2):
        if nuc_pair_a != nuc_pair_b:
            disagreement += 1.0

    return disagreement / len(seq1)


def sequence_distance(
    seq1: np.ndarray,
    seq2: np.ndarray,
    *,
    pis: np.ndarray,
    Q_sym_func: Callable,
    num_rate_params: int,
    ploidy: int,
) -> Tuple[float, float]:
    """
    Unphased diploid version of the F81 distance
    :param seq1: a sequence
    :param seq2: a sequence
    :param pis: base frequencies
    :param Q_sym_func: function which generates the symmetrized Q matrix (NOT the S matrix)
    :param num_rate_params: number of rate parameters as used in Q_sym_func
    :param ploidy: 1 for haploid, 2 for diploid
    :return: distance, variance of distance
    """
    from scipy.differentiate import derivative
    from scipy.linalg import expm
    from scipy.optimize import brentq

    sym_Q = Q_sym_func(pis, np.ones(num_rate_params, dtype=np.float64))
    Q = sym_Q / np.sqrt(pis)[:, None] * np.sqrt(pis)
    mu = -(pis @ np.diag(Q))
    beta = 1 / mu
    disagreement = sequence_disagreement(seq1, seq2)

    def expected_mutations(nu: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if isinstance(nu, np.ndarray) and len(nu.shape) > 0:
            return np.array([expected_mutations(nu_part) for nu_part in nu])
        P_nu = expm(Q * beta * ploidy * nu)
        # noinspection PyTypeChecker
        return 1 - (pis @ np.diag(P_nu))

    # at nu=0, the proportion of expected mutations is zero. Increment lower bound for nu by 1 until nu+1 surpasses
    # the measured proportion of mutations.
    lower_bound = 0.0
    while expected_mutations(lower_bound + 1.0) < disagreement:
        lower_bound += 1.0

    # at nu=lower_bound, the proportion of expected mutations is lower than the measured proportion. Increase nu
    # by 1.0 until the proportion of expected mutations is strictly greater than the measured proportion.
    upper_bound = lower_bound + 1.0
    while expected_mutations(upper_bound) <= disagreement:
        upper_bound += 1.0

    branch_length, convergence_result = brentq(
        lambda nu: disagreement - expected_mutations(nu), lower_bound, upper_bound, full_output=True
    )

    deriv_res = derivative(expected_mutations, branch_length)
    deriv = deriv_res.df

    return branch_length, disagreement * (1 - disagreement) / float(deriv) ** 2


####################################################################################################


def haploid_sequence_distance(seq1: np.ndarray, seq2: np.ndarray, *, pis: np.ndarray):
    from pruning.matrices import Qsym_gtr4

    return sequence_distance(seq1, seq2, pis=pis, Q_sym_func=Qsym_gtr4, ploidy=1, num_rate_params=6)


def phased_diploid_sequence_distance(seq1: np.ndarray, seq2: np.ndarray, *, pis: np.ndarray):
    from pruning.matrices import Qsym_GTRsq

    return sequence_distance(
        seq1, seq2, pis=pis, Q_sym_func=Qsym_GTRsq, ploidy=2, num_rate_params=6
    )


def unphased_diploid_sequence_distance(seq1: np.ndarray, seq2: np.ndarray, *, pis: np.ndarray):
    from pruning.matrices import Qsym_unphased

    return sequence_distance(
        seq1, seq2, pis=pis, Q_sym_func=Qsym_unphased, ploidy=2, num_rate_params=6
    )


def cellphy_unphased_diploid_sequence_distance(
    seq1: np.ndarray, seq2: np.ndarray, *, pis: np.ndarray
):
    from pruning.matrices import Qsym_cellphy10

    return sequence_distance(
        seq1, seq2, pis=pis, Q_sym_func=Qsym_cellphy10, ploidy=2, num_rate_params=6
    )


def gtr10z_sequence_distance(seq1: np.ndarray, seq2: np.ndarray, *, pis: np.ndarray):
    from pruning.matrices import Qsym_gtr10z

    return sequence_distance(
        seq1, seq2, pis=pis, Q_sym_func=Qsym_gtr10z, ploidy=2, num_rate_params=24
    )


def gtr10_sequence_distance(seq1: np.ndarray, seq2: np.ndarray, *, pis: np.ndarray):
    from pruning.matrices import Qsym_gtr10

    return sequence_distance(
        seq1, seq2, pis=pis, Q_sym_func=Qsym_gtr10, ploidy=2, num_rate_params=45
    )


####################################################################################################
