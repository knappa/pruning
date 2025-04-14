from typing import Tuple

import numba
import numpy as np

####################################################################################################


@numba.njit
def dna_disagreement(seq1: np.ndarray, seq2: np.ndarray) -> float:
    disagreement = 0.0
    assert seq1.shape == seq2.shape
    for idx in range(len(seq1)):
        if seq1[idx] < 4 and seq2[idx] < 4:
            if seq1[idx] != seq2[idx]:
                disagreement += 1.0
        else:
            disagreement += 0.75
    return disagreement / len(seq1)


def dna_sequence_distance(
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
    beta = 1 / (1 - np.sum(pis**2))
    disagreement = dna_disagreement(seq1, seq2)
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
def diploid_dna_disagreement(seq1: np.ndarray, seq2: np.ndarray) -> float:
    disagreement = 0.0
    assert seq1.shape == seq2.shape
    for idx in range(len(seq1)):
        seq1_a, seq1_b = np.divmod(seq1[idx], 5)
        seq2_a, seq2_b = np.divmod(seq2[idx], 5)

        if seq1_a < 4 and seq2_a < 4:
            if seq1_a != seq2_a:
                disagreement += 1.0
        else:
            disagreement += 0.75

        if seq1_b < 4 and seq2_b < 4:
            if seq1_b != seq2_b:
                disagreement += 1.0
        else:
            disagreement += 0.75
    return disagreement / (2 * len(seq1))


def diploid_dna_sequence_distance(
    seq1: np.ndarray, seq2: np.ndarray, *, pis: np.ndarray
) -> Tuple[float, float]:
    """
    F81 distance, a generalization of the JC69 distance, which takes into account nucleotide frequencies
    :param seq1: a diploid sequence
    :param seq2: a diploid sequence
    :param pis: ACGT frequencies
    :return: F81 distance, variance of distance * sequence length
    """
    #
    beta = 1 / (1 - np.sum(pis**2))
    disagreement = diploid_dna_disagreement(seq1, seq2)
    return (
        -np.log(np.maximum(1e-10, 1 - beta * disagreement)) / beta,
        np.clip(
            np.nan_to_num((1 - disagreement) * disagreement / (beta * disagreement - 1) ** 2),
            1,
            1_000,
        ),
    )


####################################################################################################


@numba.njit
def phased_dna_disagreement(seq1: np.ndarray, seq2: np.ndarray) -> float:
    disagreement = 0.0
    assert seq1.shape == seq2.shape
    for idx in range(len(seq1)):
        seq1_nuc_pair = int(seq1[idx])
        seq2_nuc_pair = int(seq2[idx])
        if seq1_nuc_pair < 10 and seq2_nuc_pair < 10:
            if seq1_nuc_pair != seq2_nuc_pair:
                disagreement += 1.0
        elif seq1_nuc_pair == 10:
            if seq2_nuc_pair < 4:
                # match with AA, CC, GG, TT
                disagreement += 0.9375  # 15/16
            elif seq2_nuc_pair < 10:
                # match with AC, AG, AT, CG, ...
                disagreement += 0.875  # 14/16 = 7/8
            elif seq2_nuc_pair == 10:
                # match between ?? and ??
                disagreement += 0.890625  # 1/16 * (15/16) * 4 + 2/16 * (7/8) * 6
            else:
                # match between ?? and A?, C?, G?, T?
                disagreement += 0.890625  # (1/4) * (15/16) + (3/4) * (7/8)
        elif seq2_nuc_pair == 10:
            if seq1_nuc_pair < 4:
                # match with AA, CC, GG, TT
                disagreement += 0.9375  # 15/16
            elif seq1_nuc_pair < 10:
                # match with AC, AG, AT, CG, ...
                disagreement += 0.875  # 14/16 = 7/8
            elif seq1_nuc_pair == 10:
                # match between ?? and ??
                disagreement += 0.890625  # 1/16 * (15/16) * 4 + 2/16 * (7/8) * 6
            else:
                # match between ?? and A?, C?, G?, T?
                disagreement += 0.890625  # (1/4) * (15/16) + (3/4) * (7/8)
        elif seq1_nuc_pair > 10:
            if seq1_nuc_pair == 11:  # A?
                if (
                    seq2_nuc_pair == 0
                    or seq2_nuc_pair == 4
                    or seq2_nuc_pair == 5
                    or seq2_nuc_pair == 6
                ):
                    # AA, AC, AG, AT
                    disagreement += 0.75  # 3/4
                elif (
                    seq2_nuc_pair == 1
                    or seq2_nuc_pair == 2
                    or seq2_nuc_pair == 3
                    or seq2_nuc_pair == 7
                    or seq2_nuc_pair == 8
                    or seq2_nuc_pair == 9
                ):
                    # CC, GG, TT, CG, CT, GT
                    disagreement += 1.0
                elif seq2_nuc_pair == 10:
                    disagreement += 0.890625  # (1/4) * (15/16) + (3/4) * (7/8)
                elif seq2_nuc_pair == 11:
                    # A?
                    disagreement += 0.75  # 3/4
                else:
                    # seq2_nuc_pair > 11: C? G? T?
                    disagreement += 0.9375  # 1-(1/4)**2
            elif seq1_nuc_pair == 12:  # C?
                if (
                    seq2_nuc_pair == 1
                    or seq2_nuc_pair == 4
                    or seq2_nuc_pair == 7
                    or seq2_nuc_pair == 8
                ):
                    # CC, AC, CG, CT
                    disagreement += 0.75  # 3/4
                elif (
                    seq2_nuc_pair == 0
                    or seq2_nuc_pair == 2
                    or seq2_nuc_pair == 3
                    or seq2_nuc_pair == 5
                    or seq2_nuc_pair == 6
                    or seq2_nuc_pair == 9
                ):
                    # AA, GG, TT, AG, AT, GT
                    disagreement += 1.0
                elif seq2_nuc_pair == 10:
                    disagreement += 0.890625  # (1/4) * (15/16) + (3/4) * (7/8)
                elif seq2_nuc_pair == 12:
                    # C?
                    disagreement += 0.75  # 3/4
                else:
                    # A? G? T?
                    disagreement += 0.9375  # 1-(1/4)**2
            elif seq1_nuc_pair == 13:  # G?
                if (
                    seq2_nuc_pair == 2
                    or seq2_nuc_pair == 5
                    or seq2_nuc_pair == 7
                    or seq2_nuc_pair == 9
                ):
                    # GG, AG, CG, GT
                    disagreement += 0.75  # 3/4
                elif (
                    seq2_nuc_pair == 0
                    or seq2_nuc_pair == 1
                    or seq2_nuc_pair == 3
                    or seq2_nuc_pair == 4
                    or seq2_nuc_pair == 6
                    or seq2_nuc_pair == 8
                ):
                    # AA, CC, TT, AC, AT, CT
                    disagreement += 1.0
                elif seq2_nuc_pair == 10:
                    disagreement += 0.890625  # (1/4) * (15/16) + (3/4) * (7/8)
                elif seq2_nuc_pair == 13:
                    # G?
                    disagreement += 0.75  # 3/4
                else:
                    # A? C? T?
                    disagreement += 0.9375  # 1-(1/4)**2
            else:
                # seq1_nuc_pair == 14:  # T?
                if (
                    seq2_nuc_pair == 3
                    or seq2_nuc_pair == 6
                    or seq2_nuc_pair == 8
                    or seq2_nuc_pair == 9
                ):
                    # TT, AT, CT, GT
                    disagreement += 0.75  # 3/4
                elif (
                    seq2_nuc_pair == 0
                    or seq2_nuc_pair == 1
                    or seq2_nuc_pair == 2
                    or seq2_nuc_pair == 4
                    or seq2_nuc_pair == 5
                    or seq2_nuc_pair == 7
                ):
                    # AA, CC, GG, AC, AG, CG
                    disagreement += 1.0
                elif seq2_nuc_pair == 10:
                    disagreement += 0.890625  # (1/4) * (15/16) + (3/4) * (7/8)
                elif seq2_nuc_pair == 14:
                    # T?
                    disagreement += 0.75  # 3/4
                else:
                    # A? C? G?
                    disagreement += 0.9375  # 1-(1/4)**
        else:  # seq2_nuc_pair > 10:
            if seq2_nuc_pair == 11:  # A?
                if (
                    seq1_nuc_pair == 0
                    or seq1_nuc_pair == 4
                    or seq1_nuc_pair == 5
                    or seq1_nuc_pair == 6
                ):
                    # AA, AC, AG, AT
                    disagreement += 0.75  # 3/4
                elif (
                    seq1_nuc_pair == 1
                    or seq1_nuc_pair == 2
                    or seq1_nuc_pair == 3
                    or seq1_nuc_pair == 7
                    or seq1_nuc_pair == 8
                    or seq1_nuc_pair == 9
                ):
                    # CC, GG, TT, CG, CT, GT
                    disagreement += 1.0
                elif seq1_nuc_pair == 10:
                    disagreement += 0.890625  # (1/4) * (15/16) + (3/4) * (7/8)
                elif seq1_nuc_pair == 11:
                    # A?
                    disagreement += 0.75  # 3/4
                else:
                    # seq1_nuc_pair > 11: C? G? T?
                    disagreement += 0.9375  # 1-(1/4)**2
            elif seq2_nuc_pair == 12:  # C?
                if (
                    seq1_nuc_pair == 1
                    or seq1_nuc_pair == 4
                    or seq1_nuc_pair == 7
                    or seq1_nuc_pair == 8
                ):
                    # CC, AC, CG, CT
                    disagreement += 0.75  # 3/4
                elif (
                    seq1_nuc_pair == 0
                    or seq1_nuc_pair == 2
                    or seq1_nuc_pair == 3
                    or seq1_nuc_pair == 5
                    or seq1_nuc_pair == 6
                    or seq1_nuc_pair == 9
                ):
                    # AA, GG, TT, AG, AT, GT
                    disagreement += 1.0
                elif seq1_nuc_pair == 10:
                    disagreement += 0.890625  # (1/4) * (15/16) + (3/4) * (7/8)
                elif seq1_nuc_pair == 12:
                    # C?
                    disagreement += 0.75  # 3/4
                else:
                    # A? G? T?
                    disagreement += 0.9375  # 1-(1/4)**2
            elif seq2_nuc_pair == 13:  # G?
                if (
                    seq1_nuc_pair == 2
                    or seq1_nuc_pair == 5
                    or seq1_nuc_pair == 7
                    or seq1_nuc_pair == 9
                ):
                    # GG, AG, CG, GT
                    disagreement += 0.75  # 3/4
                elif (
                    seq1_nuc_pair == 0
                    or seq1_nuc_pair == 1
                    or seq1_nuc_pair == 3
                    or seq1_nuc_pair == 4
                    or seq1_nuc_pair == 6
                    or seq1_nuc_pair == 8
                ):
                    # AA, CC, TT, AC, AT, CT
                    disagreement += 1.0
                elif seq1_nuc_pair == 10:
                    disagreement += 0.890625  # (1/4) * (15/16) + (3/4) * (7/8)
                elif seq1_nuc_pair == 13:
                    # G?
                    disagreement += 0.75  # 3/4
                else:
                    # A? C? T?
                    disagreement += 0.9375  # 1-(1/4)**2
            else:
                # seq2_nuc_pair == 14:  # T?
                if (
                    seq1_nuc_pair == 3
                    or seq1_nuc_pair == 6
                    or seq1_nuc_pair == 8
                    or seq1_nuc_pair == 9
                ):
                    # TT, AT, CT, GT
                    disagreement += 0.75  # 3/4
                elif (
                    seq1_nuc_pair == 0
                    or seq1_nuc_pair == 1
                    or seq1_nuc_pair == 2
                    or seq1_nuc_pair == 4
                    or seq1_nuc_pair == 5
                    or seq1_nuc_pair == 7
                ):
                    # AA, CC, GG, AC, AG, CG
                    disagreement += 1.0
                elif seq1_nuc_pair == 10:
                    disagreement += 0.890625  # (1/4) * (15/16) + (3/4) * (7/8)
                elif seq1_nuc_pair == 14:
                    # T?
                    disagreement += 0.75  # 3/4
                else:
                    # A? C? G?
                    disagreement += 0.9375  # 1-(1/4)**

    return disagreement / len(seq1)


def phased_sequence_distance(
    seq1: np.ndarray, seq2: np.ndarray, *, pis: np.ndarray
) -> Tuple[float, float]:
    """
    F81-16 distance, a generalization of the JC69 distance, which takes into account nucleotide frequencies
    :param seq1: a sequence
    :param seq2: a sequence
    :param pis: ACGT frequencies
    :return: F81-16 distance, variance of distance * sequence length
    """
    #
    beta = 1 / (2 * (1 - np.sum(pis**2)))
    disagreement = phased_dna_disagreement(seq1, seq2)
    return (
        -np.log(
            np.maximum(1e-10, 1 - 2 * beta + 2 * beta * np.sqrt(1 - disagreement)),
        )
        / beta,
        np.clip(
            np.nan_to_num(
                disagreement / (2 * beta * np.sqrt(1 - disagreement) - 2 * beta + 1) ** 2
            ),
            1,
            1_000,
        ),
    )


####################################################################################################


@numba.njit
def unphased_dna_disagreement(seq1: np.ndarray, seq2: np.ndarray) -> float:
    disagreement = 0.0
    assert seq1.shape == seq2.shape
    for idx in range(len(seq1)):
        seq1_nuc_m, seq1_nuc_p = divmod(int(seq1[idx]), 5)
        seq2_nuc_m, seq2_nuc_p = divmod(int(seq2[idx]), 5)
        if seq1_nuc_m < 4 and seq2_nuc_m < 4:
            # no maternal ambiguity
            if seq1_nuc_m != seq2_nuc_m:
                disagreement += 1.0
            else:
                if seq1_nuc_p < 4 and seq2_nuc_p < 4:
                    # no maternal ambiguity & matches, no paternal ambiguity
                    if seq1_nuc_p != seq2_nuc_p:
                        disagreement += 1.0
                else:
                    # no maternal ambiguity & matches, paternal ambiguity
                    disagreement += 0.75
        else:
            # maternal ambiguity
            if seq1_nuc_p < 4 and seq2_nuc_p < 4:
                # maternal ambiguity, no paternal ambiguity
                if seq1_nuc_p != seq2_nuc_p:
                    # maternal ambiguity, no paternal ambiguity, does not match
                    disagreement += 1.0
                else:
                    # maternal ambiguity, no paternal ambiguity, does match
                    disagreement += 0.75
            else:
                # maternal ambiguity, paternal ambiguity
                disagreement += 0.9375  # 15/16, as both must match
    return disagreement / len(seq1)


def unphased_sequence_distance(
    seq1: np.ndarray, seq2: np.ndarray, *, pis: np.ndarray
) -> Tuple[float, float]:
    """
    F81-10 distance, a generalization of the JC69 distance, which takes into account nucleotide frequencies
    :param seq1: a sequence
    :param seq2: a sequence
    :param pis: ACGT frequencies
    :return: F81-10 distance
    """
    #
    beta = 1 / (2 * (1 - np.sum(pis**2)))
    zeta = np.sum([np.prod(pis[np.arange(4) != idx]) for idx in range(4)])
    eta = np.prod(pis)
    disagreement = unphased_dna_disagreement(seq1, seq2)
    return (
        -np.log(
            np.maximum(
                1e-10,
                (
                    32 * beta**2 * (eta - zeta)
                    - 4 * beta
                    - 3
                    + beta
                    * np.sqrt(8)
                    * np.sqrt(2 + disagreement * (32 * beta**2 * (zeta - eta) + 3))
                )
                / (32 * beta**2 * (eta - zeta) - 8 * beta + 3 - 8 * beta**2 * disagreement),
            )
        ),
        np.clip(
            np.nan_to_num(
                -((32 * beta**2 * (eta - zeta) - 8 * beta**2 * disagreement - 8 * beta + 3) ** 2)
                * (
                    np.sqrt(2)
                    * (32 * beta**2 * (eta - zeta) - 3)
                    * beta
                    / (
                        (32 * beta**2 * (eta - zeta) - 8 * beta**2 * disagreement - 8 * beta + 3)
                        * np.sqrt(-(32 * beta**2 * (eta - zeta) - 3) * disagreement + 2)
                    )
                    - 8
                    * (
                        32 * beta**2 * (eta - zeta)
                        + 2
                        * np.sqrt(2)
                        * np.sqrt(-(32 * beta**2 * (eta - zeta) - 3) * disagreement + 2)
                        * beta
                        - 4 * beta
                        - 3
                    )
                    * beta**2
                    / (32 * beta**2 * (eta - zeta) - 8 * beta**2 * disagreement - 8 * beta + 3) ** 2
                )
                ** 2
                * (disagreement - 1)
                * disagreement
                / (
                    32 * beta**2 * (eta - zeta)
                    + 2
                    * np.sqrt(2)
                    * np.sqrt(-(32 * beta**2 * (eta - zeta) - 3) * disagreement + 2)
                    * beta
                    - 4 * beta
                    - 3
                )
                ** 2
            ),
            1,
            1_000,
        ),
    )


####################################################################################################
