#!/usr/bin/env python3
import argparse
import itertools
import os.path
import sys
from collections import defaultdict
from contextlib import redirect_stdout
from functools import reduce
from operator import mul
from typing import Callable, List, Tuple

import numba
import numpy as np
import scipy
from ete3 import Tree
from scipy.optimize import OptimizeResult, minimize
from scipy.special import logsumexp, xlogy

from matrices import V, make_A_GTR, make_A_GTR16v, make_A_GTR_unph, make_rate_constraint_matrix

parser = argparse.ArgumentParser(description="Compute log likelihood using the pruning algorithm")
parser.add_argument("--seqs", type=str, required=True, help="sequence alignments in phylip format")
parser.add_argument(
    "--tree",
    type=str,
    required=True,
    help="true tree in newick format",
)
parser.add_argument(
    "--model",
    type=str,
    default="DNA",
    help="Datatype for sequence",
    choices=["DNA", "PHASED_DNA", "UNPHASED_DNA"],
)
parser.add_argument(
    "--method",
    type=str,
    default="L-BFGS-B",
    help="scipy solver",
    choices=[
        "Nelder-Mead",
        "L-BFGS-B",
        "TNC",
        "SLSQP",
        "Powell",
        "trust-constr",
        "COBYLA",
        "COBYQA",
    ],
)
parser.add_argument("--output", type=str, help="output filename prefix for tree")
parser.add_argument("--overwrite", action="store_true", help="overwrite outputs, if they exist")
parser.add_argument("--log", action="store_true")
parser.add_argument("--pre_estimate_params", action="store_true")

if hasattr(sys, "ps1"):
    # opt = parser.parse_args("--seqs test_100K.phy "
    #                         "--tree test.nwk "
    #                         "--model DNA "
    #                         "--log ".split())
    opt = parser.parse_args(
        "--seqs test-diploid.phy " "--tree test-diploid.nwk " "--model PHASED_DNA " "--log ".split()
    )
    # opt = parser.parse_args(
    #     "--seqs diploid-000.phy "
    #     "--tree tree-000.nwk "
    #     "--output reconstructed-tree-000 "
    #     "--model PHASED_DNA "
    #     "--method L-BFGS-B "
    #     "--log ".split()
    # )
else:
    opt = parser.parse_args()

if opt.log:
    print(opt)

if (
    not opt.overwrite
    and opt.output is not None
    and (os.path.isfile(opt.output + ".nwk") or os.path.isfile(opt.output + ".log"))
):
    print("output files from previous run present, exiting.")
    exit()


################################################################################
# utility functions


def np_full_print(nparray):
    import shutil

    # noinspection PyTypeChecker
    with np.printoptions(
        threshold=np.inf,
        linewidth=shutil.get_terminal_size((80, 20)).columns,
        suppress=True,
    ):
        print(nparray)


num_func_evals = 0


def callback_param(x):
    global num_func_evals
    num_func_evals += 1
    print(num_func_evals, flush=True)
    np_full_print(x)


def callback_ir(intermediate_result: OptimizeResult):
    global num_func_evals
    num_func_evals += 1
    print(num_func_evals, flush=True)
    print(intermediate_result, flush=True)
    np_full_print(intermediate_result.x)


################################################################################
# read the true tree

with open(opt.tree, "r") as tree_file:
    true_tree = Tree(tree_file.read().strip())

num_tree_nodes = len([x for x in true_tree.traverse()])

# traverse the tree, assigning names to each unnamed internal node
for idx, node in enumerate(true_tree.traverse("levelorder")):
    if not node.name or len(node.name) == 0:
        node.name = f"$_{idx}"

# traverse the tree, assigning indices to each node
node_indices = {node.name: idx for idx, node in enumerate(true_tree.traverse("levelorder"))}

################################################################################
# read the sequence data and compute nucleotide frequencies

nuc_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3, "?": 4}
unphased_nuc_to_idx = {
    "AA": 0,
    "CC": 1,
    "GG": 2,
    "TT": 3,
    "AC": 4,
    "AG": 5,
    "AT": 6,
    "CG": 7,
    "CT": 8,
    "GT": 9,
    "??": 10,
    "A?": 11,
    "C?": 12,
    "G?": 13,
    "T?": 14,
}
base_freq_counts = np.zeros(4, dtype=np.int64)

if opt.model == "DNA":
    n_states = 4
    with open(opt.seqs, "r") as seq_file:
        # first line consists of counts
        ntaxa, nsites = map(int, next(seq_file).split())
        # parse sequences
        sequences = dict()
        for line in seq_file:
            taxon, *seq = line.strip().split()
            seq = "".join(seq).upper()
            seq = np.array([nuc_to_idx[nuc] for nuc in seq], dtype=np.uint8)
            # compute nucleotide frequency
            for nuc_idx in seq:
                if nuc_idx < 4:
                    base_freq_counts[nuc_idx] += 1
                elif nuc_idx == 4:
                    # encountered "?"
                    for idx in range(4):
                        base_freq_counts[idx] += 0.25
            assert taxon not in sequences
            sequences[taxon] = seq
        assert ntaxa == len(sequences)
elif opt.model == "PHASED_DNA":
    n_states = 16
    with open(opt.seqs, "r") as seq_file:
        # first line consists of counts
        ntaxa, nsites = map(int, next(seq_file).split())
        # parse sequences
        sequences = dict()
        for line in seq_file:
            taxon, *seq = line.strip().split()
            seq = list(map(lambda s: s.upper(), seq))
            assert all(len(s) == 2 for s in seq)
            # compute nucleotide frequency
            for nuc_a, nuc_b in seq:
                if nuc_a != "?":
                    base_freq_counts[nuc_to_idx[nuc_a]] += 1
                else:
                    for idx in range(4):
                        base_freq_counts[idx] += 0.25
                if nuc_b != "?":
                    base_freq_counts[nuc_to_idx[nuc_b]] += 1
                else:
                    for idx in range(4):
                        base_freq_counts[idx] += 0.25
            # sequence coding is lexicographic AA, AC, AG, AT, A?, CA, ...
            # which is equivalent to a base-5 encoding 00=0, 01=1, 02=2, 03=3, 04=4, 10=5, ...
            seq = np.array(
                [
                    nuc_to_idx[nuc[0]] * 5 + nuc_to_idx[nuc[1]]
                    for nuc in map(lambda s: s.upper(), seq)
                ],
                dtype=np.uint8,
            )
            assert taxon not in sequences
            sequences[taxon] = seq
        assert ntaxa == len(sequences)
elif opt.model == "UNPHASED_DNA":
    n_states = 10
    # custom sequence encoding
    with open(opt.seqs, "r") as seq_file:
        ntaxa, nsites = map(int, next(seq_file).split())
        sequences = dict()
        for line in seq_file:
            taxon, *seq = line.strip().split()
            seq = list(map(lambda s: s.upper(), seq))
            assert all(len(s) == 2 for s in seq)
            for nuc_a, nuc_b in seq:
                if nuc_a != "?":
                    base_freq_counts[nuc_to_idx[nuc_a]] += 1
                else:
                    for idx in range(4):
                        base_freq_counts[idx] += 0.25
                if nuc_b != "?":
                    base_freq_counts[nuc_to_idx[nuc_b]] += 1
                else:
                    for idx in range(4):
                        base_freq_counts[idx] += 0.25
            seq = np.array(
                [unphased_nuc_to_idx[nuc] for nuc in seq],
                dtype=np.uint8,
            )
            assert taxon not in sequences
            sequences[taxon] = seq
        assert ntaxa == len(sequences)

    pass
else:
    assert False, "Unknown model selection"

pis = base_freq_counts / np.sum(base_freq_counts)
pi_a, pi_c, pi_g, pi_t = pis
pis16 = np.kron(pis, pis)
pis10 = pis16 @ V

assert set(true_tree.get_leaf_names()) == set(
    sequences.keys()
), "not the same leaves! are these matching datasets?"

taxa = sorted(sequences.keys())
taxa_indices = dict(map(lambda pair_: pair_[::-1], enumerate(taxa)))

# assemble the site pattern count tensor (sparse)
counts = defaultdict(lambda: 0)
for idx in range(nsites):
    # noinspection PyShadowingNames
    pattern = tuple(
        map(
            lambda taxon: sequences[taxon][idx],
            taxa,
        )
    )
    counts[pattern] += 1

################################################################################
# initial estimates for branch lengths based on F81 distances


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


unphased_nuc_to_idx = {
    "AA": 0,
    "CC": 1,
    "GG": 2,
    "TT": 3,
    "AC": 4,
    "AG": 5,
    "AT": 6,
    "CG": 7,
    "CT": 8,
    "GT": 9,
    "??": 10,
    "A?": 11,
    "C?": 12,
    "G?": 13,
    "T?": 14,
}


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


match opt.model:
    case "DNA":

        def sequence_distance(seq1: np.ndarray, seq2: np.ndarray) -> Tuple[float, float]:
            """
            F81 distance, a generalization of the JC69 distance, which takes into account nucleotide frequencies
            :param seq1: a sequence
            :param seq2: a sequence
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
                    np.nan_to_num(
                        (1 - disagreement) * disagreement / (beta * disagreement - 1) ** 2
                    ),
                    1,
                    1_000,
                ),
            )

    case "PHASED_DNA":

        def sequence_distance(seq1: np.ndarray, seq2: np.ndarray) -> Tuple[float, float]:
            """
            F81-16 distance, a generalization of the JC69 distance, which takes into account nucleotide frequencies
            :param seq1: a sequence
            :param seq2: a sequence
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

    case "UNPHASED_DNA":

        def sequence_distance(seq1: np.ndarray, seq2: np.ndarray) -> Tuple[float, float]:
            """
            F81-10 distance, a generalization of the JC69 distance, which takes into account nucleotide frequencies
            :param seq1: a sequence
            :param seq2: a sequence
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
                        -(
                            (
                                32 * beta**2 * (eta - zeta)
                                - 8 * beta**2 * disagreement
                                - 8 * beta
                                + 3
                            )
                            ** 2
                        )
                        * (
                            np.sqrt(2)
                            * (32 * beta**2 * (eta - zeta) - 3)
                            * beta
                            / (
                                (
                                    32 * beta**2 * (eta - zeta)
                                    - 8 * beta**2 * disagreement
                                    - 8 * beta
                                    + 3
                                )
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
                            / (
                                32 * beta**2 * (eta - zeta)
                                - 8 * beta**2 * disagreement
                                - 8 * beta
                                + 3
                            )
                            ** 2
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

    case _:
        assert False

# Compute all pairwise leaf distances
pair_to_idx = {
    (n1, n2) if n1 < n2 else (n2, n1): idx
    for idx, (n1, n2) in enumerate(itertools.combinations(taxa, 2))
}
leaf_stats = np.array(
    [sequence_distance(sequences[n1], sequences[n2]) for n1, n2 in itertools.combinations(taxa, 2)]
)
leaf_distances = leaf_stats[:, 0]
leaf_vars = leaf_stats[:, 1]


def complete_paths_forward(node, incoming_paths):
    # extend (forward) all paths in incoming_paths that start "somewhere" and end at node
    extended_paths = [partial_path + [node.name] for partial_path in incoming_paths]
    if node.is_leaf():
        return extended_paths
    else:
        left_child, right_child = node.children
        return complete_paths_forward(left_child, extended_paths) + complete_paths_forward(
            right_child, extended_paths
        )


def complete_paths_backward(node, incoming_paths):
    # extend (backward) all paths that in incoming_paths that start at node
    extended_paths = [[node.name] + partial_path for partial_path in incoming_paths]
    if node.is_leaf():
        return extended_paths
    else:
        left_child, right_child = node.children
        return complete_paths_backward(left_child, extended_paths) + complete_paths_backward(
            right_child, extended_paths
        )


def get_paths(node, is_root=True) -> List[List[str]]:
    # get all paths that go through node (as it's maximal tree node)
    if node.is_leaf():
        return []

    left_child, right_child = node.children

    # get paths that stay to the left and right of this node
    left_paths = get_paths(left_child, is_root=False)
    right_paths = get_paths(right_child, is_root=False)

    # get paths that pass directly through this node
    if is_root:
        through_paths = complete_paths_forward(right_child, [[]])
    else:
        through_paths = complete_paths_forward(right_child, [[node.name]])
    through_paths = complete_paths_backward(left_child, through_paths)

    return left_paths + through_paths + right_paths


true_paths = get_paths(true_tree)

# compute constraints matrix for lengths
# constrain the root edge to 0 length on root as this isn't a real edge length
constraints_eqn = [np.array([1.0] + [0.0] * (num_tree_nodes - 1), dtype=np.float64)]
constraints_val = [0.0]

# constraints for paths:
for path in true_paths:
    pair: Tuple[str, str] = (path[0], path[-1]) if path[0] < path[-1] else (path[-1], path[0])
    leaf_dist = leaf_distances[pair_to_idx[pair]]

    vec = np.zeros(num_tree_nodes, dtype=np.float64)
    for node in path:
        idx = node_indices[node]
        vec[idx] = 1

    constraints_eqn.append(vec)
    constraints_val.append(leaf_dist)

constraints_eqn = np.array(constraints_eqn)
constraints_val = np.array(constraints_val)


def objective(x):
    prediction_err = constraints_eqn @ x - constraints_val
    return np.mean(x**2) + np.mean(prediction_err**2 / np.concatenate(([1], leaf_vars)))


num_func_evals = 0
res = minimize(
    objective,
    np.ones(num_tree_nodes, dtype=np.float64),
    bounds=[(0.0, None)] + [(1e-10, None)] * (num_tree_nodes - 1),
    callback=callback_param if opt.log else None,
)

if not res.success:
    print("Continuing anyway")

# belt and suspenders for the constraint (avoid -1e-big type bounds violations)
tree_distances = np.maximum(0.0, res.x)

if opt.log:
    print("tree distances:")
    print(tree_distances)

# collect this data for comparison, likely on a different scale (GT transversion?), so not directly
# comparable, but possibly good to have
true_branch_lens = np.zeros(num_tree_nodes, dtype=np.float64)
for node in true_tree.traverse():
    true_branch_lens[node_indices[node.name]] = node.dist

################################################################################
# set rate constraint matrices

A_GTR = make_A_GTR(pis)
A_GTR16 = make_A_GTR16v(pis)
A_GTR_UNPH = make_A_GTR_unph(pis)

rate_constraint_matrix = make_rate_constraint_matrix(pis)

################################################################################
# initial estimates for GTR parameters

if opt.pre_estimate_params:

    # compute count pattern matrices
    count_patterns = np.zeros((len(taxa) * (len(taxa) - 1) // 2, 4, 4))
    for idx, (tx1, tx2) in enumerate(itertools.combinations(taxa, 2)):
        for idx1, idx2 in zip(sequences[tx1], sequences[tx2]):
            count_patterns[idx, idx1, idx2] += 1
    count_patterns /= np.sum(count_patterns, axis=(1, 2))[:, None, None]
    reduced_pattern = np.sum(count_patterns, axis=2)

    def make_initial_param_objective(A, dim, rate_constraint_matrix):
        def initial_param_objective(s_est, leaf_to_leaf_distances):
            p_ts = scipy.linalg.expm(
                leaf_to_leaf_distances[:, None, None] * (A @ s_est).reshape(dim, dim)[None, :, :]
            )
            joint_dist = reduced_pattern[:, :, None] * p_ts
            # compute KL divergence plus rate constraint
            return (
                np.mean(
                    np.sum(
                        np.maximum(
                            np.nan_to_num(xlogy(count_patterns, count_patterns), neginf=-1e5),
                            -1e5,
                        )
                        - np.maximum(
                            np.nan_to_num(xlogy(count_patterns, joint_dist), neginf=-1e5),
                            -1e5,
                        ),
                        axis=(1, 2),
                    )
                )
                + (rate_constraint_matrix @ s_est - 1) ** 2
            )

        return initial_param_objective

    match opt.model:
        case "DNA":
            initial_param_objective = make_initial_param_objective(
                A_GTR, n_states, rate_constraint_matrix
            )

        case "PHASED_DNA":
            initial_param_objective = make_initial_param_objective(
                A_GTR16, n_states, rate_constraint_matrix
            )

        case "UNPHASED_DNA":
            initial_param_objective = make_initial_param_objective(
                A_GTR_UNPH, n_states, rate_constraint_matrix
            )

        case _:
            assert False

    num_func_evals = 0
    res = minimize(
        initial_param_objective,
        (
            np.ones(6 if opt.model in ["DNA", "UNPHASED_DNA"] else 12)
        ),  # initial guess: F81 model (all s parameters == 1)
        args=constraints_eqn[1:] @ tree_distances,
        method=opt.method,
        bounds=[(0.0, np.inf)] * 6,
        callback=(
            (callback_ir if opt.method not in {"TNC", "SLSQP", "COBYLA"} else callback_param)
            if opt.log
            else None
        ),
    )
    if opt.log:
        print(res)

    s_est = res.x / (rate_constraint_matrix @ res.x)
else:
    s_est = np.ones(6)
    s_est = s_est / (rate_constraint_matrix @ s_est)

################################################################################
# jointly optimize GTR params and branch lens using neg-log likelihood


@numba.jit(nopython=True)
def prob_model_helper(t, left, right, evals):
    # return np.clip(((left * np.exp(t * evals)) @ right).astype(np.float64), 0.0, 1.0)
    return ((left * np.exp(t * evals)) @ right).astype(np.float64)


def prob_model_helper_vec(
    t: np.ndarray, left: np.ndarray, right: np.ndarray, evals: np.ndarray
) -> np.ndarray:
    return ((np.exp(t[:, None] * evals)[:, None, :] * left[None, :, :]) @ right).astype(np.float64)


def make_GTR_prob_model(gtr_params, *, vec=False):
    pi_a, pi_c, pi_g, pi_t = np.abs(gtr_params[:4])
    # print(f"{pi_a=} {pi_c=} {pi_g=} {pi_t=}")
    s_ac, s_ag, s_at, s_cg, s_ct, s_gt = np.abs(gtr_params[4:])
    # print(f"{s_ac=} {s_ag=} {s_at=} {s_cg=} {s_ct=} {s_gt=}")

    sym_Q = np.array(
        [
            [
                -(pi_c * s_ac + pi_g * s_ag + pi_t * s_at),
                np.sqrt(pi_a * pi_c) * s_ac,
                np.sqrt(pi_a * pi_g) * s_ag,
                np.sqrt(pi_a * pi_t) * s_at,
            ],
            [
                np.sqrt(pi_a * pi_c) * s_ac,
                -(pi_a * s_ac + pi_g * s_cg + pi_t * s_ct),
                np.sqrt(pi_c * pi_g) * s_cg,
                np.sqrt(pi_c * pi_t) * s_ct,
            ],
            [
                np.sqrt(pi_a * pi_g) * s_ag,
                np.sqrt(pi_c * pi_g) * s_cg,
                -(pi_a * s_ag + pi_c * s_cg + pi_t * s_gt),
                np.sqrt(pi_g * pi_t) * s_gt,
            ],
            [
                np.sqrt(pi_a * pi_t) * s_at,
                np.sqrt(pi_c * pi_t) * s_ct,
                np.sqrt(pi_g * pi_t) * s_gt,
                -(pi_a * s_at + pi_c * s_ct + pi_g * s_gt),
            ],
        ],
        dtype=np.float64,
    )

    evals, sym_evecs = np.linalg.eigh(sym_Q)
    # print(f"{evals=}")
    # print(f"{sym_evecs=}")

    left = sym_evecs * np.sqrt([pi_a, pi_c, pi_g, pi_t])[:, None]
    right = sym_evecs.T / np.sqrt([pi_a, pi_c, pi_g, pi_t])

    if vec:
        return lambda t: prob_model_helper_vec(t, left, right, evals)
    else:
        return lambda t: prob_model_helper(t, left, right, evals)


################################################################################
################################################################################
################################################################################


def compute_leaf_vec(patterns) -> Callable:
    # print(f"compute_leaf_vector({patterns=})")
    id4 = np.identity(4, dtype=np.float64)
    arr = np.concatenate((id4, [np.ones(4) / 4]), axis=0)
    with np.errstate(divide="ignore"):
        result = np.clip(
            np.log(np.array([arr[p, :] for p in patterns], dtype=np.float64)), -1e100, 0.0
        )

    # noinspection PyUnusedLocal
    def local_score_function_terminal(prob_matrices: np.ndarray) -> np.ndarray:
        return result

    # return local_score_function_terminal
    return numba.jit(local_score_function_terminal, nopython=True)


def log_matrix_mult(v, P) -> np.ndarray:
    # noinspection PyTypeChecker
    return logsumexp(np.expand_dims(v, axis=-1) + P, axis=-2, return_sign=False, keepdims=False)


def log_dot(v, w) -> np.ndarray:
    # noinspection PyTypeChecker
    return logsumexp(v + w, axis=-1, return_sign=False, keepdims=False)


# @numba.jit(nopython=False, forceobj=True)
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


def compute_score_function_helper(node, patterns, taxa_indices_) -> Callable:
    # taxa_indices_ should a dict who's keys are taxon names (str) and values should be the
    # corresponding index (int) in the patterns.
    # print(f"compute_score_function_helper({node=},{patterns=},{taxa_indices_=})")
    assert len(node.children) == 2
    left_node, right_node = node.children

    left_leaf_names = set(leaf.name for leaf in left_node.iter_leaves())
    right_leaf_names = set(leaf.name for leaf in right_node.iter_leaves())

    left_leaf_idcs = tuple(
        [idx for leaf_name, idx in taxa_indices_.items() if leaf_name in left_leaf_names]
    )
    right_leaf_idcs = tuple(
        [idx for leaf_name, idx in taxa_indices_.items() if leaf_name in right_leaf_names]
    )
    # print(f"{left_leaf_idcs=}")
    # print(f"{right_leaf_idcs=}")

    left_taxa_rel_indices = {
        name: idx
        for idx, name in enumerate(
            [leaf_name for leaf_name in taxa_indices_.keys() if leaf_name in left_leaf_names]
        )
    }
    right_taxa_rel_indices = {
        name: idx
        for idx, name in enumerate(
            [leaf_name for leaf_name in taxa_indices_.keys() if leaf_name in right_leaf_names]
        )
    }
    # print(f"{left_taxa_rel_indices=}")
    # print(f"{right_taxa_rel_indices=}")

    left_patterns, left_pattern_inverse = np.unique(
        patterns[:, left_leaf_idcs], axis=0, return_inverse=True
    )
    right_patterns, right_pattern_inverse = np.unique(
        patterns[:, right_leaf_idcs], axis=0, return_inverse=True
    )

    # print(f"{left_patterns=}")
    # print(f"{left_pattern_inverse=}")
    # print(f"{right_patterns=}")
    # print(f"{right_pattern_inverse=}")

    if left_node.is_leaf():
        w_l_function = compute_leaf_vec(left_patterns)
    else:
        w_l_function = compute_score_function_helper(
            left_node, left_patterns, left_taxa_rel_indices
        )

    if right_node.is_leaf():
        w_r_function = compute_leaf_vec(right_patterns)
    else:
        w_r_function = compute_score_function_helper(
            right_node, right_patterns, right_taxa_rel_indices
        )

    left_index = node_indices[left_node.name]
    right_index = node_indices[right_node.name]
    # print(f"{left_index=}")
    # print(f"{right_index=}")

    def local_score_function_branching(prob_matrices: np.ndarray) -> np.ndarray:
        w_l = w_l_function(prob_matrices)
        w_r = w_r_function(prob_matrices)

        with np.errstate(divide="ignore"):
            p_l = np.clip(np.log(np.clip(prob_matrices[left_index, :, :], 0.0, 1.0)), -1e100, 0.0)
            p_r = np.clip(np.log(np.clip(prob_matrices[right_index, :, :], 0.0, 1.0)), -1e100, 0.0)

        v_n = np.clip(
            log_matrix_mult(w_l[left_pattern_inverse], p_l)
            + log_matrix_mult(w_r[right_pattern_inverse], p_r),
            -1e100,
            0.0,
        )
        # print(f"{np.any(np.abs(v_n)>1e100)=}")
        # print(f"{v_n=}")
        return v_n

    return local_score_function_branching
    # return numba.jit(local_score_function_branching, nopython=False, forceobj=True)


def compute_score_function(*, root, patterns, pattern_counts) -> Callable:
    # print(f"compute_score_function({root=},{patterns=},{pattern_counts=})")
    v_function = compute_score_function_helper(root, patterns, taxa_indices)

    def score_function(prob_matrices):
        v = v_function(prob_matrices)
        return -kahan_dot(pattern_counts, log_dot(v, np.log(pis)))

    return numba.jit(score_function, nopython=False, forceobj=True)
    # return score_function


match opt.model:
    case "DNA":
        patterns = np.array([pattern for pattern in counts.keys()])
        pattern_counts = np.array([count for count in counts.values()])

        score_function = compute_score_function(
            root=true_tree, patterns=patterns, pattern_counts=pattern_counts
        )
    case "PHASED_DNA":
        genotype_counts = defaultdict(lambda: 0)
        for pattern, count in counts.items():
            pattern_mat = tuple(map(lambda p: p % 5, pattern))
            genotype_counts[pattern_mat] += 1.0

            pattern_pat = tuple(map(lambda p: p // 5, pattern))
            genotype_counts[pattern_mat] += 1.0

        patterns = np.array([pattern for pattern in genotype_counts.keys()])
        pattern_counts = np.array([count for count in genotype_counts.values()])

        score_function = compute_score_function(
            root=true_tree, patterns=patterns, pattern_counts=pattern_counts
        )
    case "UNPHASED_DNA":
        unphased_idx_to_phased_idcs = {
            0: (0,),  # "AA"
            1: (1,),  # "CC"
            2: (2,),  # "GG"
            3: (3,),  # "TT"
            4: (0, 1),  # "AC"
            5: (0, 2),  # "AG"
            6: (0, 3),  # "AT"
            7: (1, 2),  # "CG"
            8: (1, 3),  # "CT"
            9: (2, 3),  # "GT"
            10: (4, 4),  # "??"
            11: (0, 4),  # "A?"
            12: (1, 4),  # "C?"
            13: (2, 4),  # "G?"
            14: (3, 4),  # "T?"
        }
        genotype_counts = defaultdict(lambda: 0.0)
        for pattern, count in counts.items():
            pattern_options = [unphased_idx_to_phased_idcs[idx] for idx in pattern]
            phase_weight = 1 / reduce(mul, map(len, pattern_options), 1)
            for phase_resolved_pattern in itertools.product(*pattern_options):
                genotype_counts[phase_resolved_pattern] += phase_weight

        patterns = np.array([pattern for pattern in genotype_counts.keys()])
        pattern_counts = np.array([count for count in genotype_counts.values()])

        score_function = compute_score_function(
            root=true_tree, patterns=patterns, pattern_counts=pattern_counts
        )
    case _:
        assert False


def neg_log_likelihood(gtr_params, tree_distances):
    gtr_prob_model = make_GTR_prob_model(np.concatenate((pis, gtr_params)), vec=True)
    prob_matrices = gtr_prob_model(tree_distances)
    return score_function(prob_matrices)


def full_objective(params, gt_norm=False):
    gtr_params = params[:6]
    tree_distances = params[6:]
    return (
        (
            neg_log_likelihood(gtr_params / gtr_params[-1], tree_distances)
            if gt_norm
            else neg_log_likelihood(gtr_params, tree_distances)
        )
        + ((rate_constraint_matrix @ gtr_params) - 1) ** 2  # fix the rate
        + tree_distances[0] ** 2  # zero length at root
    )


def param_objective(gtr_params, tree_distances, gt_norm=False):
    return (
        neg_log_likelihood(gtr_params / gtr_params[-1], tree_distances)
        if gt_norm
        else neg_log_likelihood(gtr_params, tree_distances)
    ) + (
        (rate_constraint_matrix @ gtr_params) - 1
    ) ** 2  # fix the rate


def branch_length_objective(params, gtr_params):
    gtr_params = s_est
    tree_distances = params
    return (neg_log_likelihood(gtr_params, tree_distances)) + tree_distances[
        0
    ] ** 2  # zero length at root


for _ in range(2):

    num_func_evals = 0
    res = minimize(
        param_objective,
        s_est,
        args=(tree_distances,),
        method=opt.method,
        bounds=[(1e-10, np.inf)] * 6,
        callback=(
            (callback_ir if opt.method not in {"TNC", "SLSQP", "COBYLA"} else callback_param)
            if opt.log
            else None
        ),
    )
    if opt.log:
        print(res)

    s_est = res.x / (rate_constraint_matrix @ res.x)  # fine tune mu

    num_func_evals = 0
    res = minimize(
        branch_length_objective,
        tree_distances,
        args=(s_est,),
        method=opt.method,
        bounds=[(0.0, np.inf)] + [(1e-8, np.inf)] * (2 * len(taxa) - 2),
        callback=(
            (callback_ir if opt.method not in {"TNC", "SLSQP", "COBYLA"} else callback_param)
            if opt.log
            else None
        ),
    )
    if opt.log:
        print(res)

    tree_distances = res.x

num_func_evals = 0
res = minimize(
    full_objective,
    np.concatenate((s_est, tree_distances)),
    method=opt.method,
    bounds=[(0.0, np.inf)] * (6 + 2 * len(taxa) - 1),
    callback=(
        (callback_ir if opt.method not in {"TNC", "SLSQP", "COBYLA"} else callback_param)
        if opt.log
        else None
    ),
    options={"maxiter": 1000, "maxfun": 100_000},
)
if opt.log:
    print(res)

s_est = res.x[:6] / (rate_constraint_matrix @ res.x[:6])  # fine tune mu
tree_distances = res.x[6:]

if not res.success:
    # try to optimize the parameters and branch lengths separately
    print("fallback optimization")

    # first, the parameters
    print("optimize parameters")

    def param_objective(params, gt_norm=False):
        gtr_params = params
        return (
            neg_log_likelihood(gtr_params / gtr_params[-1], tree_distances)
            if gt_norm
            else neg_log_likelihood(gtr_params, tree_distances)
        ) + (
            (rate_constraint_matrix @ gtr_params) - 1
        ) ** 2  # fix the rate

    num_func_evals = 0
    res = minimize(
        param_objective,
        s_est,
        method=opt.method,
        bounds=[(0.0, np.inf)] * 6,
        callback=(
            (callback_ir if opt.method not in {"TNC", "SLSQP", "COBYLA"} else callback_param)
            if opt.log
            else None
        ),
    )
    if opt.log:
        print(res)

    s_est = res.x / (rate_constraint_matrix @ res.x)  # fine tune mu

    # next, the branch lengths
    print("optimize branch lengths")

    def branch_length_objective(params):
        gtr_params = s_est
        tree_distances = params
        return (neg_log_likelihood(gtr_params, tree_distances)) + tree_distances[
            0
        ] ** 2  # zero length at root

    num_func_evals = 0
    res = minimize(
        branch_length_objective,
        tree_distances,
        method=opt.method,
        bounds=[(0.0, np.inf)] * (2 * len(taxa) - 1),
        callback=(
            (callback_ir if opt.method not in {"TNC", "SLSQP", "COBYLA"} else callback_param)
            if opt.log
            else None
        ),
    )
    if opt.log:
        print(res)

    tree_distances = res.x

################################################################################
# update branch lens


for idx, node in enumerate(true_tree.traverse()):
    node.dist = tree_distances[idx]

################################################################################
# write tree and statistics to stdout or a file, depending up command line opts


newick_rep = true_tree.write(format=5)


def print_stats():
    print(f"neg log likelihood: {res.fun}")
    print()

    for s, v in zip(["s_ac", "s_ag", "s_at", "s_cg", "s_ct", "s_gt"], s_est):
        print(f"{s}: {v}")
    print()

    for p, v in zip(["pi_a", "pi_c", "pi_g", "pi_t"], pis):
        print(f"{p}: {v}")
    print()

    print("Q:")
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


if hasattr(opt, "output") and opt.output is not None:
    with open(opt.output + ".nwk", "w") as file:
        file.write(newick_rep)
        file.write("\n")

    with open(opt.output + ".log", "w") as file:
        with redirect_stdout(file):
            print_stats()

else:
    print(newick_rep)
    print()
    print_stats()

################################################################################
################################################################################
################################################################################
################################################################################
