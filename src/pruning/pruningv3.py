#!/usr/bin/env python3
import argparse
import functools
import itertools
import os.path
import sys
from collections import defaultdict
from contextlib import redirect_stdout
from functools import partial
from typing import Callable, Literal

import numba
import numpy as np
from ete3 import Tree
from scipy.optimize import OptimizeResult, minimize

from pruning.matrices import (
    U,
    cellphy10_rate,
    gtr4_rate,
    gtr10_rate,
    gtr10z_rate,
    make_A_GTR,
    make_GTR_prob_model,
    make_unphased_GTR_prob_model,
    perm,
)
from pruning.path_constraints import make_path_constraints
from pruning.util import kahan_dot, log_dot, log_matrix_mult

model_list = ["DNA", "PHASED_DNA", "UNPHASED_DNA", "CELLPHY", "CELLPHY_PI", "GTR10Z", "GTR10"]

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
    choices=model_list,
)
parser.add_argument(
    "--optimize_freqs",
    action="store_true",
    help="optimize frequencies using maximum likelihood. otherwise, stick with default data estimate",
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
parser.add_argument("--ambig", type=str, default="?", help="ambiguity character")
parser.add_argument("--output", type=str, help="output filename prefix for tree")
parser.add_argument("--overwrite", action="store_true", help="overwrite outputs, if they exist")
parser.add_argument("--log", action="store_true")
# parser.add_argument("--pre_estimate_params", action="store_true")

if hasattr(sys, "ps1"):
    # opt = parser.parse_args("--seqs test_100K.phy "
    #                         "--tree test.nwk "
    #                         "--model DNA "
    #                         "--log ".split())
    # opt = parser.parse_args(
    #     "--seqs test/test-diploid.phy --tree test/test-diploid.nwk --model PHASED_DNA --log ".split()
    # )
    opt = parser.parse_args(
        "--seqs test/test-diploid.phy --tree test/test-diploid.nwk --model UNPHASED_DNA --log ".split()
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

model: Literal["DNA", "PHASED_DNA", "UNPHASED_DNA", "CELLPHY", "CELLPHY_PI", "GTR10Z", "GTR10"] = (
    opt.model
)

ambig_char = opt.ambig.upper()
if len(ambig_char) != 1:
    print("Ambiguity character must be a single character")
    exit(-1)
if ambig_char in ["A", "C", "G", "T"]:
    print(f"Ambiguity character as '{ambig_char}' is not supported")
    exit(-1)


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

nuc_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3, ambig_char: 4}
unphased_nuc_to_idx = {
    "AA": 0,
    "CC": 1,
    "GG": 2,
    "TT": 3,
    "AC": 4,
    "CA": 4,
    "AG": 5,
    "GA": 5,
    "AT": 6,
    "TA": 6,
    "CG": 7,
    "GC": 7,
    "CT": 8,
    "TC": 8,
    "GT": 9,
    "TG": 9,
    ambig_char + ambig_char: 10,
    "A" + ambig_char: 11,
    ambig_char + "A": 11,
    "C" + ambig_char: 12,
    ambig_char + "C": 12,
    "G" + ambig_char: 13,
    ambig_char + "G": 13,
    "T" + ambig_char: 14,
    ambig_char + "T": 14,
}
base_freq_counts = np.zeros(4, dtype=np.int64)

# read and process the sequence file
with open(opt.seqs, "r") as seq_file:
    # first line consists of counts
    ntaxa, nsites = map(int, next(seq_file).split())

    match model:
        case "DNA":
            num_states = 4
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
        case "PHASED_DNA":
            num_states = 4
            # parse sequences
            sequences = dict()
            for line in seq_file:
                taxon, *seq = line.strip().split()
                seq = list(map(lambda s: s.upper(), seq))
                assert all(len(s) == 2 for s in seq)
                # compute nucleotide frequency
                for nuc_a, nuc_b in seq:
                    for nuc in [nuc_a, nuc_b]:
                        if nuc != ambig_char:
                            base_freq_counts[nuc_to_idx[nuc]] += 1
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
        case "UNPHASED_DNA" | "CELLPHY" | "CELLPHY_PI" | "GTR10Z" | "GTR10":
            num_states = 10
            sequences = dict()
            for line in seq_file:
                taxon, *seq = line.strip().split()
                seq = list(map(lambda s: s.upper(), seq))
                assert all(len(s) == 2 for s in seq)
                for nuc_a, nuc_b in seq:
                    for nuc in [nuc_a, nuc_b]:
                        if nuc != ambig_char:
                            base_freq_counts[nuc_to_idx[nuc]] += 1
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
        case _:
            assert False, "Unknown model selection"

pis = base_freq_counts / np.sum(base_freq_counts)
pi_a, pi_c, pi_g, pi_t = pis
pis16 = np.kron(pis, pis)
pis10 = U @ perm @ pis16

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
# initial estimates for branch lengths based on (generalized) F81 distances


match model:
    case "DNA":
        from pruning.distance_functions import dna_sequence_distance

        sequence_distance = partial(dna_sequence_distance, pis=pis)
    case "PHASED_DNA":
        from pruning.distance_functions import phased_sequence_distance

        sequence_distance = partial(phased_sequence_distance, pis=pis)
    case "UNPHASED_DNA" | "CELLPHY" | "CELLPHY_PI" | "GTR10Z" | "GTR10":
        # TODO: the others (cellphy, etc.) are included here as they are 10 state, not because this is a natural
        #  distance under their model
        from pruning.distance_functions import unphased_sequence_distance

        sequence_distance = partial(unphased_sequence_distance, pis=pis)
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
leaf_variances = leaf_stats[:, 1]

# create the constraint matrices
constraints_eqn, constraints_val = make_path_constraints(
    true_tree, num_tree_nodes, leaf_distances, pair_to_idx, node_indices
)


def branch_length_estimate_objective(x):
    # prediction error (weighted by variance) plus a regularizing term
    prediction_err = constraints_eqn @ x - constraints_val
    return np.mean(prediction_err**2 / np.concatenate(([1], leaf_variances))) + np.mean(x**2)


num_func_evals = 0
res = minimize(
    branch_length_estimate_objective,
    np.ones(num_tree_nodes, dtype=np.float64),
    bounds=[(0.0, None)] + [(1e-10, None)] * (num_tree_nodes - 1),
    callback=callback_param if opt.log else None,
)

if not res.success:
    print("Error in optimization, continuing anyway", flush=True)

# belt and suspenders for the constraint (avoid -1e-big type bounds violations)
tree_distances = np.maximum(0.0, res.x)

if opt.log:
    print("tree distances:")
    print(tree_distances)

# collect true tree data for comparison, likely on a different scale (GT transversion?), so not directly
# comparable, but possibly good to have
true_branch_lens = np.zeros(num_tree_nodes, dtype=np.float64)
for node in true_tree.traverse():
    true_branch_lens[node_indices[node.name]] = node.dist

################################################################################
# set rate constraint and initial estimates for GTR parameters

match model:
    case "DNA" | "PHASED_DNA" | "UNPHASED_DNA":

        num_params = 6
        rate_constraint = gtr4_rate
        s_est = np.ones(6)
        s_est = s_est / (rate_constraint(pis, s_est))
        pis_est = pis

    case "CELLPHY" | "CELLPHY_PI":

        num_params = 6
        rate_constraint = cellphy10_rate
        s_est = np.ones(6)
        s_est = s_est / (rate_constraint(pis, s_est))
        pis_est = pis10

    case "GTR10Z":

        num_params = 24
        rate_constraint = gtr10z_rate
        s_est = np.ones(24)
        s_est = s_est / rate_constraint(pis10, s_est)
        pis_est = pis10

    case "GTR10":

        num_params = 45
        rate_constraint = gtr10_rate
        s_est = np.ones(45)
        s_est = s_est / rate_constraint(pis10, s_est)
        pis_est = pis10

    case _:
        assert False, "Unknown model type"


##########################################################################################
# jointly optimize GTR params and branch lens using neg-log likelihood
##########################################################################################


def compute_leaf_vec(patterns, num_states) -> Callable:
    # print(f"compute_leaf_vector({patterns=})")
    match num_states:
        case 4:
            id4 = np.identity(4, dtype=np.float64)
            arr = np.concatenate((id4, [np.ones(4) / 4]), axis=0)
        case 10:
            id10 = np.identity(10, dtype=np.float64)
            arr = np.concatenate(
                (
                    id10,
                    [
                        np.array([1, 1, 1, 1, 2, 2, 2, 2, 2, 2]) / 16,  # ?/?
                        np.array([1, 0, 0, 0, 1, 1, 1, 0, 0, 0]) / 4,  # A/?
                        np.array([0, 1, 0, 0, 1, 0, 0, 1, 1, 0]) / 4,  # C/?
                        np.array([0, 0, 1, 0, 0, 1, 0, 1, 0, 1]) / 4,  # G/?
                        np.array([0, 0, 0, 1, 0, 0, 1, 0, 1, 1]) / 4,  # T/?
                    ],
                ),
                axis=0,
            )
        case _:
            raise NotImplementedError(f"Num states = {num_states} not implemented")

    with np.errstate(divide="ignore"):
        result = np.clip(
            np.log(np.array([arr[p, :] for p in patterns], dtype=np.float64)), -1e100, 0.0
        )

    # noinspection PyUnusedLocal
    def local_score_function_terminal(prob_matrices: np.ndarray) -> np.ndarray:
        return result

    # return local_score_function_terminal
    return numba.jit(local_score_function_terminal, nopython=True)


def compute_score_function_helper(node, patterns, taxa_indices_, num_states) -> Callable:
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
        w_l_function = compute_leaf_vec(left_patterns, num_states)
    else:
        w_l_function = compute_score_function_helper(
            left_node, left_patterns, left_taxa_rel_indices, num_states
        )

    if right_node.is_leaf():
        w_r_function = compute_leaf_vec(right_patterns, num_states)
    else:
        w_r_function = compute_score_function_helper(
            right_node, right_patterns, right_taxa_rel_indices, num_states
        )

    left_index = node_indices[left_node.name]
    right_index = node_indices[right_node.name]

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
        return v_n

    return local_score_function_branching


def compute_score_function(*, root, patterns, pattern_counts, num_states) -> Callable:
    # print(f"compute_score_function({root=},{patterns=},{pattern_counts=})")
    v_function = compute_score_function_helper(root, patterns, taxa_indices, num_states)

    def score_function(pis, prob_matrices):
        v = v_function(prob_matrices)
        return -kahan_dot(pattern_counts, log_dot(v, np.log(pis)))

    return numba.jit(score_function, nopython=False, forceobj=True)
    # return score_function


################################################################################
################################################################################


# def compute_unphased_leaf_vec(patterns) -> Callable:
#     # print(f"compute_unphased_leaf_vector({patterns=})")
#     id10 = np.identity(10, dtype=np.float64)
#     arr = np.concatenate((id10, [np.ones(10) / 10]), axis=0)
#     # TODO: other ambiguities?
#     with np.errstate(divide="ignore"):
#         result = np.clip(
#             np.log(np.array([arr[p, :] for p in patterns], dtype=np.float64)), -1e100, 0.0
#         )
#
#     # noinspection PyUnusedLocal
#     def local_score_function_terminal(prob_matrices: np.ndarray) -> np.ndarray:
#         # print(f"local_score_function_terminal({prob_matrices=})->{result}")
#         return result
#
#     return local_score_function_terminal
#     # return numba.jit(local_score_function_terminal, nopython=True)
#
#
# def compute_unphased_score_function_helper(node, patterns, taxa_indices_) -> Callable:
#     # taxa_indices_ should a dict who's keys are taxon names (str) and values should be the
#     # corresponding index (int) in the patterns.
#     # print(f"compute_unphased_score_function_helper({node=},{patterns=},{taxa_indices_=})")
#     assert len(node.children) == 2
#     left_node, right_node = node.children
#
#     left_leaf_names = set(leaf.name for leaf in left_node.iter_leaves())
#     right_leaf_names = set(leaf.name for leaf in right_node.iter_leaves())
#
#     left_leaf_idcs = tuple(
#         [idx for leaf_name, idx in taxa_indices_.items() if leaf_name in left_leaf_names]
#     )
#     right_leaf_idcs = tuple(
#         [idx for leaf_name, idx in taxa_indices_.items() if leaf_name in right_leaf_names]
#     )
#
#     left_taxa_rel_indices = {
#         name: idx
#         for idx, name in enumerate(
#             [leaf_name for leaf_name in taxa_indices_.keys() if leaf_name in left_leaf_names]
#         )
#     }
#     right_taxa_rel_indices = {
#         name: idx
#         for idx, name in enumerate(
#             [leaf_name for leaf_name in taxa_indices_.keys() if leaf_name in right_leaf_names]
#         )
#     }
#
#     left_patterns, left_pattern_inverse = np.unique(
#         patterns[:, left_leaf_idcs], axis=0, return_inverse=True
#     )
#     right_patterns, right_pattern_inverse = np.unique(
#         patterns[:, right_leaf_idcs], axis=0, return_inverse=True
#     )
#
#     if left_node.is_leaf():
#         w_l_function = compute_unphased_leaf_vec(left_patterns)
#     else:
#         w_l_function = compute_unphased_score_function_helper(
#             left_node, left_patterns, left_taxa_rel_indices
#         )
#
#     if right_node.is_leaf():
#         w_r_function = compute_unphased_leaf_vec(right_patterns)
#     else:
#         w_r_function = compute_unphased_score_function_helper(
#             right_node, right_patterns, right_taxa_rel_indices
#         )
#
#     left_index = node_indices[left_node.name]
#     right_index = node_indices[right_node.name]
#
#     def local_score_function_branching(prob_matrices: np.ndarray) -> np.ndarray:
#         w_l = w_l_function(prob_matrices)
#         w_r = w_r_function(prob_matrices)
#
#         with np.errstate(divide="ignore"):
#             p_l = np.clip(np.log(np.clip(prob_matrices[left_index, :, :], 0.0, 1.0)), -1e100, 0.0)
#             p_r = np.clip(np.log(np.clip(prob_matrices[right_index, :, :], 0.0, 1.0)), -1e100, 0.0)
#
#         v_n = np.clip(
#             log_matrix_mult(w_l[left_pattern_inverse], p_l)
#             + log_matrix_mult(w_r[right_pattern_inverse], p_r),
#             -1e100,
#             0.0,
#         )
#         return v_n
#
#     return local_score_function_branching
#     # return numba.jit(local_score_function_branching, nopython=False, forceobj=True)
#
#
# def compute_unphased_score_function(*, root, patterns, pattern_counts) -> Callable:
#     # print(f"compute_unphased_score_function({root=},{patterns=},{pattern_counts=})")
#     def score_function(prob_matrices, *, pis10, v_function):
#         v = v_function(prob_matrices)
#         return -kahan_dot(pattern_counts, log_dot(v, np.log(pis10)))
#
#     score_function = functools.partial(
#         score_function,
#         v_function=compute_unphased_score_function_helper(root, patterns, taxa_indices),
#         pis10=U @ perm @ np.kron(pis, pis),
#     )
#
#     return score_function


################################################################################
################################################################################

match opt.model:
    case "DNA":
        patterns = np.array([pattern for pattern in counts.keys()])
        pattern_counts = np.array([count for count in counts.values()])

        prob_model_maker = make_GTR_prob_model

    case "PHASED_DNA":
        genotype_counts = defaultdict(lambda: 0)
        for pattern, count in counts.items():
            pattern_mat = tuple(map(lambda p: p % 5, pattern))
            genotype_counts[pattern_mat] += 1.0

            pattern_pat = tuple(map(lambda p: p // 5, pattern))
            genotype_counts[pattern_mat] += 1.0

        patterns = np.array([pattern for pattern in genotype_counts.keys()])
        pattern_counts = np.array([count for count in genotype_counts.values()])

        prob_model_maker = make_GTR_prob_model

    case "UNPHASED_DNA":
        patterns = np.array([pattern for pattern in counts.keys()])
        pattern_counts = np.array([count for count in counts.values()])

        prob_model_maker = make_unphased_GTR_prob_model

    case _:
        assert False


def neg_log_likelihood_prototype(
    pis, model_params, tree_distances, *, prob_model_maker, score_function
):
    prob_model = prob_model_maker(pis, model_params, vec=True)
    prob_matrices = prob_model(tree_distances)
    return score_function(pis, prob_matrices)


neg_log_likelihood = functools.partial(
    neg_log_likelihood_prototype,
    prob_model_maker=prob_model_maker,
    score_function=compute_score_function(
        root=true_tree,
        patterns=patterns,
        pattern_counts=pattern_counts,
        num_states=num_states,
    ),
)


####################################################################################################
# define optimization objectives for the model parameters and branch lengths


def param_objective(gtr_params, pis, tree_distances, gt_norm=False):
    """
    Objective function for GTR parameters.

    :param gtr_params:
    :param pis:
    :param tree_distances: (fixed)
    :param gt_norm: if True, normalize the GT rate to 1
    :return: loss
    """
    return (
        neg_log_likelihood(pis, gtr_params / gtr_params[-1], tree_distances)
        if gt_norm
        else neg_log_likelihood(pis, gtr_params, tree_distances)
    ) + (
        0 if gt_norm else (rate_constraint(pis, gtr_params) - 1) ** 2
    )  # fix the overall rate, if not normalizing on the GT rate


def branch_length_objective(tree_distances, pis, gtr_params):
    """
    Objective function for branch length estimation.

    :param tree_distances:
    :param pis:
    :param gtr_params: (fixed)
    :return: loss
    """
    return (neg_log_likelihood(pis, gtr_params, tree_distances)) + tree_distances[
        0
    ] ** 2  # zero length at root


def full_param_objective(params, tree_distances, gt_norm=False):
    """
    Objective function for model parameters + frequencies

    :param params: pis+model_params
    :param tree_distances: (fixed)
    :param gt_norm: if True, normalize the GT rate to 1
    :return: loss
    """
    pis = params[:num_states]
    model_params = params[num_states:]
    return (
        neg_log_likelihood(pis, model_params / model_params[-1], tree_distances)
        if gt_norm
        else neg_log_likelihood(pis, model_params, tree_distances)
    ) + (
        0 if gt_norm else (rate_constraint(pis, model_params) - 1) ** 2
    )  # fix the overall rate, if not normalizing on the GT rate


for _ in range(2):

    num_func_evals = 0
    res = minimize(
        param_objective,
        s_est,
        args=(
            pis,
            tree_distances,
        ),
        method=opt.method,
        bounds=[(1e-10, np.inf)] * num_params,
        callback=(
            (callback_ir if opt.method not in {"TNC", "SLSQP", "COBYLA"} else callback_param)
            if opt.log
            else None
        ),
        options={"maxiter": 1000, "maxfun": 100_000, "ftol": 1e-10},
    )
    if opt.log:
        print(res)

    s_est = res.x / rate_constraint(pis, res.x)  # fine tune mu

    num_func_evals = 0
    res = minimize(
        branch_length_objective,
        tree_distances,
        args=(pis_est, s_est),
        method=opt.method,
        bounds=[(0.0, np.inf)] + [(1e-8, np.inf)] * (2 * len(taxa) - 2),
        callback=(
            (callback_ir if opt.method not in {"TNC", "SLSQP", "COBYLA"} else callback_param)
            if opt.log
            else None
        ),
        options={"maxiter": 1000, "maxfun": 100_000, "ftol": 1e-10},
    )
    if opt.log:
        print(res)

    # belt and suspenders for the constraint (avoid -1e-big type bounds violations)
    tree_distances = np.maximum(0.0, res.x)

    if opt.optimize_freqs:
        num_func_evals = 0
        res = minimize(
            full_param_objective,
            np.concatenate((pis_est, s_est)),
            args=(tree_distances,),
            method=opt.method,
            bounds=[(1e-10, np.inf)] * (num_states + num_params),
            callback=(
                (callback_ir if opt.method not in {"TNC", "SLSQP", "COBYLA"} else callback_param)
                if opt.log
                else None
            ),
            options={"maxiter": 1000, "maxfun": 100_000, "ftol": 1e-10},
        )
        if opt.log:
            print(res)

        pis_est = np.maximum(0.0, res.x[:num_states])
        s_est = np.maximum(0.0, s_est[num_states:])
        s_est = s_est / rate_constraint(pis, s_est)  # fine tune mu


####################################################################################################

if opt.optimize_freqs:

    def full_objective(params, *, gt_norm=False):
        """
        Full objective function.

        :param params: first entries are frequencies, next entries are the model params, rest are branch lengths
        :param gt_norm: if True, normalize the GT rate to 1
        :return: loss
        """
        pis = params[:num_states]
        model_params = params[num_states : num_states + num_params]
        tree_distances = params[num_states + num_params :]
        return (
            (
                neg_log_likelihood(pis, model_params / model_params[-1], tree_distances)
                if gt_norm
                else neg_log_likelihood(pis, model_params, tree_distances)
            )
            + (0 if gt_norm else ((rate_constraint(pis, model_params) - 1) ** 2))  # fix the rate
            + tree_distances[0] ** 2  # zero length at root
        )

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
        options={"maxiter": 1000, "maxfun": 100_000, "ftol": 1e-10},
    )
    if opt.log:
        print(res)

    pis_est = np.maximum(0.0, res.x[:num_states])
    s_est = np.maximum(0.0, res.x[num_states : num_states + num_params])
    s_est = s_est / rate_constraint(pis_est, s_est)  # fine tune mu
    tree_distances = np.maximum(0.0, res.x[num_states + num_params:])

else:
    # optimize everything but the state frequencies

    def full_objective(params, pis, gt_norm=False):
        """
        Full objective function.

        :param params: first entries are the model params, rest are branch lengths
        :param pis: state frequencies
        :param gt_norm: if True, normalize the GT rate to 1
        :return: loss
        """
        model_params = params[:num_params]
        tree_distances = params[num_params:]
        return (
            (
                neg_log_likelihood(pis, model_params / model_params[-1], tree_distances)
                if gt_norm
                else neg_log_likelihood(pis, model_params, tree_distances)
            )
            + (0 if gt_norm else ((rate_constraint(pis, model_params) - 1) ** 2))  # fix the rate
            + tree_distances[0] ** 2  # zero length at root
        )

    num_func_evals = 0
    res = minimize(
        full_objective,
        np.concatenate((s_est, tree_distances)),
        args=(pis_est,),
        method=opt.method,
        bounds=[(0.0, np.inf)] * (num_params + 2 * len(taxa) - 1),
        callback=(
            (callback_ir if opt.method not in {"TNC", "SLSQP", "COBYLA"} else callback_param)
            if opt.log
            else None
        ),
        options={"maxiter": 1000, "maxfun": 100_000, "ftol": 1e-10},
    )
    if opt.log:
        print(res)

    s_est = np.maximum(0.0, res.x[:num_params])
    s_est = s_est / rate_constraint(pis_est, s_est)  # fine tune mu
    tree_distances = np.maximum(0.0, res.x[num_params:])

################################################################################
# update branch lens in ETE3 tree


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

    # TODO: per model code
    print("Q:")
    A_GTR = make_A_GTR(pis)
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
