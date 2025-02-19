#!/usr/bin/env python3
import argparse
import itertools
import sys
from collections import defaultdict
from contextlib import redirect_stdout
from typing import List, Tuple

import numpy as np
import scipy
from ete3 import Tree
from numba import jit
from scipy.optimize import OptimizeResult, minimize
from scipy.special import xlogy

from matrices import (
    V,
    make_A_GTR,
    make_A_GTR16v,
    make_A_GTR_unph,
    make_rate_constraint_matrix,
    make_rate_constraint_matrix_gtr16,
    make_rate_constraint_matrix_gtr16v,
)

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
parser.add_argument("--log", action="store_true")
parser.add_argument("--pre_estimate_params", action="store_true")

if hasattr(sys, "ps1"):
    opt = parser.parse_args("--seqs test_100K.phy --tree test.nwk --log".split())
else:
    opt = parser.parse_args()

if opt.log:
    print(opt)


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

nuc_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
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
                base_freq_counts[nuc_idx] += 1
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
                base_freq_counts[nuc_to_idx[nuc_a]] += 1
                base_freq_counts[nuc_to_idx[nuc_b]] += 1
            # sequence coding is lexicographic AA, AC, AG, AT, CA, ...
            # which is equivalent to a base-4 encoding 00=0, 01=1, 02=2, 03=3, 10=4, ...
            seq = np.array(
                [
                    nuc_to_idx[nuc[0]] * 4 + nuc_to_idx[nuc[1]]
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
                base_freq_counts[nuc_to_idx[nuc_a]] += 1
                base_freq_counts[nuc_to_idx[nuc_b]] += 1
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
            disagreement = np.sum(seq1 != seq2) / len(seq1)
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
            disagreement = np.sum(seq1 != seq2) / len(seq1)
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
            disagreement = np.sum(seq1 != seq2) / len(seq1)
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
    bounds=[(0.0, None)] * num_tree_nodes,
    callback=callback_param if opt.log else None,
)

if not res.success:
    print("Continuing anyway")

# belt and suspenders for the constraint (avoid -1e-6 type bounds violations)
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
# initial estimates for GTR parameters

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
        A_GTR = make_A_GTR(pis)
        rate_constraint_matrix = make_rate_constraint_matrix(pis)
        initial_param_objective = make_initial_param_objective(
            A_GTR, n_states, rate_constraint_matrix
        )

    case "PHASED_DNA":
        A_GTR16 = make_A_GTR16v(pis)
        rate_constraint_matrix = make_rate_constraint_matrix_gtr16v(pis)
        initial_param_objective = make_initial_param_objective(
            A_GTR16, n_states, rate_constraint_matrix
        )

    case "UNPHASED_DNA":
        A_GTR_UNPH = make_A_GTR_unph(pis)
        rate_constraint_matrix = make_rate_constraint_matrix_gtr16(pis)
        initial_param_objective = make_initial_param_objective(
            A_GTR_UNPH, n_states, rate_constraint_matrix
        )

    case _:
        assert False

if opt.pre_estimate_params:
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
    s_est = np.ones(6 if opt.model in ["DNA", "UNPHASED_DNA"] else 12)
    s_est = s_est / (rate_constraint_matrix @ s_est)

################################################################################
# jointly optimize GTR params and branch lens using neg-log likelihood


@jit(nopython=True)
def prob_model_helper(t, left, right, evals):
    return ((left * np.exp(t * evals)) @ right).astype(np.float64)


def make_GTR_prob_model(gtr_params):
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

    return lambda t: prob_model_helper(t, left, right, evals)


################################################################################
################################################################################
################################################################################

assert opt.model == "DNA", "Nothing below this line is adapted to any other model"


def compute_leaf_vec(node, patterns) -> np.ndarray:
    nucs = patterns[:, taxa_indices[node.name]]
    e_nuc = np.zeros((patterns.shape[0], 4))
    for idx, nuc in enumerate(nucs):
        e_nuc[idx, nuc] = 1
    return e_nuc


def compute_score_helper(node, prob_model, tree_distances, patterns) -> np.ndarray:
    assert len(node.children) == 2
    left_node, right_node = node.children

    if left_node.is_leaf():
        w_l = compute_leaf_vec(left_node, patterns)
    else:
        w_l = compute_score_helper(left_node, prob_model, tree_distances, patterns)

    t = tree_distances[node_indices[left_node.name]]
    w_l = np.einsum("ji,nj->ni", prob_model(t), w_l)

    if right_node.is_leaf():
        w_r = compute_leaf_vec(right_node, patterns)
    else:
        w_r = compute_score_helper(right_node, prob_model, tree_distances, patterns)

    t = tree_distances[node_indices[right_node.name]]
    w_r = np.einsum("ji,nj->ni", prob_model(t), w_r)

    v_n = w_l * w_r

    return v_n


def compute_score(*, root, prob_model, tree_distances, patterns, pattern_counts) -> float:
    v = compute_score_helper(root, prob_model, tree_distances, patterns)
    # print(f"{v=}")
    return pattern_counts @ np.nan_to_num(np.log(v @ pis))


def neg_log_likelihood(gtr_params, tree_distances):
    gtr_prob_model = make_GTR_prob_model(np.concatenate((pis, gtr_params)))
    patterns = np.array([pattern for pattern in counts.keys()])
    pattern_counts = np.array([count for count in counts.values()])

    return -compute_score(
        root=true_tree,
        prob_model=gtr_prob_model,
        tree_distances=tree_distances,
        patterns=patterns,
        pattern_counts=pattern_counts,
    )


def neg_log_likelihood_unphased(gtr_params, tree_distances):
    gtr_prob_model = make_GTR_prob_model(np.concatenate((pis, gtr_params)))

    unphased_idx_to_phased_idcs = {
        0: (0, 0),  # "AA"
        1: (1, 1),  # "CC"
        2: (2, 2),  # "GG"
        3: (3, 3),  # "TT"
        4: (0, 1),  # "AC"
        5: (0, 2),  # "AG"
        6: (0, 3),  # "AT"
        7: (1, 2),  # "CG"
        8: (1, 3),  # "CT"
        9: (2, 3),  # "GT"
    }
    genotype_counts = defaultdict(lambda: 0.0)
    # TODO: optimize
    for pattern, count in counts.keys():
        for resolved_pattern in itertools.product(
            *(unphased_idx_to_phased_idcs[idx] for idx in pattern)
        ):
            genotype_counts[resolved_pattern] += 0.5 ** len(pattern)

    patterns = np.array([pattern for pattern in genotype_counts.keys()])
    pattern_counts = np.array([count for count in genotype_counts.values()])

    return -compute_score(
        root=true_tree,
        prob_model=gtr_prob_model,
        tree_distances=tree_distances,
        patterns=patterns,
        pattern_counts=pattern_counts,
    )


def neg_log_likelihood_phased(gtr_params, tree_distances):
    gtr_prob_model = make_GTR_prob_model(np.concatenate((pis, gtr_params)))

    genotype_counts = defaultdict(lambda: 0)
    for pattern, count in counts.keys():
        pattern_mat = tuple(map(lambda p: p % 4, pattern))
        genotype_counts[pattern_mat] += 1
        pattern_pat = tuple(map(lambda p: p // 4, pattern))
        genotype_counts[pattern_pat] += 1

    patterns = np.array([pattern for pattern in genotype_counts.keys()])
    pattern_counts = np.array([count for count in genotype_counts.values()])

    return -compute_score(
        root=true_tree,
        prob_model=gtr_prob_model,
        tree_distances=tree_distances,
        patterns=patterns,
        pattern_counts=pattern_counts,
    )


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
)
if opt.log:
    print(res)

s_est = res.x[:6] / (rate_constraint_matrix @ res.x[:6])  # fine tune mu
tree_distances = res.x[6:]

################################################################################
# update branch lens


for idx, node in enumerate(true_tree.traverse()):
    node.dist = tree_distances[idx]

################################################################################
# write tree and statistics to stdout or a file, depending up command line opts


newick_rep = true_tree.write(format=5)


def print_stats():
    print(f"neg log likelihood: {-res.fun}")
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
