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

parser = argparse.ArgumentParser(
    description="Compute log likelihood using the pruning algorithm"
)
parser.add_argument(
    "--seqs", type=str, required=True, help="sequence alignments in phylip format"
)
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
parser.add_argument("--log")

if hasattr(sys, "ps1"):
    opt = parser.parse_args(
        # "--seqs /home/knappa/build-algo/data/1K-ultrametric/data_1_GTR_I_Gamma_1K_sites_ultrametric.phy "
        # "--tree /home/knappa/build-algo/data/1K-ultrametric/trees_50_taxa_ultrametric_1.nwk".split()
        # "--seqs /home/knappa/build-algo/data/100K-ultrametric/data_1_GTR_I_Gamma_100K_sites_ultrametric.phy "
        # "--tree /home/knappa/build-algo/data/100K-ultrametric/trees_50_taxa_ultrametric_1.nwk".split()
        "--seqs /home/knappa/test_100K.phy --tree /home/knappa/test.nwk --log".split()
    )
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


def callback_annealing(x, f, context):
    global num_func_evals
    num_func_evals += 1
    print(num_func_evals, flush=True)
    np_full_print(x)
    np_full_print(f)
    np_full_print(context)


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
node_indices = {
    node.name: idx for idx, node in enumerate(true_tree.traverse("levelorder"))
}

################################################################################
# read the sequence data and compute nucleotide frequencies

nuc_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
freq_counts = np.zeros(4, dtype=np.int64)

if opt.model == "DNA":
    with open(opt.seqs, "r") as seq_file:
        ntaxa, nsites = map(int, next(seq_file).split())
        sequences = dict()
        for line in seq_file:
            taxon, *seq = line.strip().split()
            seq = "".join(seq).upper()
            seq = np.array([nuc_to_idx[nuc] for nuc in seq], dtype=np.uint8)
            for nuc_idx in seq:
                freq_counts[nuc_idx] += 1
            assert taxon not in sequences
            sequences[taxon] = seq
        assert ntaxa == len(sequences)
elif opt.model == "PHASED_DNA":
    with open(opt.seqs, "r") as seq_file:
        ntaxa, nsites = map(int, next(seq_file).split())
        sequences = dict()
        for line in seq_file:
            taxon, *seq = line.strip().split()
            seq = list(map(lambda s: s.upper(), seq))
            assert all(len(s) == 2 for s in seq)
            for nuc_a, nuc_b in seq:
                freq_counts[nuc_to_idx[nuc_a]] += 1
                freq_counts[nuc_to_idx[nuc_b]] += 1
            seq = np.array(
                [
                    nuc_to_idx[nuc[1]] * 4 + nuc_to_idx[nuc[0]]
                    for nuc in map(lambda s: s.upper(), seq)
                ],
                dtype=np.uint8,
            )
            assert taxon not in sequences
            sequences[taxon] = seq
        assert ntaxa == len(sequences)
elif opt.model == "UNPHASED_DNA":
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
    with open(opt.seqs, "r") as seq_file:
        ntaxa, nsites = map(int, next(seq_file).split())
        sequences = dict()
        for line in seq_file:
            taxon, *seq = line.strip().split()
            seq = list(map(lambda s: s.upper(), seq))
            assert all(len(s) == 2 for s in seq)
            for nuc_a, nuc_b in seq:
                freq_counts[nuc_to_idx[nuc_a]] += 1
                freq_counts[nuc_to_idx[nuc_b]] += 1
            seq = np.array(
                [unphased_nuc_to_idx[nuc] for nuc in seq],
                dtype=np.uint8,
            )
            assert taxon not in sequences
            sequences[taxon] = seq
        assert ntaxa == len(sequences)

    pass
else:
    assert False, "Invalid model"

pis = freq_counts / np.sum(freq_counts)
pi_a, pi_c, pi_g, pi_t = pis

assert (
    set(true_tree.get_leaf_names()) == sequences.keys()
), "not the same leaves! are these matching datasets?"


taxa = sorted(sequences.keys())
taxa_indices = dict(map(lambda pair_: pair_[::-1], enumerate(taxa)))

# assemble the site pattern count tensor
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

assert opt.model == "DNA", "Nothing below this line is adapted to any other model"


def sequence_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """
    F81 distance, a generalization of the JC69 distance, which takes into account nucleotide frequencies
    :param seq1: a sequence
    :param seq2: a sequence
    :return: F81 distance
    """
    #
    beta = 1 / (1 - np.sum(pis**2))  # 1/mu
    disagreement = np.sum(seq1 != seq2) / len(seq1)
    return -np.log(1 - beta * np.minimum(1 / beta - 1e-10, disagreement)) / beta


# Compute all pairwise leaf distances
pair_to_idx = {
    (n1, n2) if n1 < n2 else (n2, n1): idx
    for idx, (n1, n2) in enumerate(itertools.combinations(taxa, 2))
}
leaf_distances = np.array(
    [
        sequence_distance(sequences[n1], sequences[n2])
        for n1, n2 in itertools.combinations(taxa, 2)
    ]
)


def complete_paths_forward(node, incoming_paths):
    # extend (forward) all paths in incoming_paths that start "somewhere" and end at node
    extended_paths = [partial_path + [node.name] for partial_path in incoming_paths]
    if node.is_leaf():
        return extended_paths
    else:
        left_child, right_child = node.children
        return complete_paths_forward(
            left_child, extended_paths
        ) + complete_paths_forward(right_child, extended_paths)


def complete_paths_backward(node, incoming_paths):
    # extend (backward) all paths that in incoming_paths that start at node
    extended_paths = [[node.name] + partial_path for partial_path in incoming_paths]
    if node.is_leaf():
        return extended_paths
    else:
        left_child, right_child = node.children
        return complete_paths_backward(
            left_child, extended_paths
        ) + complete_paths_backward(right_child, extended_paths)


def get_paths(node) -> List[List[str]]:
    # get all paths that go through node (as it's maximal tree node)
    if node.is_leaf():
        return []

    left_child, right_child = node.children

    # get paths that stay to the left and right of this node
    left_paths = get_paths(left_child)
    right_paths = get_paths(right_child)

    # get paths that pass directly through this node
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
    pair: Tuple[str, str] = (
        (path[0], path[-1]) if path[0] < path[-1] else (path[-1], path[0])
    )
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
    return np.mean(x**2) + np.mean(prediction_err**2)


num_func_evals = 0
res = minimize(
    objective,
    np.ones(num_tree_nodes, dtype=np.float64),
    bounds=[(0.0, 0.0)] + [(0.0, None)] * (num_tree_nodes - 1),
    callback=callback_param if opt.log else None,
)

if not res.success:
    print("Continuing anyway")

# belt and suspenders for the constraint (avoid -1e-6 type bounds violations)
tree_distances = np.maximum(0.0, res.x)

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


A_GTR = np.array(
    [
        # fmt: off
        #  s_ac   s_ag   s_at   s_cg   s_ct   s_gt
        # row 1
        [ -pi_c, -pi_g, -pi_t,     0,     0,     0],
        [  pi_c,     0,     0,     0,     0,     0],
        [     0,  pi_g,     0,     0,     0,     0],
        [     0,     0,  pi_t,     0,     0,     0],
        # row 2
        [  pi_a,     0,     0,     0,     0,     0],
        [ -pi_a,     0,     0, -pi_g, -pi_t,     0],
        [     0,     0,     0,  pi_g,     0,     0],
        [     0,     0,     0,     0,  pi_t,     0],
        # row 3
        [     0,  pi_a,     0,     0,     0,     0],
        [     0,     0,     0,  pi_c,     0,     0],
        [     0, -pi_a,     0, -pi_c,     0, -pi_t],
        [     0,     0,     0,     0,     0,  pi_t],
        # row 4
        [     0,     0,  pi_a,     0,     0,     0],
        [     0,     0,     0,     0,  pi_c,     0],
        [     0,     0,     0,     0,     0,  pi_g],
        [     0,     0, -pi_a,     0, -pi_c, -pi_g],
        # fmt: on
    ]
)

rate_constraint_matrix = np.array(
    [
        2 * pi_a * pi_c,
        2 * pi_a * pi_g,
        2 * pi_a * pi_t,
        2 * pi_c * pi_g,
        2 * pi_c * pi_t,
        2 * pi_g * pi_t,
    ]
)


def initial_gtr_param_objective(s_est, leaf_to_leaf_distances):
    p_ts = scipy.linalg.expm(
        leaf_to_leaf_distances[:, None, None]
        * (A_GTR @ s_est).reshape(4, 4)[None, :, :]
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
                    np.nan_to_num(xlogy(count_patterns, joint_dist), neginf=-1e5), -1e5
                ),
                axis=(1, 2),
            )
        )
        + (rate_constraint_matrix @ s_est - 1) ** 2
    )


num_func_evals = 0
res = minimize(
    initial_gtr_param_objective,
    np.ones(6),  # initial guess: F81 model (all s parameters == 1)
    args=constraints_eqn[1:] @ tree_distances,
    method=opt.method,
    bounds=[(0.0, np.inf)] * 6,
    callback=(
        (
            callback_ir
            if opt.method not in {"TNC", "SLSQP", "COBYLA"}
            else callback_param
        )
        if opt.log
        else None
    ),
)
if opt.log:
    print(res)


s_est = res.x / (rate_constraint_matrix @ res.x)

################################################################################
# jointly optimize GTR params and branch lens using neg-log likelihood


@jit(nopython=True)
def prob_model_helper(t, left, right, evals):
    return ((left * np.exp(t * evals)) @ right).astype(np.float64)


def make_GTR_prob_model(gtr_params):
    pi_a, pi_c, pi_g, pi_t = np.abs(gtr_params[:4])
    s_ac, s_ag, s_at, s_cg, s_ct, s_gt = np.abs(gtr_params[4:])

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

    left = sym_evecs * np.sqrt([pi_a, pi_c, pi_g, pi_t])[:, None]
    right = sym_evecs.T / np.sqrt([pi_a, pi_c, pi_g, pi_t])

    def prob_model_(t):
        return prob_model_helper(t, left, right, evals)

    return prob_model_


################################################################################
################################################################################
################################################################################


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


def compute_score(
    *, root, prob_model, tree_distances, patterns, pattern_counts
) -> float:
    v = compute_score_helper(root, prob_model, tree_distances, patterns)
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


def full_objective(params, gt_norm=False):
    gtr_params = params[:6]
    tree_distances = params[6:]
    return (
        (
            neg_log_likelihood(gtr_params / gtr_params[-1], tree_distances)
            if gt_norm
            else neg_log_likelihood(gtr_params, tree_distances)
        )
        + ((rate_constraint_matrix @ gtr_params) - 1) ** 2
        + tree_distances[0] ** 2
    )


num_func_evals = 0
res = minimize(
    full_objective,
    np.concatenate((s_est, tree_distances)),
    method=opt.method,
    bounds=[(0.0, np.inf)] * (6 + 2 * len(taxa) - 1),
    callback=(
        (
            callback_ir
            if opt.method not in {"TNC", "SLSQP", "COBYLA"}
            else callback_param
        )
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
    rel_error = (
        tree_distances[rel_mask] - true_branch_lens[rel_mask]
    ) / true_branch_lens[rel_mask]
    print(f"min rel error: {np.min(rel_error)}")
    print(f"max rel error: {np.max(rel_error)}")
    print(f"mean rel error: {np.mean(rel_error)}")
    print(f"stdev rel error: {np.std(rel_error)}")


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
