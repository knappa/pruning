#!/usr/bin/env python3
import argparse
import itertools
import sys
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import scipy
from ete3 import Tree
from numba import jit
from scipy.optimize import OptimizeResult, minimize

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

if hasattr(sys, "ps1"):
    opt = parser.parse_args(
        # "--seqs /home/knappa/build-algo/data/1K-ultrametric/data_1_GTR_I_Gamma_1K_sites_ultrametric.phy "
        # "--tree /home/knappa/build-algo/data/1K-ultrametric/trees_50_taxa_ultrametric_1.nwk".split()
        # "--seqs /home/knappa/build-algo/data/100K-ultrametric/data_1_GTR_I_Gamma_100K_sites_ultrametric.phy "
        # "--tree /home/knappa/build-algo/data/100K-ultrametric/trees_50_taxa_ultrametric_1.nwk".split()
        "--seqs /home/knappa/test.phy --tree /home/knappa/test.nwk".split()
    )
else:
    opt = parser.parse_args()
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
        node.name = "_i" + str(idx)

# traverse the tree, assigning indices to each node
node_indices = {
    node.name: idx for idx, node in enumerate(true_tree.traverse("levelorder"))
}

################################################################################
# read the sequence data and compute nucleotide frequencies
nuc_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
freq_counts = np.zeros(4, dtype=np.int64)

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


def sequence_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
    # F81 distance, a generalization of the JC69 distance, which takes into
    # account nucleotide frequencies
    beta = 1 / (1 - np.sum(pis**2))
    disagreement = np.sum(seq1 != seq2) / len(seq1)
    # TODO: bounds check DONE?
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
# constrain root edge to 0 length on root as this isn't a real edge length
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
    callback=callback_param,
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

# initially, guess F81 model (all s parameters == 1)
s_est_0 = np.ones(6)
leaf_to_leaf_distances = constraints_eqn[1:] @ tree_distances


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


def objective(s_est):
    p_ts = scipy.linalg.expm(
        leaf_to_leaf_distances[:, None, None]
        * (A_GTR @ s_est).reshape(4, 4)[None, :, :]
    )
    joint_dist = reduced_pattern[:, None, :] * p_ts
    # compute KL
    return np.mean(
        np.sum(
            np.nan_to_num(
                scipy.special.xlogy(count_patterns, count_patterns), neginf=-1e10
            )
            - np.nan_to_num(
                scipy.special.xlogy(count_patterns, joint_dist), neginf=-1e10
            ),
            axis=(1, 2),
        )
    )


num_func_evals = 0
res = minimize(
    objective,
    s_est_0,
    method=opt.method,
    bounds=[(0.0, np.inf)] * 6,
    callback=(
        callback_ir if opt.method not in {"TNC", "SLSQP", "COBYLA"} else callback_param
    ),
)
print(res)


s_est = res.x

################################################################################
# joint edge length and GTR parameter optimization

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

s_est /= rate_constraint_matrix @ s_est


def objective(params):
    s_est = params[:6]
    tree_distances = params[6:]
    leaf_to_leaf_distances = constraints_eqn[1:] @ tree_distances
    p_ts = scipy.linalg.expm(
        leaf_to_leaf_distances[:, None, None]
        * (A_GTR @ s_est).reshape(4, 4)[None, :, :]
    )
    joint_dist = reduced_pattern[:, None, :] * p_ts
    # compute KL
    return (
        np.mean(
            np.sum(
                np.nan_to_num(
                    scipy.special.xlogy(count_patterns, count_patterns), neginf=-1e10
                )
                - np.nan_to_num(
                    scipy.special.xlogy(count_patterns, joint_dist), neginf=-1e10
                ),
                axis=(1, 2),
            )
        )
        + (rate_constraint_matrix @ s_est - 1) ** 2
    )


num_func_evals = 0
res = minimize(
    objective,
    np.concatenate((s_est, tree_distances)),
    method=opt.method,
    bounds=[(0.0, np.inf)] * (6 + tree_distances.shape[0]),
    callback=(
        callback_ir if opt.method not in {"TNC", "SLSQP", "COBYLA"} else callback_param
    ),
)
print(res)

s_est = res.x[:6]
tree_distances = res.x[6:]

################################################################################
# optimize GTR params


################################################################################
################################################################################


@jit(nopython=True)
def prob_model_helper(t, left, right, evals):
    return ((left * np.exp(t * evals)) @ right).astype(np.float64)


def make_GTR_prob_model(gtr_params):
    pa, pc, pg, pt = np.abs(gtr_params[:4])
    qac, qag, qat, qcg, qct, qgt = np.abs(gtr_params[4:])

    sym_Q = np.array(
        [
            [
                -(pc * qac + pg * qag + pt * qat),
                np.sqrt(pa * pc) * qac,
                np.sqrt(pa * pg) * qag,
                np.sqrt(pa * pt) * qat,
            ],
            [
                np.sqrt(pa * pc) * qac,
                -(pa * qac + pg * qcg + pt * qct),
                np.sqrt(pc * pg) * qcg,
                np.sqrt(pc * pt) * qct,
            ],
            [
                np.sqrt(pa * pg) * qag,
                np.sqrt(pc * pg) * qcg,
                -(pa * qag + pc * qcg + pt * qgt),
                np.sqrt(pg * pt) * qgt,
            ],
            [
                np.sqrt(pa * pt) * qat,
                np.sqrt(pc * pt) * qct,
                np.sqrt(pg * pt) * qgt,
                -(pa * qat + pc * qct + pg * qgt),
            ],
        ],
        dtype=np.float64,
    )

    evals, sym_evecs = np.linalg.eigh(sym_Q)

    left = sym_evecs * np.sqrt([pa, pc, pg, pt])[:, None]
    right = sym_evecs.T / np.sqrt([pa, pc, pg, pt])

    def prob_model_(t):
        return prob_model_helper(np.abs(t), left, right, evals)

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
    w_l = np.einsum("ij,nj->ni", prob_model(t), w_l)

    if right_node.is_leaf():
        w_r = compute_leaf_vec(right_node, patterns)
    else:
        w_r = compute_score_helper(right_node, prob_model, tree_distances, patterns)

    t = tree_distances[node_indices[right_node.name]]
    w_r = np.einsum("ij,nj->ni", prob_model(t), w_r)

    v_n = w_l * w_r

    return v_n


def compute_score(
    *, root, prob_model, tree_distances, patterns, pattern_counts
) -> float:
    v = compute_score_helper(root, prob_model, tree_distances, patterns)
    return pattern_counts @ np.nan_to_num(np.log(v @ pis))


def gtr_objective(gtr_params, tree_distances):
    gtr_prob_model = make_GTR_prob_model(np.concatenate((pis, gtr_params)))
    patterns = np.array([pattern for pattern in counts.keys()])
    pattern_counts = np.array([count for count in counts.values()])

    return -compute_score(
        root=true_tree,
        prob_model=gtr_prob_model,
        tree_distances=tree_distances,
        patterns=patterns,
        pattern_counts=pattern_counts,
    )  # + (rate_constraint_matrix @ gtr_params - 1) ** 2


def full_objective(params):
    rate_constraint_err = (rate_constraint_matrix @ params[:6] - 1) ** 2
    return rate_constraint_err + gtr_objective(params[:6], params[6:])
    # x = params[6:]
    # prediction_err = constraints_eqn @ x - constraints_val
    # branch_len_err = np.mean(x**2) + np.mean(prediction_err ** 2)
    # return branch_len_err + gtr_objective(params[:6],params[6:])


# res = minimize(
#     gtr_objective,
#     s_est / (rate_constraint_matrix @ s_est),
#     args=tree_distances,
#     method=opt.method,
#     bounds=[(0.0, np.inf)] * 6,
#     # constraints=[rate_constraint],
#     # options={"verbose": 1},
#     callback=(
#         callback_ir if opt.method not in {"TNC", "SLSQP", "COBYLA"} else callback_param
#     ),
# )
# print(res)

new_s_est = res.x / (rate_constraint_matrix @ res.x)

res = minimize(
    full_objective,
    np.concatenate((new_s_est, tree_distances)),
    method=opt.method,
    bounds=[(0.0, np.inf)] * (6 + 99),
    # constraints=[rate_constraint],
    # options={"verbose": 1},
    callback=(
        callback_ir if opt.method not in {"TNC", "SLSQP", "COBYLA"} else callback_param
    ),
)
print(res)


################################################################################
################################################################################
################################################################################
################################################################################
################################################################################

#
# # noinspection PyShadowingNames
# def compute_leaf_vec(node, pattern) -> np.ndarray:
#     nuc = pattern[taxa_indices[node.name]]
#     e_nuc = np.zeros(4)
#     e_nuc[nuc] = 1
#     return e_nuc
#
#
# # noinspection PyShadowingNames
# def compute_pattern_prob_helper(pattern, node, prob_model, branch_lens) -> np.ndarray:
#     assert len(node.children) == 2
#     left_node, right_node = node.children
#
#     if left_node.is_leaf():
#         w_l = compute_leaf_vec(left_node, pattern)
#     else:
#         w_l = compute_pattern_prob_helper(pattern, left_node, prob_model, branch_lens)
#
#     w_l = left @ np.diag(np.exp(t_left * eigs)) @ right @ w_l
#
#     if right_node.is_leaf():
#         w_r = compute_leaf_vec(right_node, pattern)
#     else:
#         w_r = compute_pattern_prob_helper(pattern, right_node, prob_model, branch_lens)
#
#     w_r = left @ np.diag(np.exp(t_right * eigs)) @ right @ w_r
#
#     v_n = w_l * w_r
#
#     return v_n
#
#
# # noinspection PyShadowingNames
# def compute_pattern_prob(*, pattern, root, prob_model, params, branch_lens) -> float:
#     v_r = compute_pattern_prob_helper(pattern, root, prob_model, branch_lens)
#     pi = params[:4]
#     return v_r @ pi
#
#
# def compute_score(*, root, prob_model, params, branch_lens) -> float:
#     score = np.float64(0.0)
#     # noinspection PyShadowingNames
#     for pattern in counts.keys():
#         score += counts[pattern] * np.log(
#             compute_pattern_prob(
#                 pattern=pattern,
#                 root=root,
#                 prob_model=prob_model,
#                 params=params,
#                 branch_lens=branch_lens,
#             )
#         )
#     # print(".", end="", flush=True)
#     return score
#
#
# tree = true_tree.copy()
#
#
# def find_branch_lens(bl, parent, child, parent_name=None, child_name="_i1"):
#     if parent is not None:
#         if parent.name is not None and len(parent.name) > 0:
#             parent_name = parent.name
#         else:
#             parent.name = parent_name
#         if child.name is not None and len(child.name) > 0:
#             child_name = child.name
#         else:
#             child.name = child_name
#         bl[(parent_name, child_name)] = child.dist
#
#     if child.is_leaf():
#         return
#
#     # noinspection PyShadowingNames
#     for idx, grandchild in enumerate(child.children):
#         grandchild_name = "_i" + str(2 * int(child_name[2:]) + idx)
#
#         find_branch_lens(bl, child, grandchild, child_name, grandchild_name)
#
#
# true_branch_lens = dict()
# find_branch_lens(true_branch_lens, None, tree)
#
# true_branch_lens_vec = np.array(
#     [true_branch_lens[edge] for edge in sorted(true_branch_lens.keys())]
# )
#
# branch_to_param_idx = {
#     branch: 6 + idx for idx, branch in enumerate(sorted(true_branch_lens.keys()))
# }
#
# rate_constraint_matrix = np.array(
#     [
#         2 * pi_a * pi_c,
#         2 * pi_a * pi_g,
#         2 * pi_a * pi_t,
#         2 * pi_c * pi_g,
#         2 * pi_c * pi_t,
#         2 * pi_g * pi_t,
#     ]
#     + [0] * len(true_branch_lens)
# )
#
# rate_constraint = LinearConstraint(rate_constraint_matrix, lb=1, ub=1)
#
#
# # noinspection PyShadowingNames
# def make_objective(*, tree, model_maker, edges, pis, constraints=False):
#     if constraints:
#
#         def _objective(vec):
#             params = np.concatenate((pis, vec[:6]), axis=0)
#             branch_lens = dict(zip(edges, vec[6:]))
#             prob_model = model_maker(params)
#             return (
#                 -compute_score(
#                     root=tree,
#                     prob_model=prob_model,
#                     params=params,
#                     branch_lens=branch_lens,
#                 )
#                 + (rate_constraint_matrix @ vec - 1) ** 2
#                 + np.sum(np.minimum(-vec, 0.0)) ** 2
#             )
#
#         return _objective
#
#     else:
#
#         def _objective(vec):
#             params = np.concatenate((pis, vec[:6]), axis=0)
#             branch_lens = dict(zip(edges, vec[6:]))
#             prob_model = model_maker(params)
#             return -compute_score(
#                 root=tree, prob_model=prob_model, params=params, branch_lens=branch_lens
#             )  # Julia: minus so we can use minimize
#
#         return _objective
#
#
# # noinspection PyPep8Naming
# def Q_GTR(params) -> np.ndarray:
#     pi = np.abs(params[:4])
#     x = np.abs(params[4:])
#     # Julia's
#     Q = np.array(
#         [
#             # fmt: off
#             [0, pi[1] * x[0], pi[2] * x[1], pi[3] * x[2]],
#             [pi[0] * x[0], 0, pi[2] * x[3], pi[3] * x[4]],
#             [pi[0] * x[1], pi[1] * x[3], 0, pi[3] * x[5]],
#             [pi[0] * x[2], pi[1] * x[4], pi[2] * x[5], 0],
#             # fmt: on
#         ],
#         dtype=np.float64,
#     )
#     Q -= np.diag(np.sum(Q, axis=1))
#     return Q
#
#
# # noinspection PyPep8Naming
# def P(t, model, params) -> np.ndarray:
#     return scipy.linalg.expm(np.abs(t) * model(params))
#
#
# @jit(nopython=True)
# def prob_model_helper(t, left, right, evals):
#     return ((left * np.exp(t * evals)) @ right).astype(np.float64)
#
#
# def make_GTR_prob_model(params):
#     Q = Q_GTR(params)
#     evals, evecs = np.linalg.eig(Q)
#     left = evecs
#     right = np.linalg.pinv(evecs)
#
#     def prob_model_(t):
#         return prob_model_helper(np.abs(t), left, right, evals)
#
#     return prob_model_
#
#
# def make_GTR_prob_model_v2(params):
#     pa, pc, pg, pt = np.abs(params[:4])
#     qac, qag, qat, qcg, qct, qgt = np.abs(params[4:])
#
#     sym_Q = np.array(
#         [
#             [
#                 -(pc * qac + pg * qag + pt * qat),
#                 np.sqrt(pa * pc) * qac,
#                 np.sqrt(pa * pg) * qag,
#                 np.sqrt(pa * pt) * qat,
#             ],
#             [
#                 np.sqrt(pa * pc) * qac,
#                 -(pa * qac + pg * qcg + pt * qct),
#                 np.sqrt(pc * pg) * qcg,
#                 np.sqrt(pc * pt) * qct,
#             ],
#             [
#                 np.sqrt(pa * pg) * qag,
#                 np.sqrt(pc * pg) * qcg,
#                 -(pa * qag + pc * qcg + pt * qgt),
#                 np.sqrt(pg * pt) * qgt,
#             ],
#             [
#                 np.sqrt(pa * pt) * qat,
#                 np.sqrt(pc * pt) * qct,
#                 np.sqrt(pg * pt) * qgt,
#                 -(pa * qat + pc * qct + pg * qgt),
#             ],
#         ],
#         dtype=np.float64,
#     )
#
#     evals, sym_evecs = np.linalg.eigh(sym_Q)
#
#     left = sym_evecs * np.sqrt([pa, pc, pg, pt])[:, None]
#     right = sym_evecs.T / np.sqrt([pa, pc, pg, pt])
#
#     def prob_model_(t):
#         return prob_model_helper(np.abs(t), left, right, evals)
#
#     return prob_model_
#
#
# objective = make_objective(
#     tree=tree,
#     model_maker=make_GTR_prob_model_v2,
#     edges=sorted(true_branch_lens.keys()),
#     pis=pis,
#     constraints=True,
# )
#
# dimension = 6 + len(true_branch_lens)
#
# num_func_evals = 0
#
#
# def callback_param(x):
#     global num_func_evals
#     num_func_evals += 1
#     print(num_func_evals, flush=True)
#     np_full_print(x)
#
#
# def callback_ir(intermediate_result: OptimizeResult):
#     global num_func_evals
#     num_func_evals += 1
#     print(num_func_evals, flush=True)
#     print(intermediate_result, flush=True)
#     np_full_print(intermediate_result.x)
#
#
# def callback_annealing(x, f, context):
#     global num_func_evals
#     num_func_evals += 1
#     print(num_func_evals, flush=True)
#     np_full_print(x)
#     np_full_print(f)
#     np_full_print(context)
#
#
# # res = dual_annealing(
# #     objective,
# #     # method=opt.method,
# #     bounds=[(1e-6, 5)] * dimension,
# #     # constraints=[rate_constraint],
# #     # options={"verbose": 1},
# #     callback=callback_annealing,
# # )
# # print(res)
#
#
# # res = differential_evolution(
# #     objective,
# #     # method=opt.method,
# #     bounds=[(0.0, 10)] * dimension,
# #     constraints=[rate_constraint],
# #     # options={"verbose": 1},
# #     callback=(
# #         callback_ir if opt.method not in {"TNC", "SLSQP", "COBYLA"} else callback_param
# #     ),
# # )
# # print(res)
#
# x0 = np.abs(np.random.normal(size=dimension))
# # noinspection PyTypeChecker
# res = minimize(
#     objective,
#     x0,
#     method=opt.method,
#     bounds=[(0.0, np.inf)] * dimension,
#     # constraints=[rate_constraint],
#     # options={"verbose": 1},
#     callback=(
#         callback_ir if opt.method not in {"TNC", "SLSQP", "COBYLA"} else callback_param
#     ),
# )
# print(res)
