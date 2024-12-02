#!/usr/bin/env python3
import argparse
import itertools
import os.path
import subprocess
import sys
import tempfile
from collections import defaultdict
from functools import reduce
from typing import Dict, List, Set, Tuple

import numpy as np
import scipy
from ete3 import Tree
from scipy.optimize import LinearConstraint, minimize

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

if hasattr(sys, "ps1"):
    opt = parser.parse_args(
        "--seqs /home/knappa/build-algo/data/1K-ultrametric/data_1_GTR_I_Gamma_1K_sites_ultrametric.phy "
        "--tree /home/knappa/build-algo/data/1K-ultrametric/trees_50_taxa_ultrametric_1.nwk".split()
    )
else:
    opt = parser.parse_args()
print(opt)


def np_full_print(nparray):
    import shutil

    # noinspection PyTypeChecker
    with np.printoptions(
        threshold=np.inf, linewidth=shutil.get_terminal_size((80, 20)).columns
    ):
        print(nparray)


with open(opt.tree, "r") as tree_file:
    true_tree = Tree(tree_file.read().strip())

with open(opt.seqs, "r") as seq_file:
    ntaxa, nsites = map(int, next(seq_file).split())

    sequences = dict()
    for line in seq_file:
        taxon, *seq = line.split()
        seq = "".join(seq)

        assert taxon not in sequences

        sequences[taxon] = seq

    assert ntaxa == len(sequences)

assert (
    set(true_tree.get_leaf_names()) == sequences.keys()
), "not the same leaves! are these matching datasets?"

taxa = sorted(sequences.keys())
taxa_indices = dict(map(lambda pair: pair[::-1], enumerate(taxa)))

# assemble the site pattern count tensor
counts = defaultdict(lambda: 0)
for idx in range(nsites):
    pattern = tuple(
        map(
            lambda taxon: {"A": 0, "C": 1, "G": 2, "T": 3}[
                sequences[taxon][idx].upper()
            ],
            taxa,
        )
    )
    counts[pattern] += 1


def Q_GTR(params) -> np.ndarray:
    pi = params[:4]
    x = params[4:]
    # Julia's
    Q = np.array(
        [
            # fmt: off
            [           0, pi[1] * x[0], pi[2] * x[1], pi[3] * x[2]],
            [pi[0] * x[0],            0, pi[2] * x[3], pi[3] * x[4]],
            [pi[0] * x[1], pi[1] * x[3],            0, pi[3] * x[5]],
            [pi[0] * x[2], pi[1] * x[4], pi[2] * x[5],            0],
            # fmt: on
        ],
        dtype=np.float64,
    )
    Q -= np.diag(np.sum(Q, axis=1))
    return Q


def P(t, model, params) -> np.ndarray:
    return scipy.linalg.expm(t * model(params))


def compute_leaf_vec(node, pattern) -> np.ndarray:
    nuc = pattern[taxa_indices[node.name]]
    e_nuc = np.zeros(4)
    e_nuc[nuc] = 1
    return e_nuc


def compute_pattern_prob_helper(
    pattern, node, model, params, branch_lens
) -> np.ndarray:

    assert len(node.children) == 2
    left_node, right_node = node.children

    if left_node.is_leaf():
        w_l = compute_leaf_vec(left_node, pattern)
    else:
        w_l = compute_pattern_prob_helper(
            pattern, left_node, model, params, branch_lens
        )

    w_l = P(branch_lens[(node.name, left_node.name)], model, params) @ w_l

    if right_node.is_leaf():
        w_r = compute_leaf_vec(right_node, pattern)
    else:
        w_r = compute_pattern_prob_helper(
            pattern, right_node, model, params, branch_lens
        )

    w_r = P(branch_lens[(node.name, right_node.name)], model, params) @ w_r

    v_n = w_l * w_r

    return v_n


def compute_pattern_prob(pattern, root, model, params, branch_lens) -> float:
    v_r = compute_pattern_prob_helper(pattern, root, model, params, branch_lens)
    pi = params[:4]
    return v_r @ pi


def compute_score(root, model, params, branch_lens) -> float:
    score = np.float64(0.0)
    for pattern in counts.keys():
        score += counts[pattern] * np.log(
            compute_pattern_prob(pattern, root, model, params, branch_lens)
        )
        print('.', end='')
    print('done')
    return score


tree = true_tree.copy()


def find_branch_lens(bl, parent, child, parent_name=None, child_name="_i1"):

    if parent is not None:
        if parent.name is not None and len(parent.name) > 0:
            parent_name = parent.name
        else:
            parent.name = parent_name
        if child.name is not None and len(child.name) > 0:
            child_name = child.name
        else:
            child.name = child_name
        bl[(parent_name, child_name)] = child.dist

    if child.is_leaf():
        return

    for idx, grandchild in enumerate(child.children):
        grandchild_name = "_i" + str(2 * int(child_name[2:]) + idx)

        find_branch_lens(bl, child, grandchild, child_name, grandchild_name)


true_branch_lens = dict()
find_branch_lens(true_branch_lens, None, tree)

true_branch_lens_vec = np.array(
    [true_branch_lens[edge] for edge in sorted(true_branch_lens.keys())]
)

# NOTE: this is a placeholder, not the true params
true_params = np.concatenate(
    (
        np.array([0.25, 0.25, 0.25, 0.25]),  # pis
        np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),  # xs
    ),
    axis=0,
)


compute_pattern_prob(
    list(counts.items())[0][0], tree, Q_GTR, true_params, true_branch_lens
)


def make_objective(tree, model, edges, pis):
    def _objective(vec):
        params = np.concatenate((pis, vec[:6]), axis=0)
        branch_lens = dict(zip(edges, vec[6:]))
        return -compute_score(
            tree, model, params, branch_lens
        )  # Julia: minus so we can use minimize

    return _objective


pis = np.array([0.25, 0.25, 0.25, 0.25])
objective = make_objective(tree, Q_GTR, sorted(true_branch_lens.keys()), pis)


dimension = 6 + len(true_branch_lens)
# non_negativity = LinearConstraint(
#     np.identity(dimension), np.zeros(dimension), np.full(dimension, np.inf)
# )
non_negativity = LinearConstraint(
    np.identity(dimension), np.full(dimension, 1e-6), np.full(dimension, 1e3)
)


x0 = np.abs(np.random.normal(size=dimension))
res = minimize(
    objective,
    x0,
    method="trust-constr",
    constraints=[non_negativity],
    options={"verbose": 1},
)
