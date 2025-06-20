from typing import Callable

import numba
import numpy as np
from scipy.special import logsumexp

from pruning.util import kahan_dot, log_dot, log_matrix_mult


def compute_leaf_vec(patterns, num_states) -> Callable:
    # print(f"compute_leaf_vector({patterns=})")
    # noinspection PyUnreachableCode
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
        case 16:
            id16 = np.identity(16, dtype=np.float64)
            arr = np.concatenate(
                # fmt: off
                # @formatter:off
                (
                    id16[0:4, :],
                    np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]) / 4,  # A|?
                    id16[4:8, :],
                    np.array([[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]) / 4,  # C|?
                    id16[8:12, :],
                    np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]]) / 4,  # G|?
                    id16[12:16, :],
                    np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]]) / 4,  # T|?
                    np.array([[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]]) / 4,  # ?|A
                    np.array([[0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]]) / 4,  # ?|C
                    np.array([[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0]]) / 4,  # ?|G
                    np.array([[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]]) / 4,  # ?|T
                    [np.full(16, 1 / 16, dtype=np.float64)],  # ?|?
                ),
                # @formatter:on
                # fmt: on
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


def compute_score_function_helper(
    node, patterns, taxa_indices_, num_states, node_indices
) -> Callable:
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

    left_patterns, left_pattern_inverse = np.unique(
        patterns[:, left_leaf_idcs], axis=0, return_inverse=True
    )
    right_patterns, right_pattern_inverse = np.unique(
        patterns[:, right_leaf_idcs], axis=0, return_inverse=True
    )

    if left_node.is_leaf():
        w_l_function = compute_leaf_vec(left_patterns, num_states)
    else:
        w_l_function = compute_score_function_helper(
            left_node, left_patterns, left_taxa_rel_indices, num_states, node_indices
        )

    if right_node.is_leaf():
        w_r_function = compute_leaf_vec(right_patterns, num_states)
    else:
        w_r_function = compute_score_function_helper(
            right_node, right_patterns, right_taxa_rel_indices, num_states, node_indices
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


def compute_score_function(
    *, root, patterns, pattern_counts, num_states, taxa_indices, node_indices
) -> Callable:
    # print(f"compute_score_function({root=},{patterns=},{pattern_counts=})")
    v_function = compute_score_function_helper(
        root, patterns, taxa_indices, num_states, node_indices
    )

    def score_function(log_freq_params, prob_matrices):
        v = v_function(prob_matrices)
        # can't assume that pis are normalized
        log_freq_params_corrected = log_freq_params - logsumexp(log_freq_params)
        return -kahan_dot(pattern_counts, log_dot(v, log_freq_params_corrected))

    return numba.jit(score_function, nopython=False, forceobj=True)
    # return score_function


def compute_factored_score_function(
    *, root, patterns, pattern_counts, num_states, taxa_indices, node_indices
) -> Callable:

    # separate maternal and paternal patterns
    maternal_patterns, paternal_patterns = np.divmod(patterns, 5)

    # deduplicate maternal patterns, combining counts
    reduced_maternal_patterns, maternal_pattern_indices = np.unique(
        maternal_patterns, axis=0, return_inverse=True
    )
    maternal_pattern_counts = np.zeros(shape=reduced_maternal_patterns.shape[0], dtype=np.int64)
    for orig_idx, dedup_idx in enumerate(maternal_pattern_indices):
        maternal_pattern_counts[dedup_idx] += pattern_counts[orig_idx]

    # create maternal score function
    maternal_v_function = compute_score_function_helper(
        root, maternal_patterns, taxa_indices, num_states, node_indices
    )

    def maternal_score_function(log_freq_params, prob_matrices):
        v = maternal_v_function(prob_matrices)
        # can't assume that pis are normalized
        log_freq_params_corrected = log_freq_params - logsumexp(log_freq_params)
        return -kahan_dot(maternal_pattern_counts, log_dot(v, log_freq_params_corrected))

    # deduplicate paternal patterns, combining counts
    reduced_paternal_patterns, paternal_pattern_indices = np.unique(
        paternal_patterns, axis=0, return_inverse=True
    )
    paternal_pattern_counts = np.zeros(shape=reduced_paternal_patterns.shape[0], dtype=np.int64)
    for orig_idx, dedup_idx in enumerate(paternal_pattern_indices):
        paternal_pattern_counts[dedup_idx] += pattern_counts[orig_idx]

    # create paternal score function
    paternal_v_function = compute_score_function_helper(
        root, paternal_patterns, taxa_indices, num_states, node_indices
    )

    def paternal_score_function(log_freq_params, prob_matrices):
        v = paternal_v_function(prob_matrices)
        # can't assume that pis are normalized
        log_freq_params_corrected = log_freq_params - logsumexp(log_freq_params)
        return -kahan_dot(paternal_pattern_counts, log_dot(v, log_freq_params_corrected))

    # joint mat/pat score function
    def score_function(log_freq_params, prob_matrices):
        return maternal_score_function(log_freq_params, prob_matrices) + paternal_score_function(
            log_freq_params, prob_matrices
        )

    return numba.jit(score_function, nopython=False, forceobj=True)


def neg_log_likelihood_prototype(
    log_freq_params,
    model_params,
    tree_distances,
    *,
    prob_model_maker,
    score_function,
):
    prob_model = prob_model_maker(np.exp(log_freq_params), model_params, vec=True)
    prob_matrices = prob_model(tree_distances)
    return score_function(log_freq_params, prob_matrices)
