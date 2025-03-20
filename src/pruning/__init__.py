def main_cli():
    import argparse
    import functools
    import os.path
    import sys
    from collections import defaultdict
    from contextlib import redirect_stdout
    from typing import Callable, Literal

    import numba
    import numpy as np
    from ete3 import Tree
    from scipy.optimize import minimize
    from scipy.special import logsumexp

    from pruning.matrices import (
        U,
        cellphy10_rate,
        gtr4_rate,
        gtr10_rate,
        gtr10z_rate,
        make_A_GTR,
        make_cellphy_prob_model,
        make_gtr10_prob_model,
        make_gtr10z_prob_model,
        make_GTR_prob_model,
        make_unphased_GTR_prob_model,
        perm,
    )
    from pruning.objective_functions import (
        branch_length_objective_prototype,
        full_param_objective_prototype,
        param_objective_prototype,
        params_distances_objective_prototype,
    )
    from pruning.path_constraints import make_path_constraints
    from pruning.util import (
        CallbackIR,
        CallbackParam,
        kahan_dot,
        log_dot,
        log_matrix_mult,
        print_stats,
    )

    model_list = ["DNA", "PHASED_DNA", "UNPHASED_DNA", "CELLPHY", "GTR10Z", "GTR10", "SIEVE"]

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

    if hasattr(sys, "ps1"):
        opt = parser.parse_args(
            "--seqs test/test-diploid.phy --tree test/test-diploid.nwk --model UNPHASED_DNA --log ".split()
        )
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

    model: Literal["DNA", "PHASED_DNA", "UNPHASED_DNA", "CELLPHY", "GTR10Z", "GTR10"] = opt.model

    ambig_char = opt.ambig.upper()
    if len(ambig_char) != 1:
        print("Ambiguity character must be a single character")
        exit(-1)
    if ambig_char in ["A", "C", "G", "T"]:
        print(f"Ambiguity character as '{ambig_char}' is not supported")
        exit(-1)

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

    nsites, num_states, pis, pis10, sequences = read_sequences(ambig_char, model, opt)

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

    tree_distances = compute_initial_tree_distance_estimates(
        model=model,
        node_indices=node_indices,
        num_tree_nodes=num_tree_nodes,
        opt=opt,
        pis=pis,
        sequences=sequences,
        taxa=taxa,
        true_tree=true_tree,
    )

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
            pis_est = pis
            rate_constraint = gtr4_rate
            s_est = np.ones(6)

        case "CELLPHY":

            num_params = 6
            pis_est = pis10
            rate_constraint = cellphy10_rate
            s_est = np.ones(6)

        case "GTR10Z":

            num_params = 24
            pis_est = pis10
            rate_constraint = gtr10z_rate
            s_est = np.ones(24)

        case "GTR10":

            num_params = 45
            pis_est = pis10
            rate_constraint = gtr10_rate
            s_est = np.ones(45)

        case _:
            assert False, "Unknown model type"

    s_est = s_est / rate_constraint(pis_est, s_est)
    log_pis_est = np.log(pis_est)
    num_pis = len(pis_est)

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
                p_l = np.clip(
                    np.log(np.clip(prob_matrices[left_index, :, :], 0.0, 1.0)), -1e100, 0.0
                )
                p_r = np.clip(
                    np.log(np.clip(prob_matrices[right_index, :, :], 0.0, 1.0)), -1e100, 0.0
                )

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

        def score_function(log_pis, prob_matrices):
            v = v_function(prob_matrices)
            log_pis_corrected = log_pis - logsumexp(log_pis)  # can't assume that pis are normalized
            return -kahan_dot(pattern_counts, log_dot(v, log_pis_corrected))

        return numba.jit(score_function, nopython=False, forceobj=True)
        # return score_function

    ################################################################################
    ################################################################################

    match opt.model:
        case "DNA":
            patterns = np.array([pattern for pattern in counts.keys()])
            pattern_counts = np.array([count for count in counts.values()])

            prob_model_maker = make_GTR_prob_model

            def log_pis_modification(log_pis):
                return log_pis

        case "PHASED_DNA":
            genotype_counts = defaultdict(lambda: 0)
            for pattern, count in counts.items():
                pattern_mat = tuple(map(lambda p: p % 5, pattern))
                genotype_counts[pattern_mat] += 1.0

                pattern_pat = tuple(map(lambda p: p // 5, pattern))
                genotype_counts[pattern_pat] += 1.0

            patterns = np.array([pattern for pattern in genotype_counts.keys()])
            pattern_counts = np.array([count for count in genotype_counts.values()])

            prob_model_maker = make_GTR_prob_model

            def log_pis_modification(log_pis):
                return log_pis

        case "UNPHASED_DNA":
            patterns = np.array([pattern for pattern in counts.keys()])
            pattern_counts = np.array([count for count in counts.values()])

            prob_model_maker = make_unphased_GTR_prob_model

            def log_pis_modification(log_pis):
                pis = np.exp(log_pis)
                return np.log(U @ perm @ np.kron(pis, pis))

        case "CELLPHY":
            patterns = np.array([pattern for pattern in counts.keys()])
            pattern_counts = np.array([count for count in counts.values()])

            prob_model_maker = make_cellphy_prob_model

            def log_pis_modification(log_pis):
                return log_pis

        case "GTR10Z":

            patterns = np.array([pattern for pattern in counts.keys()])
            pattern_counts = np.array([count for count in counts.values()])

            prob_model_maker = make_gtr10z_prob_model

            def log_pis_modification(log_pis):
                return log_pis

        case "GTR10":

            patterns = np.array([pattern for pattern in counts.keys()])
            pattern_counts = np.array([count for count in counts.values()])

            prob_model_maker = make_gtr10_prob_model

            def log_pis_modification(log_pis):
                return log_pis

        case _:
            assert False

    def neg_log_likelihood_prototype(
        log_pis,
        model_params,
        tree_distances,
        *,
        prob_model_maker,
        score_function,
        log_pis_modification,
    ):
        prob_model = prob_model_maker(np.exp(log_pis), model_params, vec=True)
        prob_matrices = prob_model(tree_distances)
        return score_function(log_pis_modification(log_pis), prob_matrices)

    neg_log_likelihood = functools.partial(
        neg_log_likelihood_prototype,
        prob_model_maker=prob_model_maker,
        score_function=compute_score_function(
            root=true_tree,
            patterns=patterns,
            pattern_counts=pattern_counts,
            num_states=num_states,
        ),
        log_pis_modification=log_pis_modification,
    )

    ####################################################################################################
    # define optimization objectives for the model parameters and branch lengths

    param_objective = functools.partial(
        param_objective_prototype,
        neg_log_likelihood=neg_log_likelihood,
        rate_constraint=rate_constraint,
    )

    branch_length_objective = functools.partial(
        branch_length_objective_prototype,
        neg_log_likelihood=neg_log_likelihood,
    )

    full_param_objective = functools.partial(
        full_param_objective_prototype,
        neg_log_likelihood=neg_log_likelihood,
        rate_constraint=rate_constraint,
        num_pis=num_pis,
    )

    for _ in range(2):
        res = minimize(
            param_objective,
            s_est,
            args=(
                log_pis_est,
                tree_distances,
            ),
            method=opt.method,
            bounds=[(1e-10, np.inf)] * num_params,
            callback=(
                (CallbackIR() if opt.method not in {"TNC", "SLSQP", "COBYLA"} else CallbackParam())
                if opt.log
                else None
            ),
            options={"maxiter": 1000, "maxfun": 100_000, "ftol": 1e-10},
        )
        if opt.log:
            print(res)

        s_est = res.x / rate_constraint(np.exp(log_pis_est), res.x)  # fine tune mu

        res = minimize(
            branch_length_objective,
            tree_distances,
            args=(log_pis_est, s_est),
            method=opt.method,
            bounds=[(0.0, np.inf)] + [(1e-8, np.inf)] * (2 * len(taxa) - 2),
            callback=(
                (CallbackIR() if opt.method not in {"TNC", "SLSQP", "COBYLA"} else CallbackParam())
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
            res = minimize(
                full_param_objective,
                np.concatenate((log_pis_est, s_est)),
                args=(tree_distances,),
                method=opt.method,
                bounds=([(-np.inf, 0.0)] * num_pis) + ([(1e-10, np.inf)] * num_params),
                callback=(
                    (
                        CallbackIR()
                        if opt.method not in {"TNC", "SLSQP", "COBYLA"}
                        else CallbackParam()
                    )
                    if opt.log
                    else None
                ),
                options={"maxiter": 1000, "maxfun": 100_000, "ftol": 1e-10},
            )
            if opt.log:
                print(res)

            log_pis_est = np.minimum(0.0, res.x[:num_pis])
            log_pis_est -= logsumexp(log_pis_est)  # fine tune prob dist
            pis_est = np.exp(log_pis_est)

            s_est = np.maximum(0.0, res.x[num_pis:])
            s_est = s_est / rate_constraint(np.exp(log_pis_est), s_est)  # fine tune mu

    ####################################################################################################

    if opt.optimize_freqs:

        from pruning.objective_functions import full_objective_prototype

        full_objective = functools.partial(
            full_objective_prototype,
            num_pis=num_pis,
            num_params=num_params,
            neg_log_likelihood=neg_log_likelihood,
            rate_constraint=rate_constraint,
        )

        res = minimize(
            full_objective,
            np.concatenate((log_pis_est, s_est, tree_distances)),
            method=opt.method,
            bounds=([(-np.inf, 0.0)] * num_pis)
            + [(0.0, np.inf)] * (num_params + 2 * len(taxa) - 1),
            callback=(
                (CallbackIR() if opt.method not in {"TNC", "SLSQP", "COBYLA"} else CallbackParam())
                if opt.log
                else None
            ),
            options={"maxiter": 1000, "maxfun": 100_000, "ftol": 1e-10},
        )
        if opt.log:
            print(res)

        log_pis_est = np.minimum(0.0, res.x[:num_pis])
        log_pis_est -= logsumexp(log_pis_est)  # fine tune prob dist
        pis_est = np.exp(log_pis_est)

        s_est = np.maximum(0.0, res.x[num_pis : num_pis + num_params])
        s_est = s_est / rate_constraint(np.exp(log_pis_est), s_est)  # fine tune mu

        tree_distances = np.maximum(0.0, res.x[num_pis + num_params :])

    else:
        # optimize everything but the state frequencies
        params_distances_objective = functools.partial(
            params_distances_objective_prototype,
            num_params=num_params,
            neg_log_likelihood=neg_log_likelihood,
            rate_constraint=rate_constraint,
        )

        res = minimize(
            params_distances_objective,
            np.concatenate((s_est, tree_distances)),
            args=(log_pis_est,),
            method=opt.method,
            bounds=[(0.0, np.inf)] * (num_params + 2 * len(taxa) - 1),
            callback=(
                (CallbackIR() if opt.method not in {"TNC", "SLSQP", "COBYLA"} else CallbackParam())
                if opt.log
                else None
            ),
            options={"maxiter": 1000, "maxfun": 100_000, "ftol": 1e-10},
        )
        if opt.log:
            print(res)

        s_est = np.maximum(0.0, res.x[:num_params])
        s_est = s_est / rate_constraint(np.exp(log_pis_est), s_est)  # fine tune mu
        tree_distances = np.maximum(0.0, res.x[num_params:])

    ################################################################################
    # update branch lens in ETE3 tree

    for idx, node in enumerate(true_tree.traverse()):
        node.dist = tree_distances[idx]

    ################################################################################
    # write tree and statistics to stdout or a file, depending up command line opts

    newick_rep = true_tree.write(format=5)

    if hasattr(opt, "output") and opt.output is not None:
        with open(opt.output + ".nwk", "w") as file:
            file.write(newick_rep)
            file.write("\n")

        with open(opt.output + ".log", "w") as file:
            with redirect_stdout(file):
                print_stats(
                    s_est=s_est,
                    pis_est=pis_est,
                    neg_l=res.fun,
                    tree_distances=tree_distances,
                    true_branch_lens=true_branch_lens,
                    model=model,
                )

    else:
        print(newick_rep)
        print()
        print_stats(
            s_est=s_est,
            pis_est=pis_est,
            neg_l=res.fun,
            tree_distances=tree_distances,
            true_branch_lens=true_branch_lens,
            model=model,
        )

    ################################################################################
    ################################################################################
    ################################################################################
    ################################################################################


def compute_initial_tree_distance_estimates(
    *, model, node_indices, num_tree_nodes, opt, pis, sequences, taxa, true_tree
):
    import functools
    import itertools

    import numpy as np
    from scipy.optimize import minimize

    from pruning.path_constraints import make_path_constraints
    from pruning.util import CallbackIR, CallbackParam

    match model:
        case "DNA":
            from pruning.distance_functions import dna_sequence_distance

            sequence_distance = functools.partial(dna_sequence_distance, pis=pis)
        case "PHASED_DNA":
            from pruning.distance_functions import phased_sequence_distance

            sequence_distance = functools.partial(phased_sequence_distance, pis=pis)
        case "UNPHASED_DNA" | "CELLPHY" | "CELLPHY_PI" | "GTR10Z" | "GTR10":
            # TODO: the others (cellphy, etc.) are included here as they are 10 state, not because this is a natural
            #  distance under their model
            from pruning.distance_functions import unphased_sequence_distance

            sequence_distance = functools.partial(unphased_sequence_distance, pis=pis)
        case _:
            raise NotImplementedError(f"Model {model} not implemented")
            # TODO: SIEVE model
    # Compute all pairwise leaf distances
    pair_to_idx = {
        (n1, n2) if n1 < n2 else (n2, n1): idx
        for idx, (n1, n2) in enumerate(itertools.combinations(taxa, 2))
    }
    leaf_stats = np.array(
        [
            sequence_distance(sequences[n1], sequences[n2])
            for n1, n2 in itertools.combinations(taxa, 2)
        ]
    )
    leaf_distances = leaf_stats[:, 0]
    leaf_variances = leaf_stats[:, 1]
    # create the constraint matrices
    constraints_eqn, constraints_val = make_path_constraints(
        true_tree, num_tree_nodes, leaf_distances, pair_to_idx, node_indices
    )

    def branch_length_estimate_objective_prototype(
        x, *, constraints_eqn, constraints_val, leaf_variances
    ):
        # prediction error (weighted by variance) plus a regularizing term
        prediction_err = constraints_eqn @ x - constraints_val
        return np.mean(prediction_err**2 / np.concatenate(([1], leaf_variances))) + np.mean(x**2)

    branch_length_estimate_objective = functools.partial(
        branch_length_estimate_objective_prototype,
        constraints_eqn=constraints_eqn,
        constraints_val=constraints_val,
        leaf_variances=leaf_variances,
    )
    res = minimize(
        branch_length_estimate_objective,
        np.ones(num_tree_nodes, dtype=np.float64),
        bounds=[(0.0, None)] + [(1e-10, None)] * (num_tree_nodes - 1),
        callback=CallbackParam() if opt.log else None,
    )
    if not res.success:
        print("Error in optimization, continuing anyway", flush=True)
    # belt and suspenders for the constraint (avoid -1e-big type bounds violations)
    tree_distances = np.maximum(0.0, res.x)

    return tree_distances


def read_sequences(ambig_char, model, opt):
    import numpy as np

    from pruning.matrices import U, perm

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
    phased_joint_freq_counts = None
    unphased_joint_freq_counts = None
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
                phased_joint_freq_counts = np.zeros(16, dtype=np.int64)
                # parse sequences
                sequences = dict()
                for line in seq_file:
                    taxon, *seq = line.strip().split()
                    seq = list(map(lambda s: s.upper(), seq))
                    assert all(len(s) == 2 for s in seq)
                    # compute nucleotide frequency
                    for nuc_pair in seq:
                        if nuc_pair[0] == ambig_char:
                            if nuc_pair[1] == ambig_char:
                                for idx in range(16):
                                    phased_joint_freq_counts[idx] += 1 / 16
                            else:
                                for idx in range(4):
                                    phased_joint_freq_counts[4 * idx + nuc_to_idx[nuc_pair[1]]] += (
                                        1 / 4
                                    )
                        else:
                            if nuc_pair[1] == ambig_char:
                                for idx in range(4):
                                    phased_joint_freq_counts[4 * nuc_to_idx[nuc_pair[0]] + idx] += (
                                        1 / 4
                                    )
                            else:
                                phased_joint_freq_counts[
                                    4 * nuc_to_idx[nuc_pair[0]] + nuc_to_idx[nuc_pair[1]]
                                ] += 1

                        for nuc in nuc_pair:
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
            case "UNPHASED_DNA" | "CELLPHY" | "GTR10Z" | "GTR10":
                num_states = 10
                unphased_joint_freq_counts = np.zeros(10, dtype=np.int64)
                sequences = dict()
                for line in seq_file:
                    taxon, *seq = line.strip().split()
                    seq = list(map(lambda s: s.upper(), seq))
                    assert all(len(s) == 2 for s in seq)
                    for nuc_pair in seq:
                        joint_pair_index = unphased_nuc_to_idx[nuc_pair]
                        if joint_pair_index < 10:
                            unphased_joint_freq_counts[joint_pair_index] += 1
                        elif joint_pair_index == 10:
                            for idx, weight in enumerate(
                                [
                                    1 / 16,
                                    1 / 16,
                                    1 / 16,
                                    1 / 16,
                                    2 / 16,
                                    2 / 16,
                                    2 / 16,
                                    2 / 16,
                                    2 / 16,
                                    2 / 16,
                                ]
                            ):
                                unphased_joint_freq_counts[idx] += weight
                        elif joint_pair_index == 11:
                            for idx, weight in enumerate(
                                [1 / 4, 0, 0, 0, 1 / 4, 1 / 4, 1 / 4, 0, 0, 0]
                            ):
                                unphased_joint_freq_counts[idx] += weight
                        elif joint_pair_index == 12:
                            for idx, weight in enumerate(
                                [0, 1 / 4, 0, 0, 1 / 4, 0, 0, 1 / 4, 1 / 4, 0]
                            ):
                                unphased_joint_freq_counts[idx] += weight
                        elif joint_pair_index == 13:
                            for idx, weight in enumerate(
                                [0, 0, 1 / 4, 0, 0, 1 / 4, 0, 1 / 4, 0, 1 / 4]
                            ):
                                unphased_joint_freq_counts[idx] += weight
                        elif joint_pair_index == 14:
                            for idx, weight in enumerate(
                                [0, 0, 0, 1 / 4, 0, 0, 1 / 4, 0, 1 / 4, 1 / 4]
                            ):
                                unphased_joint_freq_counts[idx] += weight

                        for nuc in nuc_pair:
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
    # aggregate the base frequencies
    pis = base_freq_counts / np.sum(base_freq_counts)
    if phased_joint_freq_counts is None:
        pis16 = np.kron(pis, pis)
    else:
        pis16 = phased_joint_freq_counts / np.sum(phased_joint_freq_counts)
    if unphased_joint_freq_counts is None:
        pis10 = U @ perm @ pis16
    else:
        pis10 = unphased_joint_freq_counts / np.sum(unphased_joint_freq_counts)

    return nsites, num_states, pis, pis10, sequences
