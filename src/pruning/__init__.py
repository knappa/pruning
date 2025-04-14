def main_cli():
    import argparse
    import functools
    import os.path
    import sys
    from collections import defaultdict
    from contextlib import redirect_stdout
    from typing import Literal

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
        make_GTRsq_prob_model,
        make_GTRxGTR_prob_model,
        make_unphased_GTRsq_prob_model,
        perm,
        phased_mp_rate,
        phased_rate,
        pi4s_to_unphased_pi10s,
        pi10s_to_pi4s,
        unphased_freq_param_cleanup,
        unphased_rate,
    )
    from pruning.objective_functions import (
        branch_length_objective_prototype,
        param_objective_prototype,
        rate_param_objective_prototype,
        rates_distances_objective_prototype,
    )
    from pruning.path_constraints import make_path_constraints
    from pruning.score_function_gen import compute_score_function, neg_log_likelihood_prototype
    from pruning.util import (
        CallbackIR,
        CallbackParam,
        kahan_dot,
        log_dot,
        log_matrix_mult,
        print_stats,
        rate_param_cleanup,
    )

    model_list = [
        "DNA",
        "PHASED_DNA4",
        "PHASED_DNA16",
        "PHASED_DNA16_MP",
        "UNPHASED_DNA",
        "CELLPHY",
        "GTR10Z",
        "GTR10",
        "SIEVE",
    ]

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

    ################################################################################
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--optimize_freq_params",
        action="store_true",
        help="optimize frequency parameters using maximum likelihood",
    )
    group.add_argument(
        "--freq_params_from_seq",
        action="store_true",
        help="Estimate root distribution from sequence data (default)",
    )
    group.add_argument(
        "--fix_freq_params4",
        nargs=4,
        metavar=("pi_a", "pi_c", "pi_g", "pi_t"),
        help="",
        type=float,
        default=None,
    )
    group.add_argument(
        "--fix_freq_params10",
        nargs=10,
        # fmt: off
        # @formatter:off
        metavar=(
            "pi_aa", "pi_cc", "pi_gg", "pi_tt",
            "pi_ac", "pi_ag", "pi_at",
            "pi_cg", "pi_ct",
            "pi_gt",
        ),
        # @formatter:on
        # fmt: on
        help="",
        type=float,
        default=None,
    )
    group.add_argument(
        "--fix_freq_params16",
        nargs=16,
        # fmt: off
        # @formatter:off
        metavar=(
            "pi_aa", "pi_ac", "pi_ag", "pi_at",
            "pi_ca", "pi_cc", "pi_cg", "pi_ct",
            "pi_ga", "pi_gc", "pi_gg", "pi_gt",
            "pi_ta", "pi_tc", "pi_tg", "pi_tt",
        ),
        # @formatter:on
        # fmt: on
        help="",
        type=float,
        default=None,
    )
    ################################################################################

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

    model: Literal[
        "DNA",
        "PHASED_DNA4",
        "PHASED_DNA16",
        "PHASED_DNA16_MP",
        "UNPHASED_DNA",
        "CELLPHY",
        "GTR10Z",
        "GTR10",
    ] = opt.model

    match model:
        case "DNA" | "PHASED_DNA4":
            ploidy = 1
        case "PHASED_DNA16" | "PHASED_DNA16_MP" | "UNPHASED_DNA" | "CELLPHY" | "GTR10Z" | "GTR10":
            ploidy = 2
        case _:
            raise NotImplementedError("Unknown model")

    ambig_char = opt.ambig.upper()
    if len(ambig_char) != 1:
        print("Ambiguity character must be a single character")
        exit(-1)
    if ambig_char in ["A", "C", "G", "T"]:
        print(f"Ambiguity character as '{ambig_char}' is not supported")
        exit(-1)

    freq_params = None
    if opt.optimize_freq_params:
        freq_params_option = "OPTIMIZE"
    elif opt.freq_params_from_seq:
        freq_params_option = "FROM_SEQ"
    elif opt.fix_freq_params4:
        freq_params_option = "FIX4"
        freq_params = np.array(opt.fix_freq_params4)
    elif opt.fix_freq_params10:
        freq_params_option = "FIX10"
        freq_params = np.array(opt.fix_freq_params10)
    elif opt.fix_freq_params16:
        freq_params_option = "FIX16"
        freq_params = np.array(opt.fix_freq_params16)
    else:
        freq_params_option = "FROM_SEQ"

    ################################################################################
    # solver options

    solver_options = defaultdict(dict)
    solver_options["L-BFGS-B"] = {"maxiter": 1000, "maxfun": 100_000, "ftol": 1e-10}
    solver_options["Powell"] = {"maxiter": 1000, "ftol": 1e-10}
    solver_options["Nelder-Mead"] = {"adaptive": True, "fatol": 0.1}

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

    nsites, num_states, seq_pis, seq_pis10, seq_pis16, sequences = read_sequences(
        ambig_char, model, opt.seqs
    )

    if freq_params_option not in {"FIX4", "FIX10", "FIX16"}:
        # if we are deriving this from sequence data, this is the correct value;
        # if we are optimizing, this is a good initial seed value
        match model:
            case "DNA" | "PHASED_DNA4":
                freq_params = seq_pis
            case "UNPHASED_DNA" | "CELLPHY" | "GTR10Z" | "GTR10":
                freq_params = seq_pis10
                if model == "UNPHASED_DNA":
                    freq_params = unphased_freq_param_cleanup(freq_params)
            case "PHASED_DNA16" | "PHASED_DNA16_MP":
                freq_params = seq_pis16
            case _:
                raise NotImplementedError("Missing a model?")

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

    branch_lengths = compute_initial_tree_distance_estimates(
        model=model,
        node_indices=node_indices,
        num_tree_nodes=num_tree_nodes,
        opt=opt,
        pis=seq_pis,
        sequences=sequences,
        taxa=taxa,
        true_tree=true_tree,
    )

    if opt.log:
        print("Branch length estimate:")
        print(branch_lengths)

    # collect true tree data for comparison, likely on a different scale (GT transversion?), so not directly
    # comparable, but possibly good to have
    true_branch_lens = np.zeros(num_tree_nodes, dtype=np.float64)
    for node in true_tree.traverse():
        true_branch_lens[node_indices[node.name]] = node.dist

    ################################################################################
    # set rate constraint and initial estimates for GTR parameters

    match model:
        case "DNA" | "PHASED_DNA4":
            num_rate_params = 6
            rate_constraint = gtr4_rate
            if freq_params is None:
                freq_params = seq_pis

        case "PHASED_DNA16":
            num_rate_params = 6
            rate_constraint = phased_rate
            if freq_params is None:
                freq_params = seq_pis16

        case "PHASED_DNA16_MP":
            num_rate_params = 12
            rate_constraint = phased_mp_rate
            if freq_params is None:
                freq_params = seq_pis16

        case "UNPHASED_DNA":
            num_rate_params = 6
            rate_constraint = unphased_rate
            if freq_params is None:
                freq_params = unphased_freq_param_cleanup(seq_pis10)

        case "CELLPHY":
            num_rate_params = 6
            rate_constraint = cellphy10_rate
            if freq_params is None:
                freq_params = seq_pis10

        case "GTR10Z":
            num_rate_params = 24
            rate_constraint = gtr10z_rate
            if freq_params is None:
                freq_params = seq_pis10

        case "GTR10":
            num_rate_params = 45
            rate_constraint = gtr10_rate
            if freq_params is None:
                freq_params = seq_pis10

        case _:
            assert False, "Unknown model type"

    num_freq_params = len(freq_params)
    with np.errstate(divide="ignore"):
        log_freq_params = np.clip(np.log(freq_params), -1e100, 0.0)

    rate_params = rate_param_cleanup(
        np.ones(num_rate_params), log_freq_params, ploidy, rate_constraint
    )

    ##########################################################################################
    # jointly optimize GTR params and branch lens using neg-log likelihood
    ##########################################################################################

    match opt.model:
        case "DNA":
            patterns = np.array([pattern for pattern in counts.keys()])
            pattern_counts = np.array([count for count in counts.values()])

            prob_model_maker = make_GTR_prob_model

        case "PHASED_DNA4":
            genotype_counts = defaultdict(lambda: 0)
            for pattern, count in counts.items():
                pattern_mat = tuple(map(lambda p: p % 5, pattern))
                genotype_counts[pattern_mat] += 1.0

                pattern_pat = tuple(map(lambda p: p // 5, pattern))
                genotype_counts[pattern_pat] += 1.0

            patterns = np.array([pattern for pattern in genotype_counts.keys()])
            pattern_counts = np.array([count for count in genotype_counts.values()])

            prob_model_maker = make_GTR_prob_model

        case "PHASED_DNA16":
            patterns = np.array([pattern for pattern in counts.keys()])
            pattern_counts = np.array([count for count in counts.values()])

            prob_model_maker = make_GTRsq_prob_model

        case "PHASED_DNA16_MP":
            patterns = np.array([pattern for pattern in counts.keys()])
            pattern_counts = np.array([count for count in counts.values()])

            prob_model_maker = make_GTRxGTR_prob_model

        case "UNPHASED_DNA":
            patterns = np.array([pattern for pattern in counts.keys()])
            pattern_counts = np.array([count for count in counts.values()])

            prob_model_maker = make_unphased_GTRsq_prob_model

        case "CELLPHY":
            patterns = np.array([pattern for pattern in counts.keys()])
            pattern_counts = np.array([count for count in counts.values()])

            prob_model_maker = make_cellphy_prob_model

        case "GTR10Z":

            patterns = np.array([pattern for pattern in counts.keys()])
            pattern_counts = np.array([count for count in counts.values()])

            prob_model_maker = make_gtr10z_prob_model

        case "GTR10":

            patterns = np.array([pattern for pattern in counts.keys()])
            pattern_counts = np.array([count for count in counts.values()])

            prob_model_maker = make_gtr10_prob_model

        case _:
            assert False

    neg_log_likelihood = functools.partial(
        neg_log_likelihood_prototype,
        prob_model_maker=prob_model_maker,
        score_function=compute_score_function(
            root=true_tree,
            patterns=patterns,
            pattern_counts=pattern_counts,
            num_states=num_states,
            taxa_indices=taxa_indices,
            node_indices=node_indices,
        ),
    )

    ####################################################################################################
    # define optimization objectives for the model parameters and branch lengths

    param_objective = functools.partial(
        rate_param_objective_prototype,
        neg_log_likelihood=neg_log_likelihood,
        rate_constraint=rate_constraint,
        ploidy=ploidy,
    )

    branch_length_objective = functools.partial(
        branch_length_objective_prototype,
        neg_log_likelihood=neg_log_likelihood,
    )

    full_param_objective = functools.partial(
        param_objective_prototype,
        neg_log_likelihood=neg_log_likelihood,
        rate_constraint=rate_constraint,
        num_freq_params=num_freq_params,
        ploidy=ploidy,
    )

    # alternate a few times between optimizing the rate parameters and the branch lengths.
    # if requested, also optimize the frequency parameters
    for _ in range(2):
        res = minimize(
            param_objective,
            rate_params,
            args=(
                log_freq_params,
                branch_lengths,
            ),
            method="L-BFGS-B",
            bounds=[(1e-10, np.inf)] * num_rate_params,
            callback=CallbackParam() if opt.log else None,
            options=solver_options["L-BFGS-B"],
        )
        if opt.log:
            print(res)

        # fine tune mu
        rate_params = rate_param_cleanup(res.x, log_freq_params, ploidy, rate_constraint)

        res = minimize(
            branch_length_objective,
            branch_lengths,
            args=(log_freq_params, rate_params),
            method="L-BFGS-B",
            bounds=[(0.0, np.inf)] + [(1e-8, np.inf)] * (2 * len(taxa) - 2),
            callback=CallbackParam() if opt.log else None,
            options=solver_options["L-BFGS-B"],
        )
        if opt.log:
            print(res)

        # belt and suspenders for the constraint (avoid -1e-big type bounds violations)
        branch_lengths = np.maximum(0.0, res.x)

        if freq_params_option == "OPTIMIZE":
            if model == "UNPHASED_DNA":

                def unphased_full_param_objective(params, branch_lengths):
                    """
                    Run the full parameter objective function, but first convert the 4-state frequency parameters to
                    10 state parameters.

                    :param params: 4-state frequency parameters + rate params
                    :param branch_lengths:
                    :return:
                    """
                    log_freq_params_4, rate_params = params[:4], params[4:]
                    with np.errstate(divide="ignore"):
                        log_freq_params_10 = np.clip(
                            np.log(
                                np.clip(pi4s_to_unphased_pi10s(np.exp(log_freq_params_4)), 0.0, 1.0)
                            ),
                            -1e100,
                            0.0,
                        )
                        log_freq_params_4 -= logsumexp(log_freq_params_4)
                    return full_param_objective(
                        np.concatenate((log_freq_params_10, rate_params)), branch_lengths
                    )

                with np.errstate(divide="ignore"):
                    log_freq_params_4 = np.clip(
                        np.log(np.clip(pi10s_to_pi4s(np.exp(log_freq_params)), 0.0, 1.0)),
                        -1e100,
                        0.0,
                    )
                    log_freq_params_4 -= logsumexp(log_freq_params_4)
                res = minimize(
                    unphased_full_param_objective,
                    np.concatenate((log_freq_params_4, rate_params)),
                    args=(branch_lengths,),
                    method="L-BFGS-B",
                    bounds=([(-np.inf, 0.0)] * 4) + ([(1e-10, np.inf)] * num_rate_params),
                    callback=CallbackParam() if opt.log else None,
                    options=solver_options["L-BFGS-B"],
                )
                if opt.log:
                    print(res)

                log_freq_params_4 = np.minimum(0.0, res.x[:4])
                log_freq_params_4 -= logsumexp(log_freq_params_4)  # fine tune prob dist
                # convert back to 10 state frequencies
                freq_params = pi4s_to_unphased_pi10s(np.exp(log_freq_params_4))
                freq_params /= np.sum(freq_params)
                with np.errstate(divide="ignore"):
                    log_freq_params = np.clip(
                        np.log(freq_params),
                        -1e100,
                        0.0,
                    )

                # fine tune mu
                rate_params = rate_param_cleanup(
                    res.x[4:], log_freq_params, ploidy, rate_constraint
                )

            else:
                res = minimize(
                    full_param_objective,
                    np.concatenate((log_freq_params, rate_params)),
                    args=(branch_lengths,),
                    method="L-BFGS-B",
                    bounds=([(-np.inf, 0.0)] * num_freq_params)
                    + ([(1e-10, np.inf)] * num_rate_params),
                    callback=CallbackParam() if opt.log else None,
                    options=solver_options["L-BFGS-B"],
                )
                if opt.log:
                    print(res)

                log_freq_params = np.minimum(0.0, res.x[:num_freq_params])
                log_freq_params -= logsumexp(log_freq_params)  # fine tune prob dist
                freq_params = np.exp(log_freq_params)

                # fine tune mu
                rate_params = rate_param_cleanup(
                    res.x[num_freq_params:], log_freq_params, ploidy, rate_constraint
                )

    ####################################################################################################
    # Full joint (params + branch lengths) optimization

    if freq_params_option == "OPTIMIZE":

        from pruning.objective_functions import full_objective_prototype

        full_objective = functools.partial(
            full_objective_prototype,
            num_freq_params=num_freq_params,
            num_rate_params=num_rate_params,
            neg_log_likelihood=neg_log_likelihood,
            rate_constraint=rate_constraint,
            ploidy=ploidy,
        )

        if model == "UNPHASED_DNA":
            # in the unphased dna model, the 10 state frequency parameters are dependant on the 4-state parameters
            # so we have to handle this one separately

            def unphased_full_objective(params):
                """
                Run the full objective function, but first convert the 4-state frequency parameters to 10 state
                parameters.

                :param params: 4-state frequency parameters + rate params + branch lengths
                :return:
                """
                log_freq_params_4, other_params = params[:4], params[4:]
                with np.errstate(divide="ignore"):
                    log_freq_params_10 = np.clip(
                        np.log(
                            np.clip(pi4s_to_unphased_pi10s(np.exp(log_freq_params_4)), 0.0, 1.0)
                        ),
                        -1e100,
                        0.0,
                    )
                    log_freq_params_4 -= logsumexp(log_freq_params_4)
                return full_objective(np.concatenate((log_freq_params_10, other_params)))

            with np.errstate(divide="ignore"):
                log_freq_params_4 = np.clip(
                    np.log(np.clip(pi10s_to_pi4s(np.exp(log_freq_params)), 0.0, 1.0)),
                    -1e100,
                    0.0,
                )
                log_freq_params_4 -= logsumexp(log_freq_params_4)
            res = minimize(
                unphased_full_objective,
                np.concatenate((log_freq_params_4, rate_params, branch_lengths)),
                method="L-BFGS-B",
                bounds=([(-np.inf, 0.0)] * 4)
                + [(0.0, np.inf)] * (num_rate_params + 2 * len(taxa) - 1),
                callback=CallbackParam() if opt.log else None,
                options=solver_options["L-BFGS-B"],
            )
            if opt.log:
                print(res)

            log_freq_params_4 = np.minimum(0.0, res.x[:4])
            log_freq_params_4 -= logsumexp(log_freq_params_4)  # fine tune prob dist
            # convert back to 10 state frequencies
            freq_params = pi4s_to_unphased_pi10s(np.exp(log_freq_params_4))
            freq_params /= np.sum(freq_params)
            with np.errstate(divide="ignore"):
                log_freq_params = np.clip(
                    np.log(freq_params),
                    -1e100,
                    0.0,
                )

            # fine tune mu
            rate_params = rate_param_cleanup(
                res.x[4 : 4 + num_rate_params],
                log_freq_params,
                ploidy,
                rate_constraint,
            )

            branch_lengths = np.maximum(0.0, res.x[4 + num_rate_params :])
        else:
            # optimize the freq+rates+branch lengths

            res = minimize(
                full_objective,
                np.concatenate((log_freq_params, rate_params, branch_lengths)),
                method="L-BFGS-B",
                bounds=([(-np.inf, 0.0)] * num_freq_params)
                + [(0.0, np.inf)] * (num_rate_params + 2 * len(taxa) - 1),
                callback=CallbackParam() if opt.log else None,
                options=solver_options["L-BFGS-B"],
            )
            if opt.log:
                print(res)

            log_freq_params = np.minimum(0.0, res.x[:num_freq_params])
            log_freq_params -= logsumexp(log_freq_params)  # fine tune prob dist
            freq_params = np.exp(log_freq_params)

            # fine tune mu
            rate_params = rate_param_cleanup(
                res.x[num_freq_params : num_freq_params + num_rate_params],
                log_freq_params,
                ploidy,
                rate_constraint,
            )

            branch_lengths = np.maximum(0.0, res.x[num_freq_params + num_rate_params :])

    else:
        # optimize everything but the state frequencies
        params_distances_objective = functools.partial(
            rates_distances_objective_prototype,
            num_rate_params=num_rate_params,
            neg_log_likelihood=neg_log_likelihood,
            rate_constraint=rate_constraint,
            ploidy=ploidy,
        )

        res = minimize(
            params_distances_objective,
            np.concatenate((rate_params, branch_lengths)),
            args=(log_freq_params,),
            method="L-BFGS-B",
            bounds=[(0.0, np.inf)] * (num_rate_params + 2 * len(taxa) - 1),
            callback=CallbackParam() if opt.log else None,
            options=solver_options["L-BFGS-B"],
        )
        if opt.log:
            print(res)

        # fine tune mu
        rate_params = rate_param_cleanup(
            res.x[:num_rate_params], log_freq_params, ploidy, rate_constraint
        )

        branch_lengths = np.maximum(0.0, res.x[num_rate_params:])

        res = minimize(
            params_distances_objective,
            np.concatenate((rate_params, branch_lengths)),
            args=(log_freq_params,),
            method="Powell",
            bounds=[(0.0, np.inf)] * (num_rate_params + 2 * len(taxa) - 1),
            callback=CallbackParam() if opt.log else None,
            options=solver_options["Powell"],
        )
        if opt.log:
            print(res)

        # fine tune mu
        rate_params = rate_param_cleanup(
            res.x[:num_rate_params], log_freq_params, ploidy, rate_constraint
        )

        branch_lengths = np.maximum(0.0, res.x[num_rate_params:])

    ################################################################################
    # update branch lens in ETE3 tree

    for idx, node in enumerate(true_tree.traverse()):
        node.dist = branch_lengths[idx]

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
                    rate_params=rate_params,
                    freq_params=freq_params,
                    neg_l=res.fun,
                    tree_distances=branch_lengths,
                    true_branch_lens=true_branch_lens,
                    model=model,
                )

    else:
        print(newick_rep)
        print()
        print_stats(
            rate_params=rate_params,
            freq_params=freq_params,
            neg_l=res.fun,
            tree_distances=branch_lengths,
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
        case "PHASED_DNA4":
            from pruning.distance_functions import phased_sequence_distance

            sequence_distance = functools.partial(phased_sequence_distance, pis=pis)
        case "PHASED_DNA16" | "PHASED_DNA16_MP":
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


def read_sequences(ambig_char, model, sequence_file):
    import numpy as np

    from pruning.matrices import U, V, perm

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
    with open(sequence_file, "r") as seq_file:
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
            case "PHASED_DNA4":
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
            case "PHASED_DNA16" | "PHASED_DNA16_MP":
                num_states = 16
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
    if phased_joint_freq_counts is not None:
        pis16 = phased_joint_freq_counts / np.sum(phased_joint_freq_counts)
        if unphased_joint_freq_counts is not None:
            pis10 = unphased_joint_freq_counts / np.sum(unphased_joint_freq_counts)
        else:
            pis10 = pis16 @ perm @ V
    else:
        if unphased_joint_freq_counts is not None:
            pis10 = unphased_joint_freq_counts / np.sum(unphased_joint_freq_counts)
            # pis10 @ U = pis16 @ perm @ V @ U ~= pis16 @ perm
            # pis10 @ U @ perm.T ~= pis16 @ perm @ perm.T  = pis16
            pis16 = pis10 @ U @ perm.T
        else:
            pis16 = np.kron(pis, pis)
            pis10 = pis16 @ perm @ V

    return nsites, num_states, pis, pis10, pis16, sequences
