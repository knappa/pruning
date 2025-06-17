def main_cli():
    import argparse
    import functools
    import itertools
    import os.path
    import sys
    from collections import defaultdict

    import numpy as np
    from ete3 import Tree

    from pruning.fit import (
        compute_initial_tree_distance_estimates,
        fit_model,
        print_states,
        save_as_newick,
    )
    from pruning.matrices import (
        cellphy10_rate,
        gtr4_rate,
        gtr10_rate,
        gtr10z_rate,
        make_cellphy_prob_model,
        make_gtr10_prob_model,
        make_gtr10z_prob_model,
        make_GTR_prob_model,
        make_GTRsq_prob_model,
        make_GTRxGTR_prob_model,
        make_unphased_GTRsq_prob_model,
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
    from pruning.read_sequences import read_sequences
    from pruning.score_function_gen import compute_score_function, neg_log_likelihood_prototype
    from pruning.util import (
        CallbackParam,
        kahan_dot,
        log_dot,
        log_matrix_mult,
        rate_param_cleanup,
        rate_param_scale,
    )

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
        required=True,
        help="Datatype for sequence",
        choices=[
            "DNA",
            "PHASED_DNA16",
            "PHASED_DNA16_MP",
            "UNPHASED_DNA",
            "CELLPHY",
            "GTR10Z",
            "GTR10",
            # "SIEVE",
        ],
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

    parser.add_argument(
        "--ploidy", type=int, default=-1, help="force the ploidy to a specific value"
    )

    parser.add_argument(
        "--final_rp_norm",
        action="store_true",
        help="normalize rate parameters by setting final rate parameter to 1, instead of normalizing via mu",
    )

    parser.add_argument("--ambig", type=str, default="?", help="ambiguity character")
    parser.add_argument("--output", type=str, help="output filename prefix for tree")
    parser.add_argument("--overwrite", action="store_true", help="overwrite outputs, if they exist")
    parser.add_argument("--log", action="store_true")

    if hasattr(sys, "ps1"):
        opt = parser.parse_args(
            "--seqs test-data/test-data-diploid.phy "
            "--tree test-data/test-data-diploid.nwk "
            "--model UNPHASED_DNA "
            "--log ".split()
        )
    else:
        opt = parser.parse_args()

    if opt.log:
        print(opt)

    if (
        (not opt.overwrite)
        and (hasattr(opt, "output") and opt.output is not None)
        and (
            os.path.isfile(opt.output + ".nwk")
            or os.path.isfile(opt.output + ".log")
            or os.path.isfile(opt.output + ".csv")
        )
    ):
        print("output files from previous run present, exiting.")
        exit(-1)

    if hasattr(opt, "ploidy") and opt.ploidy != -1:
        ploidy = int(opt.ploidy)
    else:
        match getattr(opt, "model"):
            case "DNA" | "PHASED_DNA4":
                ploidy = 1
            case (
                "PHASED_DNA16" | "PHASED_DNA16_MP" | "UNPHASED_DNA" | "CELLPHY" | "GTR10Z" | "GTR10"
            ):
                ploidy = 2
            case _:
                raise NotImplementedError("Unknown model")

    final_rp_norm: bool = opt.final_rp_norm if hasattr(opt, "final_rp_norm") else False

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
    (
        nsites,
        pi4s_from_seq,
        pi10s_from_seq,
        pi16s_from_seq,
        sequences_16state,
        sequences_10state,
        sequences_4state,
    ) = read_sequences(ambig_char, opt.seqs, log=opt.log)

    if freq_params_option not in {"FIX4", "FIX10", "FIX16"}:
        # if we are deriving this from sequence data, this is the correct value;
        # if we are optimizing, this is a good initial seed value
        # noinspection PyUnreachableCode
        match opt.model:
            case "DNA" | "PHASED_DNA4":
                freq_params = pi4s_from_seq
            case "UNPHASED_DNA" | "CELLPHY" | "GTR10Z" | "GTR10":
                freq_params = pi10s_from_seq
                if opt.model == "UNPHASED_DNA":
                    freq_params = unphased_freq_param_cleanup(freq_params)
            case "PHASED_DNA16" | "PHASED_DNA16_MP":
                freq_params = pi16s_from_seq
            case _:
                raise NotImplementedError("Missing a model?")

    assert set(true_tree.get_leaf_names()) == set(
        sequences_16state.keys()
    ), "not the same leaves! are these matching datasets?"

    taxa = sorted(sequences_16state.keys())
    taxa_indices = dict(map(lambda pair_: pair_[::-1], enumerate(taxa)))

    # assemble the site pattern count tensor (sparse)
    # noinspection PyUnreachableCode
    match opt.model:
        case "DNA" | "PHASED_DNA4":
            sequences = sequences_4state
        case "UNPHASED_DNA" | "CELLPHY" | "GTR10Z" | "GTR10":
            sequences = sequences_10state
        case "PHASED_DNA16" | "PHASED_DNA16_MP":
            sequences = sequences_10state
        case _:
            raise NotImplementedError("Missing a model?")

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

    branch_lengths_init = compute_initial_tree_distance_estimates(
        node_indices=node_indices,
        num_tree_nodes=num_tree_nodes,
        pis=freq_params,
        sequences=sequences,
        taxa=taxa,
        true_tree=true_tree,
        model=opt.model,
        log=opt.log,
    )

    if opt.log:
        print("Branch length estimate:")
        print(branch_lengths_init)

    # collect true tree data for comparison, possibly on a different scale, so not directly
    # comparable, but good to have
    true_branch_lens = np.zeros(num_tree_nodes, dtype=np.float64)
    for node in true_tree.traverse():
        true_branch_lens[node_indices[node.name]] = node.dist

    ################################################################################
    # set rate constraint and initial estimates for GTR parameters

    # noinspection PyUnreachableCode
    match opt.model:
        case "DNA":
            num_states = 4
            num_rate_params = 6
            rate_constraint = gtr4_rate
            if freq_params is None:
                freq_params = pi4s_from_seq
            prob_model_maker = make_GTR_prob_model

        case "PHASED_DNA16":
            num_states = 16
            num_rate_params = 6
            rate_constraint = phased_rate
            if freq_params is None:
                freq_params = pi16s_from_seq
            prob_model_maker = make_GTRsq_prob_model

        case "PHASED_DNA16_MP":
            num_states = 16
            num_rate_params = 12
            rate_constraint = phased_mp_rate
            if freq_params is None:
                freq_params = pi16s_from_seq
            prob_model_maker = make_GTRxGTR_prob_model

        case "UNPHASED_DNA":
            num_states = 10
            num_rate_params = 6
            rate_constraint = unphased_rate
            if freq_params is None:
                freq_params = unphased_freq_param_cleanup(pi10s_from_seq)
            prob_model_maker = make_unphased_GTRsq_prob_model

        case "CELLPHY":
            num_states = 10
            num_rate_params = 6
            rate_constraint = cellphy10_rate
            if freq_params is None:
                freq_params = pi10s_from_seq
            prob_model_maker = make_cellphy_prob_model

        case "GTR10Z":
            num_states = 10
            num_rate_params = 24
            rate_constraint = gtr10z_rate
            if freq_params is None:
                freq_params = pi10s_from_seq
            prob_model_maker = make_gtr10z_prob_model

        case "GTR10":
            num_states = 10
            num_rate_params = 45
            rate_constraint = gtr10_rate
            if freq_params is None:
                freq_params = pi10s_from_seq
            prob_model_maker = make_gtr10_prob_model

        case _:
            assert False, "Unknown model type"

    # num_freq_params = len(freq_params)
    with np.errstate(divide="ignore"):
        log_freq_params = np.clip(np.nan_to_num(np.log(freq_params)), -1e100, 0.0)

    rate_params_init = np.ones(num_rate_params)
    if not final_rp_norm:
        rate_params_init = rate_param_cleanup(
            x=rate_params_init,
            log_freq_params=log_freq_params,
            ploidy=ploidy,
            rate_constraint=rate_constraint,
        )
    # if final_rp_norm, we would do the below, but we just initialized this to 1. so the update is omitted
    # rate_params_init /= rate_params_init[-1]

    ##########################################################################################
    # save tree before likelihood optimization
    ##########################################################################################

    save_as_newick(
        branch_lengths=branch_lengths_init,
        scale=(
            rate_param_scale(
                x=rate_params_init,
                log_freq_params=log_freq_params,
                ploidy=ploidy,
                rate_constraint=rate_constraint,
            )
            if final_rp_norm
            else 1
        ),
        output=opt.output + "-before-lklyhd-opt.nwk",
        true_tree=true_tree,
        to_stdout=(not hasattr(opt, "output")) or opt.output is None,
    )



    ##########################################################################################
    # jointly optimize GTR params and branch lens using neg-log likelihood
    ##########################################################################################

    neg_log_likelihood = functools.partial(
        neg_log_likelihood_prototype,
        prob_model_maker=prob_model_maker,
        score_function=compute_score_function(
            root=true_tree,
            patterns=np.array([pattern for pattern in counts.keys()]),
            pattern_counts=np.array([count for count in counts.values()]),
            num_states=num_states,
            taxa_indices=taxa_indices,
            node_indices=node_indices,
        ),
    )

    rate_params, branch_lengths, nll = fit_model(
        neg_log_likelihood=neg_log_likelihood,
        branch_lengths=branch_lengths_init,
        rate_params=rate_params_init,
        log_freq_params=log_freq_params,
        rate_constraint=rate_constraint,
        ploidy=ploidy,
        final_rp_norm=final_rp_norm,
        # optimize_freq_params =(freq_params_option == "OPTIMIZE"),
    )

    save_as_newick(
        branch_lengths=branch_lengths,
        scale=(
            rate_param_scale(
                x=rate_params,
                log_freq_params=log_freq_params,
                ploidy=ploidy,
                rate_constraint=rate_constraint,
            )
            if final_rp_norm
            else 1
        ),
        output=opt.output + ".nwk",
        true_tree=true_tree,
        to_stdout=(not hasattr(opt, "output")) or opt.output is None,
    )

    with open(opt.output + ".csv", "w") as file:
        file.write("nll")
        # noinspection PyUnreachableCode
        match opt.model:
            case "DNA" | "PHASED_DNA4":
                file.write(",pi_a,pi_c,pi_g,pi_t")
            case "UNPHASED_DNA" | "CELLPHY" | "GTR10Z" | "GTR10":
                file.write(",pi_aa,pi_cc,pi_gg,pi_tt,pi_ac,pi_ag,pi_at,pi_cg,pi_ct,pi_gt")
            case "PHASED_DNA16" | "PHASED_DNA16_MP":
                file.write(
                    ","
                    + ",".join(
                        f"pi_{x}{y}" for x, y in itertools.product(["a", "c", "g", "t"], repeat=2)
                    )
                )
            case _:
                raise NotImplementedError("Missing a model?")
        file.write("," + ",".join(f"s_{i}" for i in range(len(rate_params))))
        file.write("\n")

        file.write(f"{nll}")
        file.write("," + ",".join(map(str, freq_params)))
        file.write("," + ",".join(map(str, rate_params)))

    #
    #     if freq_params_option == "OPTIMIZE":
    #         if opt.model == "UNPHASED_DNA":
    #
    #             def unphased_full_param_objective(params, branch_lengths):
    #                 """
    #                 Run the full parameter objective function, but first convert the 4-state frequency parameters to
    #                 10 state parameters.
    #
    #                 :param params: 4-state frequency parameters + rate params
    #                 :param branch_lengths:
    #                 :return:
    #                 """
    #                 log_freq_params_4, rate_params = params[:4], params[4:]
    #                 with np.errstate(divide="ignore"):
    #                     log_freq_params_10 = np.clip(
    #                         np.log(
    #                             np.clip(pi4s_to_unphased_pi10s(np.exp(log_freq_params_4)), 0.0, 1.0)
    #                         ),
    #                         -1e100,
    #                         0.0,
    #                     )
    #                     log_freq_params_4 -= logsumexp(log_freq_params_4)
    #                 return full_param_objective(
    #                     np.concatenate((log_freq_params_10, rate_params)), branch_lengths
    #                 )
    #
    #             with np.errstate(divide="ignore"):
    #                 log_freq_params_4 = np.clip(
    #                     np.log(np.clip(pi10s_to_pi4s(np.exp(log_freq_params)), 0.0, 1.0)),
    #                     -1e100,
    #                     0.0,
    #                 )
    #                 log_freq_params_4 -= logsumexp(log_freq_params_4)
    #             res = minimize(
    #                 unphased_full_param_objective,
    #                 np.concatenate((log_freq_params_4, rate_params)),
    #                 args=(branch_lengths,),
    #                 method="L-BFGS-B",
    #                 bounds=([(-np.inf, 0.0)] * 4) + ([(1e-10, np.inf)] * num_rate_params),
    #                 callback=CallbackParam() if opt.log else None,
    #                 options=solver_options["L-BFGS-B"],
    #             )
    #             if opt.log:
    #                 print(res)
    #
    #             log_freq_params_4 = np.minimum(0.0, res.x[:4])
    #             log_freq_params_4 -= logsumexp(log_freq_params_4)  # fine tune prob dist
    #             # convert back to 10 state frequencies
    #             freq_params = pi4s_to_unphased_pi10s(np.exp(log_freq_params_4))
    #             freq_params /= np.sum(freq_params)
    #             with np.errstate(divide="ignore"):
    #                 log_freq_params = np.clip(
    #                     np.log(freq_params),
    #                     -1e100,
    #                     0.0,
    #                 )
    #
    #             # fine tune mu
    #             rate_params = rate_param_cleanup(
    #                 x=res.x[4:],
    #                 log_freq_params=log_freq_params,
    #                 ploidy=ploidy,
    #                 rate_constraint=rate_constraint,
    #             )
    #
    #         else:
    #             res = minimize(
    #                 full_param_objective,
    #                 np.concatenate((log_freq_params, rate_params)),
    #                 args=(branch_lengths,),
    #                 method="L-BFGS-B",
    #                 bounds=([(-np.inf, 0.0)] * num_freq_params)
    #                 + ([(1e-10, np.inf)] * num_rate_params),
    #                 callback=CallbackParam() if opt.log else None,
    #                 options=solver_options["L-BFGS-B"],
    #             )
    #             if opt.log:
    #                 print(res)
    #
    #             log_freq_params = np.minimum(0.0, res.x[:num_freq_params])
    #             log_freq_params -= logsumexp(log_freq_params)  # fine tune prob dist
    #             freq_params = np.exp(log_freq_params)
    #
    #             # fine tune mu
    #             rate_params = rate_param_cleanup(
    #                 x=res.x[num_freq_params:],
    #                 log_freq_params=log_freq_params,
    #                 ploidy=ploidy,
    #                 rate_constraint=rate_constraint,
    #             )
    #
    # ####################################################################################################
    # # Full joint (params + branch lengths) optimization
    #
    # if freq_params_option == "OPTIMIZE":
    #
    #     from pruning.objective_functions import full_objective_prototype
    #
    #     full_objective = functools.partial(
    #         full_objective_prototype,
    #         num_freq_params=num_freq_params,
    #         num_rate_params=num_rate_params,
    #         neg_log_likelihood=neg_log_likelihood,
    #         rate_constraint=rate_constraint,
    #         ploidy=ploidy,
    #     )
    #
    #     if opt.model == "UNPHASED_DNA":
    #         # in the unphased dna model, the 10 state frequency parameters are dependant on the 4-state parameters
    #         # so we have to handle this one separately
    #
    #         def unphased_full_objective(params):
    #             """
    #             Run the full objective function, but first convert the 4-state frequency parameters to 10 state
    #             parameters.
    #
    #             :param params: 4-state frequency parameters + rate params + branch lengths
    #             :return:
    #             """
    #             log_freq_params_4, other_params = params[:4], params[4:]
    #             with np.errstate(divide="ignore"):
    #                 log_freq_params_10 = np.clip(
    #                     np.log(
    #                         np.clip(pi4s_to_unphased_pi10s(np.exp(log_freq_params_4)), 0.0, 1.0)
    #                     ),
    #                     -1e100,
    #                     0.0,
    #                 )
    #                 log_freq_params_4 -= logsumexp(log_freq_params_4)
    #             return full_objective(np.concatenate((log_freq_params_10, other_params)))
    #
    #         with np.errstate(divide="ignore"):
    #             log_freq_params_4 = np.clip(
    #                 np.log(np.clip(pi10s_to_pi4s(np.exp(log_freq_params)), 0.0, 1.0)),
    #                 -1e100,
    #                 0.0,
    #             )
    #             log_freq_params_4 -= logsumexp(log_freq_params_4)
    #         res = minimize(
    #             unphased_full_objective,
    #             np.concatenate((log_freq_params_4, rate_params, branch_lengths)),
    #             method="L-BFGS-B",
    #             bounds=([(-np.inf, 0.0)] * 4)
    #             + [(0.0, np.inf)] * (num_rate_params + 2 * len(taxa) - 1),
    #             callback=CallbackParam() if opt.log else None,
    #             options=solver_options["L-BFGS-B"],
    #         )
    #         if opt.log:
    #             print(res)
    #
    #         log_freq_params_4 = np.minimum(0.0, res.x[:4])
    #         log_freq_params_4 -= logsumexp(log_freq_params_4)  # fine tune prob dist
    #         # convert back to 10 state frequencies
    #         freq_params = pi4s_to_unphased_pi10s(np.exp(log_freq_params_4))
    #         freq_params /= np.sum(freq_params)
    #         with np.errstate(divide="ignore"):
    #             log_freq_params = np.clip(
    #                 np.log(freq_params),
    #                 -1e100,
    #                 0.0,
    #             )
    #
    #         # fine tune mu
    #         rate_params = rate_param_cleanup(
    #             x=res.x[4 : 4 + num_rate_params],
    #             log_freq_params=log_freq_params,
    #             ploidy=ploidy,
    #             rate_constraint=rate_constraint,
    #         )
    #
    #         branch_lengths = np.maximum(0.0, res.x[4 + num_rate_params :])
    #     else:
    #         # optimize the freq+rates+branch lengths
    #
    #         res = minimize(
    #             full_objective,
    #             np.concatenate((log_freq_params, rate_params, branch_lengths)),
    #             method="L-BFGS-B",
    #             bounds=([(-np.inf, 0.0)] * num_freq_params)
    #             + [(0.0, np.inf)] * (num_rate_params + 2 * len(taxa) - 1),
    #             callback=CallbackParam() if opt.log else None,
    #             options=solver_options["L-BFGS-B"],
    #         )
    #         if opt.log:
    #             print(res)
    #
    #         log_freq_params = np.minimum(0.0, res.x[:num_freq_params])
    #         log_freq_params -= logsumexp(log_freq_params)  # fine tune prob dist
    #         freq_params = np.exp(log_freq_params)
    #
    #         # fine tune mu
    #         rate_params = rate_param_cleanup(
    #             x=res.x[num_freq_params : num_freq_params + num_rate_params],
    #             log_freq_params=log_freq_params,
    #             ploidy=ploidy,
    #             rate_constraint=rate_constraint,
    #         )
    #
    #         branch_lengths = np.maximum(0.0, res.x[num_freq_params + num_rate_params :])

    ################################################################################
    ################################################################################
    ################################################################################
    ################################################################################
