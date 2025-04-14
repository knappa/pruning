def main_cli():
    import argparse
    import functools
    import os.path
    import sys
    from collections import defaultdict
    from contextlib import redirect_stdout

    import numpy as np
    from ete3 import Tree

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
        pi10s_to_pi4s,
        unphased_freq_param_cleanup,
        unphased_rate,
    )
    from pruning.score_function_gen import compute_score_function, neg_log_likelihood_prototype
    from pruning.util import rate_param_cleanup

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
        "--fix_freq_params",
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
            "--seqs test/test-diploid.phy --tree test/test-diploid.nwk --log ".split()
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
        raise NotImplementedError()
    elif opt.freq_params_from_seq:
        freq_params_option = "FROM_SEQ"
    elif opt.fix_freq_params:
        freq_params_option = "FIX"
        freq_params = np.array(opt.fix_freq_params)
        raise NotImplementedError()
    else:
        freq_params_option = "FROM_SEQ"

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
        seq_pi4,
        seq_pi10,
        seq_pi16,
        sequences_16state,
        sequences_10state,
        sequences_4state,
    ) = read_sequences(ambig_char, opt.seqs)

    assert set(true_tree.get_leaf_names()) == set(
        sequences_16state.keys()
    ), "not the same leaves! are these matching datasets?"

    taxa = sorted(sequences_16state.keys())
    taxa_indices = dict(map(lambda pair_: pair_[::-1], enumerate(taxa)))

    # assemble the site pattern count tensors (sparse)
    counts_16state = defaultdict(lambda: 0)
    for idx in range(nsites):
        # noinspection PyShadowingNames
        pattern = tuple(
            map(
                lambda taxon: sequences_16state[taxon][idx],
                taxa,
            )
        )
        counts_16state[pattern] += 1

    counts_10state = defaultdict(lambda: 0)
    for idx in range(nsites):
        # noinspection PyShadowingNames
        pattern = tuple(
            map(
                lambda taxon: sequences_10state[taxon][idx],
                taxa,
            )
        )
        counts_10state[pattern] += 1

    counts_4state = defaultdict(lambda: 0)
    for idx in range(nsites):
        # noinspection PyShadowingNames
        pattern = tuple(
            map(
                lambda taxon: sequences_4state[taxon][idx],
                taxa,
            )
        )
        counts_4state[pattern] += 1

    ################################################################################
    # initial estimates for branch lengths based on (generalized) F81 distances

    branch_lengths_init = compute_initial_tree_distance_estimates(
        node_indices=node_indices,
        num_tree_nodes=num_tree_nodes,
        opt=opt,
        pis=seq_pi4,
        sequences=sequences_16state,
        taxa=taxa,
        true_tree=true_tree,
    )

    if opt.log:
        print("Branch length estimate:")
        print(branch_lengths_init)

    # collect true tree data for comparison, likely on a different scale (GT transversion?), so not directly
    # comparable, but possibly good to have
    true_branch_lens = np.zeros(num_tree_nodes, dtype=np.float64)
    for node in true_tree.traverse():
        true_branch_lens[node_indices[node.name]] = node.dist

    ##########################################################################################
    # Fit a 4 state model
    # num_rate_params = 6

    freq_params_4state = seq_pi4
    with np.errstate(divide="ignore"):
        log_freq_params_4state = np.clip(np.log(freq_params_4state), -1e100, 0.0)

    rate_params_4state = rate_param_cleanup(
        x=np.ones(6), log_freq_params=log_freq_params_4state, ploidy=1, rate_constraint=gtr4_rate
    )

    neg_log_likelihood_4state = functools.partial(
        neg_log_likelihood_prototype,
        prob_model_maker=make_GTR_prob_model,
        score_function=compute_score_function(
            root=true_tree,
            patterns=np.array([pattern for pattern in counts_4state.keys()]),
            pattern_counts=np.array([count for count in counts_4state.values()]),
            num_states=4,
            taxa_indices=taxa_indices,
            node_indices=node_indices,
        ),
    )

    rate_params_4state, branch_lengths_4state, nll_4state = fit_model(
        neg_log_likelihood=neg_log_likelihood_4state,
        branch_lengths=branch_lengths_init,
        log_freq_params=log_freq_params_4state,
        rate_params=rate_params_4state,
        rate_constraint=gtr4_rate,
        ploidy=1,
    )

    # update branch lens in ETE3 tree, and write tree to a file, depending up command line opts
    for idx, node in enumerate(true_tree.traverse()):
        node.dist = branch_lengths_4state[idx]

    newick_rep = true_tree.write(format=5)

    if hasattr(opt, "output") and opt.output is not None:
        with open(opt.output + "-4state.nwk", "w") as file:
            file.write(newick_rep)
            file.write("\n")

    ##########################################################################################
    # Fit an unphased model
    # num_rate_params = 6

    freq_params_unphased = unphased_freq_param_cleanup(seq_pi10)
    with np.errstate(divide="ignore"):
        log_freq_params_unphased = np.clip(np.log(freq_params_unphased), -1e100, 0.0)

    rate_params_unphased = rate_param_cleanup(
        x=rate_params_4state,
        log_freq_params=log_freq_params_unphased,
        ploidy=2,
        rate_constraint=unphased_rate,
    )

    neg_log_likelihood_unphased = functools.partial(
        neg_log_likelihood_prototype,
        prob_model_maker=make_unphased_GTRsq_prob_model,
        score_function=compute_score_function(
            root=true_tree,
            patterns=np.array([pattern for pattern in counts_10state.keys()]),
            pattern_counts=np.array([count for count in counts_10state.values()]),
            num_states=10,
            taxa_indices=taxa_indices,
            node_indices=node_indices,
        ),
    )

    rate_params_unphased, branch_lengths_unphased, nll_unphased = fit_model(
        neg_log_likelihood=neg_log_likelihood_unphased,
        branch_lengths=branch_lengths_4state,
        log_freq_params=log_freq_params_unphased,
        rate_params=rate_params_unphased,
        rate_constraint=unphased_rate,
        ploidy=2,
    )

    # update branch lens in ETE3 tree, and write tree to a file, depending up command line opts
    for idx, node in enumerate(true_tree.traverse()):
        node.dist = branch_lengths_unphased[idx]

    newick_rep = true_tree.write(format=5)

    if hasattr(opt, "output") and opt.output is not None:
        with open(opt.output + "-unphased.nwk", "w") as file:
            file.write(newick_rep)
            file.write("\n")

    ##########################################################################################
    # Fit the cellphy model
    # num_rate_params = 6

    freq_params_cellphy = seq_pi10
    with np.errstate(divide="ignore"):
        log_freq_params_cellphy = np.clip(np.log(freq_params_cellphy), -1e100, 0.0)

    rate_params_cellphy = rate_param_cleanup(
        x=np.ones(6, dtype=np.float64),
        log_freq_params=log_freq_params_cellphy,
        ploidy=2,
        rate_constraint=cellphy10_rate,
    )

    neg_log_likelihood_cellphy = functools.partial(
        neg_log_likelihood_prototype,
        prob_model_maker=make_cellphy_prob_model,
        score_function=compute_score_function(
            root=true_tree,
            patterns=np.array([pattern for pattern in counts_10state.keys()]),
            pattern_counts=np.array([count for count in counts_10state.values()]),
            num_states=10,
            taxa_indices=taxa_indices,
            node_indices=node_indices,
        ),
    )

    rate_params_cellphy, branch_lengths_cellphy, nll_cellphy = fit_model(
        neg_log_likelihood=neg_log_likelihood_cellphy,
        branch_lengths=branch_lengths_4state,
        log_freq_params=log_freq_params_cellphy,
        rate_params=rate_params_cellphy,
        rate_constraint=cellphy10_rate,
        ploidy=2,
    )

    # update branch lens in ETE3 tree, and write tree to a file, depending up command line opts
    for idx, node in enumerate(true_tree.traverse()):
        node.dist = branch_lengths_cellphy[idx]

    newick_rep = true_tree.write(format=5)

    if hasattr(opt, "output") and opt.output is not None:
        with open(opt.output + "-cellphy.nwk", "w") as file:
            file.write(newick_rep)
            file.write("\n")

    ##########################################################################################
    # Fit a GTR10Z model
    # num_rate_params = 24

    freq_params_gtr10z = seq_pi10
    with np.errstate(divide="ignore"):
        log_freq_params_gtr10z = np.clip(np.log(freq_params_gtr10z), -1e100, 0.0)

    def unphased_to_gtr10z(pis10, rate_params):
        pi_a, pi_c, pi_g, pi_t = pi10s_to_pi4s(pis10)
        s_ac, s_ag, s_at, s_cg, s_ct, s_gt = np.clip(rate_params, 0.0, np.inf)
        return np.array(
            [
                s_ac / pi_a,
                s_ag / pi_a,
                s_at / pi_a,
                s_ac / pi_c,
                s_cg / pi_c,
                s_ct / pi_c,
                s_ag / pi_g,
                s_cg / pi_g,
                s_gt / pi_g,
                s_at / pi_t,
                s_ct / pi_t,
                s_gt / pi_t,
                s_cg / (2 * pi_a),
                s_ct / (2 * pi_a),
                s_ag / (2 * pi_c),
                s_at / (2 * pi_c),
                s_gt / (2 * pi_a),
                s_ac / (2 * pi_g),
                s_at / (2 * pi_g),
                s_ac / (2 * pi_t),
                s_ag / (2 * pi_t),
                s_gt / (2 * pi_c),
                s_ct / (2 * pi_g),
                s_cg / (2 * pi_t),
            ]
        )

    rate_params_gtr10z = rate_param_cleanup(
        x=unphased_to_gtr10z(freq_params_gtr10z, rate_params_unphased),
        log_freq_params=log_freq_params_gtr10z,
        ploidy=2,
        rate_constraint=gtr10z_rate,
    )

    neg_log_likelihood_gtr10z = functools.partial(
        neg_log_likelihood_prototype,
        prob_model_maker=make_gtr10z_prob_model,
        score_function=compute_score_function(
            root=true_tree,
            patterns=np.array([pattern for pattern in counts_10state.keys()]),
            pattern_counts=np.array([count for count in counts_10state.values()]),
            num_states=10,
            taxa_indices=taxa_indices,
            node_indices=node_indices,
        ),
    )

    rate_params_gtr10z, branch_lengths_gtr10z, nll_gtr10z = fit_model(
        neg_log_likelihood=neg_log_likelihood_gtr10z,
        branch_lengths=branch_lengths_unphased,
        log_freq_params=log_freq_params_gtr10z,
        rate_params=rate_params_gtr10z,
        rate_constraint=gtr10z_rate,
        ploidy=2,
    )

    # update branch lens in ETE3 tree, and write tree to a file, depending up command line opts
    for idx, node in enumerate(true_tree.traverse()):
        node.dist = branch_lengths_gtr10z[idx]

    newick_rep = true_tree.write(format=5)

    if hasattr(opt, "output") and opt.output is not None:
        with open(opt.output + "-gtr10z.nwk", "w") as file:
            file.write(newick_rep)
            file.write("\n")

    ##########################################################################################
    # Fit a GTR10 model
    # num_rate_params = 45

    freq_params_gtr10 = seq_pi10
    with np.errstate(divide="ignore"):
        log_freq_params_gtr10 = np.clip(np.log(freq_params_gtr10), -1e100, 0.0)

    def gtr10z_to_gtr10(s):
        return np.array(
            [
                0,  # AA -> CC
                0,  # AA -> GG
                0,  # AA -> TT
                s[0],  # AA -> AC
                s[1],  # AA -> AG
                s[2],  # AA -> AT
                0,  # AA -> CG
                0,  # AA -> CT
                0,  # AA -> GT
                0,  # CC -> GG
                0,  # CC -> TT
                s[3],  # CC -> AC
                0,  # CC -> AG
                0,  # CC -> AT
                s[4],  # CC -> CG
                s[5],  # CC -> CT
                0,  # CC -> GT
                0,  # GG -> TT
                0,  # GG -> AC
                s[6],  # GG -> AG
                0,  # GG -> AT
                s[7],  # GG -> CG
                0,  # GG -> CT
                s[8],  # GG -> GT
                0,  # TT -> AC
                0,  # TT -> AG
                s[9],  # TT -> AT
                0,  # TT -> CG
                s[10],  # TT -> CT
                s[11],  # TT -> GT
                s[12],  # AC -> AG
                s[13],  # AC -> AT
                s[14],  # AC -> CG
                s[15],  # AC -> CT
                0,  # AC -> GT
                s[16],  # AG -> AT
                s[17],  # AG -> CG
                0,  # AG -> CT
                s[18],  # AG -> GT
                0,  # AT -> CG
                s[19],  # AT -> CT
                s[20],  # AT -> GT
                s[21],  # CG -> CT
                s[22],  # CG -> GT
                s[23],  # CT -> GT
            ]
        )

    rate_params_gtr10 = rate_param_cleanup(
        x=gtr10z_to_gtr10(rate_params_gtr10z),
        log_freq_params=log_freq_params_gtr10,
        ploidy=2,
        rate_constraint=gtr10_rate,
    )

    neg_log_likelihood_gtr10 = functools.partial(
        neg_log_likelihood_prototype,
        prob_model_maker=make_gtr10_prob_model,
        score_function=compute_score_function(
            root=true_tree,
            patterns=np.array([pattern for pattern in counts_10state.keys()]),
            pattern_counts=np.array([count for count in counts_10state.values()]),
            num_states=10,
            taxa_indices=taxa_indices,
            node_indices=node_indices,
        ),
    )

    rate_params_gtr10, branch_lengths_gtr10, nll_gtr10 = fit_model(
        neg_log_likelihood=neg_log_likelihood_gtr10,
        branch_lengths=branch_lengths_gtr10z,
        log_freq_params=log_freq_params_gtr10,
        rate_params=rate_params_gtr10,
        rate_constraint=gtr10_rate,
        ploidy=2,
    )

    # update branch lens in ETE3 tree, and write tree to a file, depending up command line opts
    for idx, node in enumerate(true_tree.traverse()):
        node.dist = branch_lengths_gtr10[idx]

    newick_rep = true_tree.write(format=5)

    if hasattr(opt, "output") and opt.output is not None:
        with open(opt.output + "-gtr10.nwk", "w") as file:
            file.write(newick_rep)
            file.write("\n")

    ##########################################################################################
    # Fit a 16 state model with same Mat/Pat rates
    # num_rate_params = 6

    freq_params_16state = seq_pi16
    with np.errstate(divide="ignore"):
        log_freq_params_16state = np.clip(np.log(freq_params_16state), -1e100, 0.0)

    rate_params_16state = rate_param_cleanup(
        x=rate_params_4state,
        log_freq_params=log_freq_params_16state,
        ploidy=2,
        rate_constraint=phased_rate,
    )

    neg_log_likelihood_16state = functools.partial(
        neg_log_likelihood_prototype,
        prob_model_maker=make_GTRsq_prob_model,
        score_function=compute_score_function(
            root=true_tree,
            patterns=np.array([pattern for pattern in counts_16state.keys()]),
            pattern_counts=np.array([count for count in counts_16state.values()]),
            num_states=16,
            taxa_indices=taxa_indices,
            node_indices=node_indices,
        ),
    )

    rate_params_16state, branch_lengths_16state, nll_16state = fit_model(
        neg_log_likelihood=neg_log_likelihood_16state,
        branch_lengths=branch_lengths_4state,
        log_freq_params=log_freq_params_16state,
        rate_params=rate_params_16state,
        rate_constraint=phased_rate,
        ploidy=2,
    )

    # update branch lens in ETE3 tree, and write tree to a file, depending up command line opts
    for idx, node in enumerate(true_tree.traverse()):
        node.dist = branch_lengths_16state[idx]

    newick_rep = true_tree.write(format=5)

    if hasattr(opt, "output") and opt.output is not None:
        with open(opt.output + "-16state.nwk", "w") as file:
            file.write(newick_rep)
            file.write("\n")

    ##########################################################################################
    # Fit a 16 state model with differing Mat/Pat rates
    # num_rate_params = 12

    freq_params_16state_mp = seq_pi16
    with np.errstate(divide="ignore"):
        log_freq_params_16state_mp = np.clip(np.log(freq_params_16state_mp), -1e100, 0.0)

    rate_params_16state_mp = rate_param_cleanup(
        x=np.concatenate((rate_params_16state, rate_params_16state), axis=0),
        log_freq_params=log_freq_params_16state_mp,
        ploidy=2,
        rate_constraint=phased_mp_rate,
    )

    neg_log_likelihood_16state_mp = functools.partial(
        neg_log_likelihood_prototype,
        prob_model_maker=make_GTRxGTR_prob_model,
        score_function=compute_score_function(
            root=true_tree,
            patterns=np.array([pattern for pattern in counts_16state.keys()]),
            pattern_counts=np.array([count for count in counts_16state.values()]),
            num_states=16,
            taxa_indices=taxa_indices,
            node_indices=node_indices,
        ),
    )

    rate_params_16state_mp, branch_lengths_16state_mp, nll_16state_mp = fit_model(
        neg_log_likelihood=neg_log_likelihood_16state_mp,
        branch_lengths=branch_lengths_16state,
        log_freq_params=log_freq_params_16state_mp,
        rate_params=rate_params_16state_mp,
        rate_constraint=phased_mp_rate,
        ploidy=2,
    )

    # update branch lens in ETE3 tree, and write tree to a file, depending up command line opts
    for idx, node in enumerate(true_tree.traverse()):
        node.dist = branch_lengths_16state_mp[idx]

    newick_rep = true_tree.write(format=5)

    if hasattr(opt, "output") and opt.output is not None:
        with open(opt.output + "-16state_mp.nwk", "w") as file:
            file.write(newick_rep)
            file.write("\n")

    ################################################################################

    if hasattr(opt, "output") and opt.output is not None:
        with open(opt.output + ".csv", "w") as file:
            with redirect_stdout(file):
                print_states(
                    freq_params_4state,
                    rate_params_4state,
                    nll_4state,
                    freq_params_unphased,
                    rate_params_unphased,
                    nll_unphased,
                    freq_params_cellphy,
                    rate_params_cellphy,
                    nll_cellphy,
                    freq_params_gtr10z,
                    rate_params_gtr10z,
                    nll_gtr10z,
                    freq_params_gtr10,
                    rate_params_gtr10,
                    nll_gtr10,
                    freq_params_16state,
                    rate_params_16state,
                    nll_16state,
                    freq_params_16state_mp,
                    rate_params_16state_mp,
                    nll_16state_mp,
                )
    else:
        print_states(
            freq_params_4state,
            rate_params_4state,
            nll_4state,
            freq_params_unphased,
            rate_params_unphased,
            nll_unphased,
            freq_params_cellphy,
            rate_params_cellphy,
            nll_cellphy,
            freq_params_gtr10z,
            rate_params_gtr10z,
            nll_gtr10z,
            freq_params_gtr10,
            rate_params_gtr10,
            nll_gtr10,
            freq_params_16state,
            rate_params_16state,
            nll_16state,
            freq_params_16state_mp,
            rate_params_16state_mp,
            nll_16state_mp,
        )

    # if hasattr(opt, "output") and opt.output is not None:
    #     with open(opt.output + "-16state_mp.nwk", "w") as file:
    #         file.write(newick_rep)
    #         file.write("\n")

    ################################################################################
    ################################################################################
    ################################################################################


def fit_model(
    *,
    branch_lengths,
    log_freq_params,
    rate_params,
    neg_log_likelihood,
    rate_constraint,
    ploidy,
    log=True,
):
    import functools

    import numpy as np
    from scipy.optimize import minimize

    from pruning.objective_functions import (
        branch_length_objective_prototype,
        rate_param_objective_prototype,
        rates_distances_objective_prototype,
    )
    from pruning.util import CallbackParam, rate_param_cleanup, solver_options

    num_rate_params = len(rate_params)
    num_branch_lens = len(branch_lengths)

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

    # alternate a few times between optimizing the rate parameters and the branch lengths.
    for _ in range(3):

        res = minimize(
            param_objective,
            rate_params,
            args=(
                log_freq_params,
                branch_lengths,
            ),
            method="L-BFGS-B",
            bounds=[(1e-10, np.inf)] * num_rate_params,
            callback=CallbackParam() if log else None,
            options=solver_options["L-BFGS-B"],
        )
        if log:
            print(res)

        # fine tune mu
        rate_params = rate_param_cleanup(
            x=res.x,
            log_freq_params=log_freq_params,
            ploidy=ploidy,
            rate_constraint=rate_constraint,
        )

        res = minimize(
            branch_length_objective,
            branch_lengths,
            args=(log_freq_params, rate_params),
            method="L-BFGS-B",
            bounds=[(0.0, np.inf)] + [(1e-8, np.inf)] * (num_branch_lens - 1),
            callback=CallbackParam() if log else None,
            options=solver_options["L-BFGS-B"],
        )
        if log:
            print(res)

        # belt and suspenders for the constraint (avoid -1e-big type bounds violations)
        branch_lengths = np.maximum(0.0, res.x)

    # Joint (params + branch lengths) optimization
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
        bounds=[(0.0, np.inf)] * (num_rate_params + num_branch_lens),
        callback=CallbackParam() if log else None,
        options=solver_options["L-BFGS-B"],
    )
    if log:
        print(res)

    # fine tune mu
    rate_params = rate_param_cleanup(
        x=res.x[:num_rate_params],
        log_freq_params=log_freq_params,
        ploidy=1,
        rate_constraint=rate_constraint,
    )

    branch_lengths = np.maximum(0.0, res.x[num_rate_params:])

    res = minimize(
        params_distances_objective,
        np.concatenate((rate_params, branch_lengths)),
        args=(log_freq_params,),
        method="Powell",
        bounds=[(0.0, np.inf)] * (num_rate_params + num_branch_lens),
        callback=CallbackParam() if log else None,
        options=solver_options["Powell"],
    )
    if log:
        print(res)

    # fine tune mu
    rate_params = rate_param_cleanup(
        x=res.x[:num_rate_params],
        log_freq_params=log_freq_params,
        ploidy=ploidy,
        rate_constraint=rate_constraint,
    )

    branch_lengths = np.maximum(0.0, res.x[num_rate_params:])

    return rate_params, branch_lengths, res.fun


def compute_initial_tree_distance_estimates(
    *, node_indices, num_tree_nodes, opt, pis, sequences, taxa, true_tree
):
    import functools
    import itertools

    import numpy as np
    from scipy.optimize import minimize

    from pruning.distance_functions import diploid_dna_sequence_distance
    from pruning.path_constraints import make_path_constraints
    from pruning.util import CallbackParam

    sequence_distance = functools.partial(diploid_dna_sequence_distance, pis=pis)

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


def read_sequences(ambig_char, sequence_file):
    import numpy as np

    from pruning.matrices import V, perm

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

    # read and process the sequence file
    with open(sequence_file, "r") as seq_file:
        # first line consists of counts
        ntaxa, nsites = map(int, next(seq_file).split())

        phased_joint_freq_counts = np.zeros(25, dtype=np.int64)
        # parse sequences
        sequences_16state = dict()
        sequences_10state = dict()
        sequences_4state = dict()

        for line in seq_file:
            taxon, *seq = line.strip().split()
            assert taxon not in sequences_16state and taxon not in sequences_10state

            seq = list(map(lambda s: s.upper(), seq))
            assert all(len(s) == 2 for s in seq)

            sequences_4state[taxon] = np.array(
                [nuc_to_idx[nuc] for nuc in "".join(seq)],
                dtype=np.uint8,
            )

            sequences_10state[taxon] = np.array(
                [unphased_nuc_to_idx[nuc] for nuc in seq],
                dtype=np.uint8,
            )

            # sequence coding is lexicographic AA, AC, AG, AT, A?, CA, ...
            # which is equivalent to a base-5 encoding 00=0, 01=1, 02=2, 03=3, 04=4, 10=5, ...
            sequences_16state[taxon] = np.array(
                [nuc_to_idx[nuc[0]] * 5 + nuc_to_idx[nuc[1]] for nuc in seq],
                dtype=np.uint8,
            )

            for nuc in seq:
                phased_joint_freq_counts[nuc_to_idx[nuc[0]] * 5 + nuc_to_idx[nuc[1]]] += 1

        assert ntaxa == len(sequences_16state)

    freq_count_mat5 = np.sum(phased_joint_freq_counts.reshape(5, 5), axis=1)
    freq_count_pat5 = np.sum(phased_joint_freq_counts.reshape(5, 5), axis=0)

    freq_count_mat4 = freq_count_mat5[:4] + freq_count_mat5[4] / 4
    freq_count_pat4 = freq_count_pat5[:4] + freq_count_pat5[4] / 4

    freq_count_4 = freq_count_mat4 + freq_count_pat4
    pi4 = freq_count_4 / np.sum(freq_count_4)

    print(f"{pi4=}")

    freq_count_16 = (
        phased_joint_freq_counts.reshape(5, 5)[:4, :4]
        + phased_joint_freq_counts.reshape(5, 5)[4, :4][None, :] / 4
        + phased_joint_freq_counts.reshape(5, 5)[:4, 4][:, None] / 4
        + phased_joint_freq_counts.reshape(5, 5)[4, 4] / 16
    ).reshape(-1)
    pi16 = freq_count_16 / np.sum(freq_count_16)

    pi10 = pi16 @ perm @ V

    return (
        nsites,
        pi4,
        pi10,
        pi16,
        sequences_16state,
        sequences_10state,
        sequences_4state,
    )


def print_states(
    freq_params_4state,
    rate_params_4state,
    nll_4state,
    freq_params_unphased,
    rate_params_unphased,
    nll_unphased,
    freq_params_cellphy,
    rate_params_cellphy,
    nll_cellphy,
    freq_params_gtr10z,
    rate_params_gtr10z,
    nll_gtr10z,
    freq_params_gtr10,
    rate_params_gtr10,
    nll_gtr10,
    freq_params_16state,
    rate_params_16state,
    nll_16state,
    freq_params_16state_mp,
    rate_params_16state_mp,
    nll_16state_mp,
):
    data = {
        "nll_4state": nll_4state,
        "nll_unphased": nll_unphased,
        "nll_cellphy": nll_cellphy,
        "nll_gtr10z": nll_gtr10z,
        "nll_gtr10": nll_gtr10,
        "nll_16state": nll_16state,
        "nll_16state_mp": nll_16state_mp,
        **{"4state S_" + str(i): s for i, s in enumerate(rate_params_4state)},
        **{"unphased S_" + str(i): s for i, s in enumerate(rate_params_unphased)},
        **{"cellphy S_" + str(i): s for i, s in enumerate(rate_params_cellphy)},
        **{"gtr10z S_" + str(i): s for i, s in enumerate(rate_params_gtr10z)},
        **{"gtr10 S_" + str(i): s for i, s in enumerate(rate_params_gtr10)},
        **{"16state S_" + str(i): s for i, s in enumerate(rate_params_16state)},
        **{"16state_mp S_" + str(i): s for i, s in enumerate(rate_params_16state_mp)},
        **{"4state pi_" + str(i): s for i, s in enumerate(freq_params_4state)},
        **{"unphased pi_" + str(i): s for i, s in enumerate(freq_params_unphased)},
        **{"cellphy pi_" + str(i): s for i, s in enumerate(freq_params_cellphy)},
        **{"gtr10z pi_" + str(i): s for i, s in enumerate(freq_params_gtr10z)},
        **{"gtr10 pi_" + str(i): s for i, s in enumerate(freq_params_gtr10)},
        **{"16state pi_" + str(i): s for i, s in enumerate(freq_params_16state)},
        **{"16state_mp pi_" + str(i): s for i, s in enumerate(freq_params_16state_mp)},
    }
    print(",".join(data.keys()))
    print(",".join(data.values()))
