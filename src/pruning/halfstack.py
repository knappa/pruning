def main_cli():
    import argparse
    import functools
    import os.path
    import sys
    from collections import defaultdict
    from contextlib import redirect_stdout

    import numpy as np
    from ete3 import Tree

    from pruning.fit import (
        compute_initial_tree_distance_estimates,
        fit_model,
        gtr10z_to_gtr10,
        read_sequences,
        save_as_newick,
    )
    from pruning.matrices import (
        cellphy10_rate,
        gtr10_rate,
        gtr10z_rate,
        make_cellphy_prob_model,
        make_gtr10_prob_model,
        make_gtr10z_prob_model,
        make_unphased_GTRsq_prob_model,
        pi10s_to_pi4s,
        unphased_freq_param_cleanup,
        unphased_rate,
    )
    from pruning.score_function_gen import compute_score_function, neg_log_likelihood_prototype
    from pruning.util import rate_param_cleanup, rate_param_scale

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
            "--seqs test/test-diploid.phy --tree test/test-diploid.nwk --log ".split()
        )
    else:
        opt = parser.parse_args()

    if opt.log:
        print(opt)

    if (
        not opt.overwrite
        and opt.output is not None
        and (
            os.path.isfile(opt.output + ".nwk")
            or os.path.isfile(opt.output + ".log")
            or os.path.isfile(opt.output + ".csv")
        )
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

    force_ploidy: int = opt.ploidy
    final_rp_norm: bool = opt.final_rp_norm if hasattr(opt, "final_rp_norm") else False

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
    # Fit an unphased model
    # num_rate_params = 6

    freq_params_unphased = unphased_freq_param_cleanup(seq_pi10)
    with np.errstate(divide="ignore"):
        log_freq_params_unphased = np.clip(np.log(freq_params_unphased), -1e100, 0.0)

    rate_params_unphased = np.ones(6, dtype=np.float64)
    if final_rp_norm:
        rate_params_unphased = rate_params_unphased / rate_params_unphased[-1]
    else:
        rate_params_unphased = rate_param_cleanup(
            x=rate_params_unphased,
            log_freq_params=log_freq_params_unphased,
            ploidy=2 if force_ploidy == -1 else force_ploidy,
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
        branch_lengths=branch_lengths_init,
        rate_params=rate_params_unphased,
        log_freq_params=log_freq_params_unphased,
        rate_constraint=unphased_rate,
        ploidy=2 if force_ploidy == -1 else force_ploidy,
        final_rp_norm=final_rp_norm,
    )

    if hasattr(opt, "output") and opt.output is not None:
        save_as_newick(
            branch_lengths=branch_lengths_unphased,
            scale=(
                rate_param_scale(
                    x=rate_params_unphased,
                    log_freq_params=log_freq_params_unphased,
                    ploidy=2 if force_ploidy == -1 else force_ploidy,
                    rate_constraint=unphased_rate,
                )
                if final_rp_norm
                else 1
            ),
            output=opt.output + "-unphased.nwk",
            true_tree=true_tree,
        )

    ##########################################################################################
    # Fit a GTR10Z model, based on the unphased model
    # num_rate_params = 24

    freq_params_gtr10z_from_unphased = seq_pi10
    with np.errstate(divide="ignore"):
        log_freq_params_gtr10z = np.clip(np.log(freq_params_gtr10z_from_unphased), -1e100, 0.0)

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

    rate_params_gtr10z_from_unphased = unphased_to_gtr10z(
        freq_params_gtr10z_from_unphased, rate_params_unphased
    )
    if final_rp_norm:
        rate_params_gtr10z_from_unphased /= rate_params_gtr10z_from_unphased[-1]
    else:
        rate_params_gtr10z_from_unphased = rate_param_cleanup(
            x=rate_params_gtr10z_from_unphased,
            log_freq_params=log_freq_params_gtr10z,
            ploidy=2 if force_ploidy == -1 else force_ploidy,
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

    (
        rate_params_gtr10z_from_unphased,
        branch_lengths_gtr10z_from_unphased,
        nll_gtr10z_from_unphased,
    ) = fit_model(
        neg_log_likelihood=neg_log_likelihood_gtr10z,
        branch_lengths=branch_lengths_unphased,
        rate_params=rate_params_gtr10z_from_unphased,
        log_freq_params=log_freq_params_gtr10z,
        rate_constraint=gtr10z_rate,
        ploidy=2 if force_ploidy == -1 else force_ploidy,
        final_rp_norm=final_rp_norm,
    )

    if hasattr(opt, "output") and opt.output is not None:
        save_as_newick(
            branch_lengths=branch_lengths_gtr10z_from_unphased,
            scale=(
                rate_param_scale(
                    x=rate_params_gtr10z_from_unphased,
                    log_freq_params=log_freq_params_gtr10z,
                    ploidy=2 if force_ploidy == -1 else force_ploidy,
                    rate_constraint=gtr10z_rate,
                )
                if final_rp_norm
                else 1
            ),
            output=opt.output + "-gtr10z-from-unphased.nwk",
            true_tree=true_tree,
        )

    ##########################################################################################
    # Fit a GTR10 model
    # num_rate_params = 45

    freq_params_gtr10_from_unphased = seq_pi10
    with np.errstate(divide="ignore"):
        log_freq_params_gtr10 = np.clip(np.log(freq_params_gtr10_from_unphased), -1e100, 0.0)

    rate_params_gtr10_from_unphased = gtr10z_to_gtr10(rate_params_gtr10z_from_unphased)
    if final_rp_norm:
        rate_params_gtr10_from_unphased /= rate_params_gtr10_from_unphased[-1]
    else:
        rate_params_gtr10_from_unphased = rate_param_cleanup(
            x=rate_params_gtr10_from_unphased,
            log_freq_params=log_freq_params_gtr10,
            ploidy=2 if force_ploidy == -1 else force_ploidy,
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

    rate_params_gtr10_from_unphased, branch_lengths_gtr10_from_unphased, nll_gtr10_from_unphased = (
        fit_model(
            neg_log_likelihood=neg_log_likelihood_gtr10,
            branch_lengths=branch_lengths_gtr10z_from_unphased,
            rate_params=rate_params_gtr10_from_unphased,
            log_freq_params=log_freq_params_gtr10,
            rate_constraint=gtr10_rate,
            ploidy=2 if force_ploidy == -1 else force_ploidy,
            final_rp_norm=final_rp_norm,
        )
    )

    if hasattr(opt, "output") and opt.output is not None:
        save_as_newick(
            branch_lengths=branch_lengths_gtr10_from_unphased,
            scale=(
                rate_param_scale(
                    x=rate_params_gtr10_from_unphased,
                    log_freq_params=log_freq_params_gtr10,
                    ploidy=2 if force_ploidy == -1 else force_ploidy,
                    rate_constraint=gtr10_rate,
                )
                if final_rp_norm
                else 1
            ),
            output=opt.output + "-gtr10-from-unphased.nwk",
            true_tree=true_tree,
        )

    ##########################################################################################
    # Fit the cellphy model
    # num_rate_params = 6

    freq_params_cellphy = seq_pi10
    with np.errstate(divide="ignore"):
        log_freq_params_cellphy = np.clip(np.log(freq_params_cellphy), -1e100, 0.0)

    if final_rp_norm:
        rate_params_cellphy = np.ones(6, dtype=np.float64)
    else:
        rate_params_cellphy = rate_param_cleanup(
            x=np.ones(6, dtype=np.float64),
            log_freq_params=log_freq_params_cellphy,
            ploidy=2 if force_ploidy == -1 else force_ploidy,
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
        branch_lengths=branch_lengths_init,
        rate_params=rate_params_cellphy,
        log_freq_params=log_freq_params_cellphy,
        rate_constraint=cellphy10_rate,
        ploidy=2 if force_ploidy == -1 else force_ploidy,
        final_rp_norm=final_rp_norm,
    )

    if hasattr(opt, "output") and opt.output is not None:
        save_as_newick(
            branch_lengths=branch_lengths_cellphy,
            scale=(
                rate_param_scale(
                    x=rate_params_cellphy,
                    log_freq_params=log_freq_params_cellphy,
                    ploidy=2 if force_ploidy == -1 else force_ploidy,
                    rate_constraint=cellphy10_rate,
                )
                if final_rp_norm
                else 1
            ),
            output=opt.output + "-cellphy.nwk",
            true_tree=true_tree,
        )

    ##########################################################################################
    # Fit a GTR10Z model, from the cellphy model
    # num_rate_params = 24

    freq_params_gtr10z_from_cellphy = seq_pi10
    with np.errstate(divide="ignore"):
        log_freq_params_gtr10z = np.clip(np.log(freq_params_gtr10z_from_cellphy), -1e100, 0.0)

    def cellphy_to_gtr10z(rate_params):
        s_ac, s_ag, s_at, s_cg, s_ct, s_gt = np.clip(rate_params, 0.0, np.inf)
        return np.array(
            [
                s_ac,
                s_ag,
                s_at,
                s_ac,
                s_cg,
                s_ct,
                s_ag,
                s_cg,
                s_gt,
                s_at,
                s_ct,
                s_gt,
                s_cg,
                s_ct,
                s_ag,
                s_at,
                s_gt,
                s_ac,
                s_at,
                s_ac,
                s_ag,
                s_gt,
                s_ct,
                s_cg,
            ]
        )

    rate_params_gtr10z_from_cellphy = cellphy_to_gtr10z(rate_params_cellphy)
    if final_rp_norm:
        rate_params_gtr10z_from_cellphy /= rate_params_gtr10z_from_cellphy[-1]
    else:
        rate_params_gtr10z_from_cellphy = rate_param_cleanup(
            x=rate_params_gtr10z_from_cellphy,
            log_freq_params=log_freq_params_gtr10z,
            ploidy=2 if force_ploidy == -1 else force_ploidy,
            rate_constraint=gtr10z_rate,
        )

    neg_log_likelihood_gtr10z_from_cellphy = functools.partial(
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

    rate_params_gtr10z_from_cellphy, branch_lengths_gtr10z_from_cellphy, nll_gtr10z_from_cellphy = (
        fit_model(
            neg_log_likelihood=neg_log_likelihood_gtr10z_from_cellphy,
            branch_lengths=branch_lengths_cellphy,
            rate_params=rate_params_gtr10z_from_cellphy,
            log_freq_params=log_freq_params_gtr10z,
            rate_constraint=gtr10z_rate,
            ploidy=2 if force_ploidy == -1 else force_ploidy,
            final_rp_norm=final_rp_norm,
        )
    )

    if hasattr(opt, "output") and opt.output is not None:
        save_as_newick(
            branch_lengths=branch_lengths_gtr10z_from_cellphy,
            scale=(
                rate_param_scale(
                    x=rate_params_gtr10z_from_cellphy,
                    log_freq_params=log_freq_params_gtr10z,
                    ploidy=2 if force_ploidy == -1 else force_ploidy,
                    rate_constraint=gtr10z_rate,
                )
                if final_rp_norm
                else 1
            ),
            output=opt.output + "-gtr10z-from-cellphy.nwk",
            true_tree=true_tree,
        )

    ##########################################################################################
    # Fit a GTR10 model
    # num_rate_params = 45

    freq_params_gtr10_from_cellphy = seq_pi10
    with np.errstate(divide="ignore"):
        log_freq_params_gtr10 = np.clip(np.log(freq_params_gtr10_from_cellphy), -1e100, 0.0)

    rate_params_gtr10_from_cellphy = gtr10z_to_gtr10(rate_params_gtr10z_from_cellphy)
    if final_rp_norm:
        rate_params_gtr10_from_cellphy /= rate_params_gtr10_from_cellphy[-1]
    else:
        rate_params_gtr10_from_cellphy = rate_param_cleanup(
            x=rate_params_gtr10_from_cellphy,
            log_freq_params=log_freq_params_gtr10,
            ploidy=2 if force_ploidy == -1 else force_ploidy,
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

    rate_params_gtr10_from_cellphy, branch_lengths_gtr10_from_cellphy, nll_gtr10_from_cellphy = (
        fit_model(
            neg_log_likelihood=neg_log_likelihood_gtr10,
            branch_lengths=branch_lengths_gtr10z_from_cellphy,
            rate_params=rate_params_gtr10_from_cellphy,
            log_freq_params=log_freq_params_gtr10,
            rate_constraint=gtr10_rate,
            ploidy=2 if force_ploidy == -1 else force_ploidy,
            final_rp_norm=final_rp_norm,
        )
    )

    if hasattr(opt, "output") and opt.output is not None:
        save_as_newick(
            branch_lengths=branch_lengths_gtr10_from_cellphy,
            scale=(
                rate_param_scale(
                    x=rate_params_gtr10_from_cellphy,
                    log_freq_params=log_freq_params_gtr10,
                    ploidy=2 if force_ploidy == -1 else force_ploidy,
                    rate_constraint=gtr10_rate,
                )
                if final_rp_norm
                else 1
            ),
            output=opt.output + "-gtr10-from-cellphy.nwk",
            true_tree=true_tree,
        )

    ################################################################################

    if hasattr(opt, "output") and opt.output is not None:
        with open(opt.output + ".csv", "w") as file:
            with redirect_stdout(file):
                print_states(
                    freq_params_unphased,
                    rate_params_unphased,
                    nll_unphased,
                    freq_params_cellphy,
                    rate_params_cellphy,
                    nll_cellphy,
                    freq_params_gtr10z_from_unphased,
                    rate_params_gtr10z_from_unphased,
                    nll_gtr10z_from_unphased,
                    freq_params_gtr10_from_unphased,
                    rate_params_gtr10_from_unphased,
                    nll_gtr10_from_unphased,
                    freq_params_gtr10z_from_cellphy,
                    rate_params_gtr10z_from_cellphy,
                    nll_gtr10z_from_cellphy,
                    freq_params_gtr10_from_cellphy,
                    rate_params_gtr10_from_cellphy,
                    nll_gtr10_from_cellphy,
                )
    else:
        print_states(
            freq_params_unphased,
            rate_params_unphased,
            nll_unphased,
            freq_params_cellphy,
            rate_params_cellphy,
            nll_cellphy,
            freq_params_gtr10z_from_unphased,
            rate_params_gtr10z_from_unphased,
            nll_gtr10z_from_unphased,
            freq_params_gtr10_from_unphased,
            rate_params_gtr10_from_unphased,
            nll_gtr10_from_unphased,
            freq_params_gtr10z_from_cellphy,
            rate_params_gtr10z_from_cellphy,
            nll_gtr10z_from_cellphy,
            freq_params_gtr10_from_cellphy,
            rate_params_gtr10_from_cellphy,
            nll_gtr10_from_cellphy,
        )


def print_states(
    freq_params_unphased,
    rate_params_unphased,
    nll_unphased,
    freq_params_cellphy,
    rate_params_cellphy,
    nll_cellphy,
    freq_params_gtr10z_from_unphased,
    rate_params_gtr10z_from_unphased,
    nll_gtr10z_from_unphased,
    freq_params_gtr10_from_unphased,
    rate_params_gtr10_from_unphased,
    nll_gtr10_from_unphased,
    freq_params_gtr10z_from_cellphy,
    rate_params_gtr10z_from_cellphy,
    nll_gtr10z_from_cellphy,
    freq_params_gtr10_from_cellphy,
    rate_params_gtr10_from_cellphy,
    nll_gtr10_from_cellphy,
):
    data = {
        "nll_unphased": nll_unphased,
        "nll_gtr10z_from_unphased": nll_gtr10z_from_unphased,
        "nll_gtr10_from_unphased": nll_gtr10_from_unphased,
        "nll_cellphy": nll_cellphy,
        "nll_gtr10z_from_cellphy": nll_gtr10z_from_cellphy,
        "nll_gtr10_from_cellphy": nll_gtr10_from_cellphy,
        **{"unphased S_" + str(i): s for i, s in enumerate(rate_params_unphased)},
        **{
            "gtr10z_from_unphased S_" + str(i): s
            for i, s in enumerate(rate_params_gtr10z_from_unphased)
        },
        **{
            "gtr10_from_unphased S_" + str(i): s
            for i, s in enumerate(rate_params_gtr10_from_unphased)
        },
        **{"cellphy S_" + str(i): s for i, s in enumerate(rate_params_cellphy)},
        **{
            "gtr10z_from_cellphy S_" + str(i): s
            for i, s in enumerate(rate_params_gtr10z_from_cellphy)
        },
        **{
            "gtr10_from_cellphy S_" + str(i): s
            for i, s in enumerate(rate_params_gtr10_from_cellphy)
        },
        **{"unphased pi_" + str(i): s for i, s in enumerate(freq_params_unphased)},
        **{
            "gtr10z_from_unphased pi_" + str(i): s
            for i, s in enumerate(freq_params_gtr10z_from_unphased)
        },
        **{
            "gtr10_from_unphased pi_" + str(i): s
            for i, s in enumerate(freq_params_gtr10_from_unphased)
        },
        **{"cellphy pi_" + str(i): s for i, s in enumerate(freq_params_cellphy)},
        **{
            "gtr10z_from_cellphy pi_" + str(i): s
            for i, s in enumerate(freq_params_gtr10z_from_cellphy)
        },
        **{
            "gtr10_from_cellphy pi_" + str(i): s
            for i, s in enumerate(freq_params_gtr10_from_cellphy)
        },
    }
    print(",".join(data.keys()))
    print(",".join(map(str, data.values())), flush=True)
