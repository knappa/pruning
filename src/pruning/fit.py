import numpy as np


def save_as_newick(
    *, branch_lengths: np.ndarray, scale: float, output: str, true_tree, to_stdout: bool = False
):
    # update branch lens in ETE3 tree, and write tree to a file, depending up command line opts
    for idx, node in enumerate(true_tree.traverse()):
        node.dist = branch_lengths[idx] * scale

    newick_rep = true_tree.write(format=5)

    if to_stdout:
        print(newick_rep)
    else:
        with open(output, "w") as file:
            file.write(newick_rep)
            file.write("\n")


def fit_model(
    *,
    branch_lengths,
    rate_params,
    log_freq_params,
    neg_log_likelihood,
    rate_constraint,
    ploidy,
    final_rp_norm: bool,
    log: bool = True,
    n_split_iters: int = 2,
    min_branch_length: float = 1e-7,
    min_rate_param: float = 1e-7,
):
    """
    Fit the model.

    :param branch_lengths:
    :param rate_params:
    :param log_freq_params:
    :param neg_log_likelihood:
    :param rate_constraint:
    :param ploidy:
    :param final_rp_norm:
    :param log:
    :param n_split_iters:
    :param min_branch_length: TODO questionable
    :param min_rate_param: TODO questionable
    :return:
    """
    import functools

    import numpy as np
    from scipy.optimize import minimize

    from pruning.objective_functions import (
        branch_length_objective_prototype,
        rate_param_objective_prototype,
        rates_distances_objective_prototype,
    )
    from pruning.util import CallbackParam, solver_options

    num_rate_params = len(rate_params)
    num_branch_lens = len(branch_lengths)

    param_objective = functools.partial(
        rate_param_objective_prototype,
        neg_log_likelihood=neg_log_likelihood,
        rate_constraint=rate_constraint,
        ploidy=ploidy,
        final_rp_norm=final_rp_norm,
    )
    branch_length_objective = functools.partial(
        branch_length_objective_prototype,
        neg_log_likelihood=neg_log_likelihood,
    )

    rate_params = scale_params(
        rate_params=rate_params,
        log_freq_params=log_freq_params,
        ploidy=ploidy,
        rate_constraint=rate_constraint,
        final_rp_norm=final_rp_norm,
    )

    best_nll = np.inf

    # enforce constraints
    rate_params = np.clip(np.nan_to_num(rate_params), min_rate_param, np.inf)
    branch_lengths[0] = 0.0
    branch_lengths[1:] = np.clip(np.nan_to_num(branch_lengths[1:]), min_branch_length, np.inf)

    # alternate a few times between optimizing the branch lengths and the rate parameters.
    for _ in range(n_split_iters):

        try:
            # optimize branch lengths
            res = minimize(
                branch_length_objective,
                branch_lengths,
                args=(log_freq_params, rate_params),
                method="L-BFGS-B",
                bounds=[(0.0, max(min_branch_length / 2, 1e-10))]
                + [(min_branch_length, np.inf)] * (num_branch_lens - 1),
                callback=CallbackParam(print_period=10) if log else None,
                options=solver_options["L-BFGS-B"],
            )
            best_nll = np.minimum(best_nll, res.fun)
            if log:
                print(res)

            # belt and suspenders for the constraint
            branch_lengths[0] = 0.0
            branch_lengths[1:] = np.clip(np.nan_to_num(res.x[1:]), min_branch_length, np.inf)
        except ValueError:
            pass

        try:
            # optimize rate parameters
            res = minimize(
                param_objective,
                rate_params,
                args=(
                    log_freq_params,
                    branch_lengths,
                ),
                method="L-BFGS-B",
                bounds=[(min_rate_param, np.inf)] * num_rate_params,
                callback=CallbackParam(print_period=10) if log else None,
                options=solver_options["L-BFGS-B"],
            )
            best_nll = np.minimum(best_nll, res.fun)
            if log:
                print(res)

            rate_params = np.clip(np.nan_to_num(res.x), min_rate_param, np.inf)

            rate_params = scale_params(
                rate_params=rate_params,
                log_freq_params=log_freq_params,
                ploidy=ploidy,
                rate_constraint=rate_constraint,
                final_rp_norm=final_rp_norm,
            )
        except ValueError:
            pass

    # Joint (params + branch lengths) optimization
    # optimize everything but the state frequencies
    params_distances_objective = functools.partial(
        rates_distances_objective_prototype,
        num_rate_params=num_rate_params,
        neg_log_likelihood=neg_log_likelihood,
        rate_constraint=rate_constraint,
        ploidy=ploidy,
        final_rp_norm=final_rp_norm,
    )
    try:
        res = minimize(
            params_distances_objective,
            np.concatenate((rate_params, branch_lengths)),
            args=(log_freq_params,),
            method="L-BFGS-B",
            bounds=[(min_rate_param, np.inf)] * num_rate_params
            + [(0.0, max(min_branch_length / 2, 1e-10))]
            + [(min_branch_length, np.inf)] * (num_branch_lens - 1),
            callback=CallbackParam(print_period=10) if log else None,
            options=solver_options["L-BFGS-B"],
        )
        best_nll = np.minimum(best_nll, res.fun)
        if log:
            print(res)

        rate_params = np.clip(np.nan_to_num(res.x[:num_rate_params]), min_rate_param, np.inf)
        branch_lengths[0] = 0.0
        branch_lengths[1:] = np.clip(
            np.nan_to_num(res.x[num_rate_params + 1 :]), min_branch_length, np.inf
        )

        rate_params = scale_params(
            rate_params=rate_params,
            log_freq_params=log_freq_params,
            ploidy=ploidy,
            rate_constraint=rate_constraint,
            final_rp_norm=final_rp_norm,
        )
    except ValueError:
        pass

    # final cleanup optimization with boosted constraint weights
    params_distances_objective = functools.partial(
        params_distances_objective,
        constraint_weight=best_nll,
    )
    res = minimize(
        params_distances_objective,
        np.concatenate((rate_params, branch_lengths)),
        args=(log_freq_params,),
        method="Nelder-Mead",
        bounds=[(min_rate_param, np.inf)] * num_rate_params
        + [(0.0, max(min_branch_length / 2, 1e-10))]
        + [(min_branch_length, np.inf)] * (num_branch_lens - 1),
        callback=CallbackParam(print_period=100) if log else None,
        options=solver_options["Nelder-Mead"],
    )
    if log:
        print(res)

    rate_params = np.clip(np.nan_to_num(res.x[:num_rate_params]), min_rate_param, np.inf)
    branch_lengths[0] = 0.0
    branch_lengths[1:] = np.clip(res.x[num_rate_params + 1 :], 0.0, np.inf)

    rate_params = scale_params(
        rate_params=rate_params,
        log_freq_params=log_freq_params,
        ploidy=ploidy,
        rate_constraint=rate_constraint,
        final_rp_norm=final_rp_norm,
    )

    bare_nll = neg_log_likelihood(log_freq_params, rate_params, branch_lengths)

    if log:
        print(
            f"Best nll with constraint loss {best_nll}, Bare nll {bare_nll}, delta = {np.abs(best_nll-bare_nll)}"
        )

    return rate_params, branch_lengths, bare_nll


def scale_params(
    *,
    rate_params,
    final_rp_norm,
    log_freq_params,
    ploidy,
    rate_constraint,
):
    """
    Rescale branch lengths and rate parameters in tandem.

    :param rate_params:
    :param final_rp_norm:
    :param log_freq_params:
    :param ploidy:
    :param rate_constraint:
    :return:
    """
    from pruning.util import rate_param_scale

    if final_rp_norm:
        if rate_params[-1] <= 0.0:
            rate_params += 1e-6 - rate_params[-1]
        inv_scale = rate_params[-1]
        return rate_params / inv_scale

    else:
        # fine tune mu
        scale = rate_param_scale(
            x=rate_params,
            log_freq_params=log_freq_params,
            ploidy=ploidy,
            rate_constraint=rate_constraint,
        )
        return rate_params * scale


def compute_initial_tree_distance_estimates(
    *,
    node_indices,
    num_tree_nodes,
    opt,
    pis,
    sequences,
    taxa,
    true_tree,
    min_branch_length: float = 1e-7,
):
    import functools
    import itertools

    import numpy as np
    from scipy.optimize import minimize

    from pruning.distance_functions import diploid_dna_sequence_distance
    from pruning.path_constraints import make_path_constraints
    from pruning.util import CallbackParam, solver_options

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
        bounds=[(0.0, max(min_branch_length / 2, 1e-10))]
        + [(min_branch_length, np.inf)] * (num_tree_nodes - 1),
        callback=CallbackParam(print_period=10) if opt.log else None,
        method="L-BFGS-B",
        options=solver_options["L-BFGS-B-Lite"],
    )
    if opt.log and not res.success:
        print("Optimization did not terminate, continuing anyway", flush=True)

    # belt and suspenders for the constraint (avoid -1e-big type bounds violations)
    tree_distances = np.zeros_like(res.x, dtype=np.float64)
    tree_distances[1:] = np.maximum(min_branch_length, np.nan_to_num(res.x[1:]))

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
    print(",".join(map(str, data.values())), flush=True)


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
