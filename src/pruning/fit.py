import numpy as np

from pruning.util import np_full_print


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
    min_branch_length: float = 1e-8,
    min_rate_param: float = 1e-8,
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
    params_distances_objective = functools.partial(
        rates_distances_objective_prototype,
        num_rate_params=num_rate_params,
        neg_log_likelihood=neg_log_likelihood,
        rate_constraint=rate_constraint,
        ploidy=ploidy,
        final_rp_norm=final_rp_norm,
    )

    best_nll = np.inf

    # enforce constraints
    rate_params = np.clip(np.nan_to_num(rate_params), min_rate_param, np.inf)
    rate_params = scale_params(
        rate_params=rate_params,
        log_freq_params=log_freq_params,
        ploidy=ploidy,
        rate_constraint=rate_constraint,
        final_rp_norm=final_rp_norm,
    )
    branch_lengths[0] = 0.0
    branch_lengths[1:] = np.clip(np.nan_to_num(branch_lengths[1:]), min_branch_length, np.inf)

    # Joint (params + branch lengths) optimization
    # optimize everything but the state frequencies

    # # initial global search
    # res = minimize(
    #     params_distances_objective,
    #     np.concatenate((rate_params, branch_lengths)),
    #     args=(log_freq_params,),
    #     method="Nelder-Mead",
    #     bounds=[(min_rate_param, np.inf)] * num_rate_params
    #     + [(0.0, max(min_branch_length / 2, 1e-10))]
    #     + [(min_branch_length, np.inf)] * (num_branch_lens - 1),
    #     callback=CallbackParam(print_period=100) if log else None,
    #     options=solver_options["Nelder-Mead"],
    # )
    # if log:
    #     print(res)
    #
    # branch_lengths[0] = 0.0
    # branch_lengths[1:] = np.clip(res.x[num_rate_params + 1 :], 0.0, np.inf)
    #
    # rate_params = np.clip(np.nan_to_num(res.x[:num_rate_params]), min_rate_param, np.inf)
    # rate_params = scale_params(
    #     rate_params=rate_params,
    #     log_freq_params=log_freq_params,
    #     ploidy=ploidy,
    #     rate_constraint=rate_constraint,
    #     final_rp_norm=final_rp_norm,
    # )

    # fine-tuning: alternate a few times between optimizing the branch lengths and the rate parameters.
    for options in ["L-BFGS-B-Medium", "L-BFGS-B-Heavy"]:
        if log:
            print("optimizing branch lengths from likelihood function " + options, flush=True)
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
                options=solver_options[options],
            )
            best_nll = np.minimum(best_nll, res.fun)
            if log:
                print(res, flush=True)

            # belt and suspenders for the constraint
            branch_lengths[0] = 0.0
            branch_lengths[1:] = np.clip(np.nan_to_num(res.x[1:]), min_branch_length, np.inf)
        except ValueError:
            pass

        if log:
            print("optimizing rate parameters from likelihood function " + options, flush=True)
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
                options=solver_options[options],
            )
            best_nll = np.minimum(best_nll, res.fun)
            if log:
                print(res, flush=True)

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

        if log:
            print(
                "optimizing joint rate parameters and branch lengths from likelihood function "
                + options,
                flush=True,
            )
        # local search in joint rate parameter + branch length space
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
                options=solver_options[options],
            )
            best_nll = np.minimum(best_nll, res.fun)
            if log:
                print(res, flush=True)

            rate_params = np.clip(np.nan_to_num(res.x[:num_rate_params]), min_rate_param, np.inf)
            rate_params = scale_params(
                rate_params=rate_params,
                log_freq_params=log_freq_params,
                ploidy=ploidy,
                rate_constraint=rate_constraint,
                final_rp_norm=final_rp_norm,
            )

            branch_lengths[0] = 0.0
            branch_lengths[1:] = np.clip(
                np.nan_to_num(res.x[num_rate_params + 1 :]), min_branch_length, np.inf
            )

        except ValueError:
            pass

    if log:
        print("final optimization of branch lengths from likelihood function", flush=True)
    # final cleanup pass at the branch lengths
    try:
        # optimize branch lengths
        res = minimize(
            branch_length_objective,
            branch_lengths,
            args=(log_freq_params, rate_params),
            method="Powell",
            bounds=[(0.0, max(min_branch_length / 2, 1e-10))]
            + [(min_branch_length, np.inf)] * (num_branch_lens - 1),
            callback=CallbackParam(print_period=10) if log else None,
            options=solver_options["Powell"],
        )
        best_nll = np.minimum(best_nll, res.fun)
        if log:
            print(res, flush=True)

        # belt and suspenders for the constraint
        branch_lengths[0] = 0.0
        branch_lengths[1:] = np.clip(np.nan_to_num(res.x[1:]), min_branch_length, np.inf)
    except ValueError:
        pass

    if log:
        print(
            "optimizing joint rate parameters and branch lengths from likelihood function "
            "with extra constraint weight",
            flush=True,
        )
    # local search in joint rate parameter + branch length space
    params_distances_objective = functools.partial(
        params_distances_objective,
        constraint_weight=best_nll,
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
            options=solver_options["L-BFGS-B-Heavy"],
        )
        best_nll = np.minimum(best_nll, res.fun)
        if log:
            print(res, flush=True)

        rate_params = np.clip(np.nan_to_num(res.x[:num_rate_params]), min_rate_param, np.inf)
        rate_params = scale_params(
            rate_params=rate_params,
            log_freq_params=log_freq_params,
            ploidy=ploidy,
            rate_constraint=rate_constraint,
            final_rp_norm=final_rp_norm,
        )

        branch_lengths[0] = 0.0
        branch_lengths[1:] = np.clip(
            np.nan_to_num(res.x[num_rate_params + 1 :]), min_branch_length, np.inf
        )

    except ValueError:
        pass

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
    pis,
    sequences,
    taxa,
    true_tree,
    model,
    log: bool = False,
    min_branch_length: float = -np.inf,
):
    import functools
    import itertools

    import numpy as np
    from scipy.optimize import minimize

    from pruning.path_constraints import make_path_constraints
    from pruning.util import CallbackParam, solver_options

    # noinspection PyUnreachableCode
    match model:
        case "DNA":
            from pruning.distance_functions import haploid_sequence_distance

            sequence_distance = functools.partial(haploid_sequence_distance, pis=pis)
        case "PHASED_DNA16" | "PHASED_DNA16_MP":
            from pruning.distance_functions import phased_diploid_sequence_distance

            sequence_distance = functools.partial(phased_diploid_sequence_distance, pis=pis)
        case "UNPHASED_DNA":
            from pruning.distance_functions import unphased_diploid_sequence_distance

            sequence_distance = functools.partial(unphased_diploid_sequence_distance, pis=pis)
        case "CELLPHY":
            from pruning.distance_functions import cellphy_unphased_diploid_sequence_distance

            sequence_distance = functools.partial(
                cellphy_unphased_diploid_sequence_distance, pis=pis
            )
        case "GTR10Z":
            from pruning.distance_functions import gtr10z_sequence_distance

            sequence_distance = functools.partial(gtr10z_sequence_distance, pis=pis)
        case "GTR10":
            from pruning.distance_functions import gtr10_sequence_distance

            sequence_distance = functools.partial(gtr10_sequence_distance, pis=pis)
        case _:
            assert False

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
    leaf_variances = np.maximum(1e-8, leaf_stats[:, 1])

    if min_branch_length < 0.0:
        # noinspection PyTypeChecker
        min_branch_length = 1e-5 * np.mean(leaf_distances)

    leaf_distances = np.maximum(min_branch_length, leaf_distances)
    leaf_variances /= np.mean(leaf_variances)

    # create the constraint matrices
    constraints_eqn, constraints_val = make_path_constraints(
        true_tree, num_tree_nodes, leaf_distances, pair_to_idx, node_indices
    )

    branch_len_initial_guess = np.linalg.pinv(constraints_eqn) @ constraints_val
    branch_len_initial_guess[0] = 0.0
    branch_len_initial_guess[1:] = np.maximum(min_branch_length, branch_len_initial_guess[1:])

    def branch_length_estimate_objective_prototype(
        x, *, constraints_eqn, constraints_val, leaf_variances
    ):
        # prediction error (weighted by variance) plus a regularizing term
        prediction_err = (constraints_eqn @ x - constraints_val)[1:]
        return np.mean(prediction_err**2 / leaf_variances) + x[0] ** 2

    branch_length_estimate_objective = functools.partial(
        branch_length_estimate_objective_prototype,
        constraints_eqn=constraints_eqn,
        constraints_val=constraints_val,
        leaf_variances=leaf_variances,
    )
    res = minimize(
        branch_length_estimate_objective,
        # np.ones(num_tree_nodes, dtype=np.float64),
        branch_len_initial_guess,
        bounds=[(0.0, max(min_branch_length / 2, 1e-10))]
        + [(min_branch_length, np.inf)] * (num_tree_nodes - 1),
        callback=CallbackParam(print_period=10) if log else None,
        method="L-BFGS-B",
        options=dict({"maxfun": np.inf}, **solver_options["L-BFGS-B-Heavy"]),
    )
    print(res)
    if log and not res.success:
        print("Optimization did not terminate, continuing anyway", flush=True)

    # belt and suspenders for the constraint (avoid -1e-big type bounds violations)
    tree_distances = np.zeros_like(res.x, dtype=np.float64)
    tree_distances[1:] = np.maximum(min_branch_length, np.nan_to_num(res.x[1:]))

    return tree_distances


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
