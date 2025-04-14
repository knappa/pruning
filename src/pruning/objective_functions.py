from typing import Union


def rate_param_objective_prototype(
    rate_params,
    log_freq_params,
    branch_lengths,
    *,
    gt_norm=False,
    neg_log_likelihood,
    rate_constraint,
    ploidy: Union[int, float],
):
    """
    Objective function for rate parameters.

    :param rate_params:
    :param log_freq_params: (fixed)
    :param branch_lengths: (fixed)
    :param gt_norm: if True, normalize the GT rate to 1
    :param neg_log_likelihood:
    :param rate_constraint:
    :param ploidy:
    :return: loss
    """
    import numpy as np

    if gt_norm:
        loss = neg_log_likelihood(log_freq_params, rate_params / rate_params[-1], branch_lengths)
        loss += (rate_params[-1] - 1) ** 2
    else:
        rate_constraint_val = rate_constraint(np.exp(log_freq_params), rate_params)
        rate_params_corrected = rate_params * ploidy / rate_constraint_val
        loss = neg_log_likelihood(log_freq_params, rate_params_corrected, branch_lengths)
        # fix the overall rate, if not normalizing on the GT rate
        loss += (rate_constraint_val - ploidy) ** 2
    return loss


def branch_length_objective_prototype(
    branch_lengths, log_freq_param, rate_params, *, neg_log_likelihood
):
    """
    Objective function for branch length estimation.

    :param branch_lengths:
    :param log_freq_param: (fixed)
    :param rate_params: (fixed)
    :param neg_log_likelihood:
    :return: loss
    """
    loss = neg_log_likelihood(log_freq_param, rate_params, branch_lengths)
    # zero length at root
    loss += branch_lengths[0] ** 2
    return loss


def param_objective_prototype(
    params,
    branch_lengths,
    gt_norm=False,
    *,
    num_freq_params: int,
    neg_log_likelihood,
    rate_constraint,
    ploidy: Union[int, float],
):
    """
    Objective function for rate parameters + frequencies holding branch lengths fixed

    :param params: frequencies+rate_params
    :param branch_lengths: (fixed)
    :param gt_norm: if True, normalize the GT rate to 1
    :param num_freq_params:
    :param neg_log_likelihood:
    :param rate_constraint:
    :param ploidy:
    :return: loss
    """
    import numpy as np
    from scipy.special import logsumexp

    log_freq_params = params[:num_freq_params]
    freq_error = logsumexp(log_freq_params)
    log_freq_params -= freq_error

    rate_params = params[num_freq_params:]

    if gt_norm:
        loss = neg_log_likelihood(log_freq_params, rate_params / rate_params[-1], branch_lengths)
        loss += (rate_params[-1] - 1) ** 2
    else:
        rate_constraint_val = rate_constraint(np.exp(log_freq_params), rate_params)
        rate_params_corrected = rate_params * ploidy / rate_constraint_val
        loss = neg_log_likelihood(log_freq_params, rate_params_corrected, branch_lengths)
        # fix the overall rate, if not normalizing on the GT rate
        loss += (rate_constraint_val - ploidy) ** 2
    # should be a probability dist
    # noinspection PyTypeChecker
    loss += freq_error**2
    return loss


def full_objective_prototype(
    params,
    *,
    gt_norm=False,
    num_freq_params: int,
    num_rate_params: int,
    neg_log_likelihood,
    rate_constraint,
    ploidy: Union[int, float],
):
    """
    Full objective function.

    :param params: first entries are frequencies, next entries are the model params, rest are branch lengths
    :param gt_norm: if True, normalize the GT rate to 1
    :param num_freq_params:
    :param num_rate_params:
    :param neg_log_likelihood:
    :param rate_constraint:
    :param ploidy:
    :return: loss
    """
    import numpy as np
    from scipy.special import logsumexp

    log_freq_params = params[:num_freq_params]
    freq_error = logsumexp(log_freq_params)
    log_freq_params -= freq_error

    rate_params = params[num_freq_params : num_freq_params + num_rate_params]
    branch_lengths = params[num_freq_params + num_rate_params :]

    if gt_norm:
        loss = neg_log_likelihood(log_freq_params, rate_params / rate_params[-1], branch_lengths)
        loss += (rate_params[-1] - 1) ** 2
    else:
        rate_constraint_val = rate_constraint(np.exp(log_freq_params), rate_params)
        rate_params_corrected = rate_params * ploidy / rate_constraint_val
        loss = neg_log_likelihood(log_freq_params, rate_params_corrected, branch_lengths)
        # fix the rate
        loss += (rate_constraint_val - ploidy) ** 2

    # should be a probability dist
    # noinspection PyTypeChecker
    loss += freq_error**2

    # zero length at root
    loss += branch_lengths[0] ** 2

    return loss


def rates_distances_objective_prototype(
    params,
    log_freq_params,
    *,
    gt_norm=False,
    num_rate_params,
    neg_log_likelihood,
    rate_constraint,
    ploidy: Union[int, float],
):
    """
    Objective function for holding the frequency parameters constant

    :param params: first entries are the model params, rest are branch lengths
    :param log_freq_params: state frequencies
    :param gt_norm: if True, normalize the GT rate to 1
    :param num_rate_params:
    :param neg_log_likelihood:
    :param rate_constraint:
    :param ploidy:
    :return: loss
    """
    import numpy as np

    rate_params = params[:num_rate_params]
    branch_lengths = params[num_rate_params:]

    if gt_norm:
        loss = neg_log_likelihood(log_freq_params, rate_params / rate_params[-1], branch_lengths)
        loss += (rate_params[-1] - 1) ** 2  # fix the rate
    else:
        rate_constraint_val = rate_constraint(np.exp(log_freq_params), rate_params)
        if rate_constraint_val > 0:
            rate_params_corrected = rate_params * ploidy / rate_constraint_val
            loss = neg_log_likelihood(log_freq_params, rate_params_corrected, branch_lengths)
        else:
            loss = neg_log_likelihood(log_freq_params, rate_params, branch_lengths)
        loss += (rate_constraint_val - ploidy) ** 2  # fix the rate

    loss += branch_lengths[0] ** 2  # zero length at root
    return loss
