def param_objective_prototype(
    model_params, log_pis, tree_distances, *, gt_norm=False, neg_log_likelihood, rate_constraint
):
    """
    Objective function for GTR parameters.

    :param model_params:
    :param log_pis:
    :param tree_distances: (fixed)
    :param gt_norm: if True, normalize the GT rate to 1
    :return: loss
    """
    import numpy as np

    if gt_norm:
        loss = neg_log_likelihood(log_pis, model_params / model_params[-1], tree_distances)
        loss += (model_params[-1] - 1) ** 2
    else:
        loss = neg_log_likelihood(log_pis, model_params, tree_distances)
        # fix the overall rate, if not normalizing on the GT rate
        loss += (rate_constraint(np.exp(log_pis), model_params) - 1) ** 2
    return loss


def branch_length_objective_prototype(tree_distances, log_pis, model_params, *, neg_log_likelihood):
    """
    Objective function for branch length estimation.

    :param tree_distances:
    :param log_pis:
    :param model_params: (fixed)
    :return: loss
    """
    loss = neg_log_likelihood(log_pis, model_params, tree_distances)
    # zero length at root
    loss += tree_distances[0] ** 2
    return loss


def full_param_objective_prototype(
    params, tree_distances, gt_norm=False, *, num_pis, neg_log_likelihood, rate_constraint
):
    """
    Objective function for model parameters + frequencies

    :param params: pis+model_params
    :param tree_distances: (fixed)
    :param gt_norm: if True, normalize the GT rate to 1
    :return: loss
    """
    import numpy as np
    from scipy.special import logsumexp

    log_pis = params[:num_pis]
    model_params = params[num_pis:]

    if gt_norm:
        loss = neg_log_likelihood(log_pis, model_params / model_params[-1], tree_distances)
        loss += (model_params[-1] - 1) ** 2
    else:
        loss = neg_log_likelihood(log_pis, model_params, tree_distances)
        # fix the overall rate, if not normalizing on the GT rate
        loss += (rate_constraint(np.exp(log_pis), model_params) - 1) ** 2
    # should be a probability dist
    # noinspection PyTypeChecker
    loss += logsumexp(log_pis, return_sign=False) ** 2
    return loss


def full_objective_prototype(
    params, *, gt_norm=False, num_pis, num_params, neg_log_likelihood, rate_constraint
):
    """
    Full objective function.

    :param params: first entries are frequencies, next entries are the model params, rest are branch lengths
    :param gt_norm: if True, normalize the GT rate to 1
    :return: loss
    """
    import numpy as np
    from scipy.special import logsumexp

    log_pis = params[:num_pis]
    model_params = params[num_pis : num_pis + num_params]
    tree_distances = params[num_pis + num_params :]

    if gt_norm:
        loss = neg_log_likelihood(log_pis, model_params / model_params[-1], tree_distances)
        loss += (model_params[-1] - 1) ** 2
    else:
        loss = neg_log_likelihood(log_pis, model_params, tree_distances)
        # fix the rate
        loss += (rate_constraint(np.exp(log_pis), model_params) - 1) ** 2

    # should be a probability dist
    # noinspection PyTypeChecker
    loss += logsumexp(log_pis) ** 2

    # zero length at root
    loss += tree_distances[0] ** 2

    return loss


def params_distances_objective_prototype(
    params, log_pis, *, gt_norm=False, num_params, neg_log_likelihood, rate_constraint
):
    """
    Full objective function.

    :param params: first entries are the model params, rest are branch lengths
    :param log_pis: state frequencies
    :param gt_norm: if True, normalize the GT rate to 1
    :return: loss
    """
    import numpy as np

    model_params = params[:num_params]
    tree_distances = params[num_params:]

    if gt_norm:
        loss = neg_log_likelihood(log_pis, model_params / model_params[-1], tree_distances)
        loss += (model_params[-1] - 1) ** 2  # fix the rate
    else:
        loss = neg_log_likelihood(log_pis, model_params, tree_distances)
        loss += (rate_constraint(np.exp(log_pis), model_params) - 1) ** 2  # fix the rate

    loss += tree_distances[0] ** 2  # zero length at root
    return loss
