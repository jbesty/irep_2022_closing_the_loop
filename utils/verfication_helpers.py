import numpy as np
import torch

import utils


def disregard_points_within_ball(data_unscaled, reference_point_unscaled, distance):
    """
    Remove all points that are within a specified distance around a reference point. Evaluated in normalised
    coordinates and using the infinity norm
    :param data_unscaled: points to be filtered
    :param reference_point_unscaled: the value from which the norm is measured
    :param distance: a scalar setting the discard threshold
    :return: the filter dataset
    """
    reference_point_normalised = reference_point_unscaled.reshape((1, -1))
    distance_norm = torch.linalg.norm(data_unscaled - reference_point_normalised, ord=torch.tensor(np.inf), axis=1)
    data_normalised_filtered = data_unscaled[distance_norm >= distance]

    return data_normalised_filtered


def get_radius_from_reference_point_statistical(network, reference_point_unscaled, resampling_region,
                                                test_points_unscaled):
    """
    Find the smallest distance from a reference point for which the prediction classification changes. A statistical
    estimate of the verification result
    :param network: the (trained) neural network
    :param reference_point_unscaled: the value from which the norm is measured
    :param resampling_region: defines the classification change
    :param test_points_unscaled: a set of points used for the estimate of the resulting radius
    :return: the estimated radius based on the given dataset
    """
    network.eval()
    # lower_bound, upper_bound = utils.convert_resampling_region_to_bounds(resampling_region=resampling_region)
    bounds = utils.convert_resampling_region_to_bounds(resampling_region=resampling_region)
    bounds_tensor = utils.ensure_tensor_array(bounds)

    reference_point_unscaled = utils.ensure_tensor_array(input_array=reference_point_unscaled.reshape((1, -1)))

    test_points_to_reference_point_max_norm = torch.linalg.norm(test_points_unscaled - reference_point_unscaled,
                                                                dim=1,
                                                                ord=torch.tensor(np.inf))

    reference_point_prediction = network(utils.scale_array_with_input_bounds(reference_point_unscaled))
    test_points_predictions = network(utils.scale_array_with_input_bounds(test_points_unscaled))

    if reference_point_prediction > bounds_tensor[1]:  # stable ( + marginal stable)
        indices_opposite_class = utils.return_index_if_value_in_region(test_points_predictions[:, 0],
                                                                       upper_bound=bounds_tensor[1])
        radius = min(test_points_to_reference_point_max_norm[indices_opposite_class])
    elif reference_point_prediction < bounds_tensor[0]:  # unstable ( + marginal unstable)
        indices_opposite_class = utils.return_index_if_value_in_region(test_points_predictions[:, 0],
                                                                       lower_bound=bounds_tensor[0])
        radius = min(test_points_to_reference_point_max_norm[indices_opposite_class])
    else:  # marginal
        radius = torch.tensor(0.0)

    return radius
