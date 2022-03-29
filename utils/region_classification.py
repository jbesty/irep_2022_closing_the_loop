import numpy as np
import torch


def return_index_if_value_in_region(value, lower_bound=None, upper_bound=None):
    """
    Return the index if the value lies in between the upper and lower bound.
    """
    if lower_bound is None:
        lower_bound = torch.tensor(-np.inf)
    if upper_bound is None:
        upper_bound = torch.tensor(np.inf)

    value_in_region = torch.where(torch.logical_and(lower_bound < value,
                                                    value <= upper_bound),
                                  True, False)

    return value_in_region


def convert_resampling_region_to_bounds(resampling_region: str):
    """
    Definition of the resampling regions in terms of upper and lower bound
    """
    if resampling_region == 'marginal_tight':
        bounds = np.array([2.75, 3.25])
    elif resampling_region == 'marginal_wide':
        bounds = np.array([0.0, 6.0])
    else:
        raise Exception('Please specify a valid resampling region.')

    return bounds


def compute_region_classification_len(dataset_output,
                                      dataset_type: str):
    """
    Compute the number of points per class and return a dictionary (dataset_type specifies the keys) with the results
    """

    stable_region_indices, marginal_stable_region_indices, marginal_region_indices, marginal_unstable_region_indices, unstable_region_indices = compute_regions_belongings(
        value=dataset_output)

    region_len_dict = {f"len_{dataset_type}_stable_region": sum(stable_region_indices),
                       f"len_{dataset_type}_marginal_stable_region": sum(marginal_stable_region_indices),
                       f"len_{dataset_type}_marginal_region": sum(marginal_region_indices),
                       f"len_{dataset_type}_marginal_unstable_region": sum(marginal_unstable_region_indices),
                       f"len_{dataset_type}_unstable_region": sum(unstable_region_indices),
                       }

    return region_len_dict


def compute_regions_belongings(value):
    """
    Classify the damping ratio according to the 5 defined classes and return the indices.
    """
    stable_region_indices = return_index_if_value_in_region(value=value, lower_bound=6.0)
    marginal_stable_region_indices = return_index_if_value_in_region(value=value, lower_bound=3.25, upper_bound=6.0)
    marginal_region_indices = return_index_if_value_in_region(value=value, lower_bound=2.75, upper_bound=3.25)
    marginal_unstable_region_indices = return_index_if_value_in_region(value=value, lower_bound=0.0, upper_bound=2.75)
    unstable_region_indices = return_index_if_value_in_region(value=value, upper_bound=0.0)

    return stable_region_indices, marginal_stable_region_indices, marginal_region_indices, marginal_unstable_region_indices, unstable_region_indices
