import numpy as np
import torch

from definitions import input_lower_bound, input_upper_bound


def bound_format_checks(verbose=True):
    """
    The transformations rely on a certain format of the input bounds, namely:
    - being np.ndarray
    - the 0 dimension (reserved for the data point index)
    :return:
    """
    assert isinstance(input_lower_bound, np.ndarray) and isinstance(input_upper_bound, np.ndarray), \
        'Provide the input bounds as np.ndarray.'

    assert input_lower_bound.shape == input_upper_bound.shape, 'Mismatch in the shapes of the input bounds.'
    assert input_lower_bound.shape[0] == 1, 'Ensure that the length of the 0-dimension of the input bounds equals 1'

    assert np.alltrue(input_upper_bound - input_lower_bound != 0), 'Check the input bounds for difference 0.'

    if verbose:
        print('Input bounds checked successfully.')

    pass


bound_format_checks()


def transform_np_array_with_input_bounds(input_array, transformation: str):
    assert input_lower_bound.shape[1] == input_array.shape[1]
    assert input_upper_bound.shape[1] == input_array.shape[1]

    if transformation == 'normalise' or transformation == 'normalize':
        output_array = (input_array - input_lower_bound) / (input_upper_bound - input_lower_bound)
    elif transformation == 'scale':
        output_array = input_lower_bound + input_array * (input_upper_bound - input_lower_bound)
    else:
        raise Exception('Available transformations: "normalise" and "scale"')

    return output_array


def transform_array_with_input_bounds(input_array, transformation: str):
    if isinstance(input_array, np.ndarray):
        output_array = transform_np_array_with_input_bounds(input_array=input_array, transformation=transformation)
    elif isinstance(input_array, torch.Tensor):
        input_array_np = input_array.detach().numpy()
        output_array_np = transform_np_array_with_input_bounds(input_array=input_array_np,
                                                               transformation=transformation)
        output_array = torch.tensor(output_array_np, dtype=torch.float32)
    else:
        raise Exception('Array transformations only implemented for instances: np.ndarray and torch.Tensor')
    return output_array


def scale_array_with_input_bounds(input_array):
    return transform_array_with_input_bounds(input_array=input_array, transformation='scale')


def normalise_array_with_input_bounds(input_array):
    return transform_array_with_input_bounds(input_array=input_array, transformation='normalise')
