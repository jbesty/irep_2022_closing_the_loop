import numpy as np
import torch


def ensure_numpy_array(input_array):
    """
    Conversion or assertion to or of a numpy array - don't overuse, it indicates improper handling of data types.
    :param input_array: any array
    :return: a numpy array or throws an error
    """
    if isinstance(input_array, np.ndarray):
        pass
    elif isinstance(input_array, torch.Tensor):
        input_array = input_array.detach().numpy()
    else:
        raise Exception('Allowed instances: np.ndarray and torch.Tensor')
    return input_array


def ensure_tensor_array(input_array):
    """
    Conversion or assertion to or of a torch tensor - don't overuse, it indicates improper handling of data types.
    :param input_array: any array
    :return: a torch tensor or throws an error
    """
    if isinstance(input_array, torch.Tensor):
        input_array_torch = input_array
    elif isinstance(input_array, np.ndarray):
        input_array_torch = torch.tensor(input_array, dtype=torch.float32)
    else:
        raise Exception('Allowed instances: np.ndarray and torch.Tensor')
    return input_array_torch
