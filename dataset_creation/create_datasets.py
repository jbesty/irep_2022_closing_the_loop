import numpy as np
from pyDOE import lhs

import dataset_creation
import utils
from definitions import input_upper_bound, input_lower_bound, valid_sampling_methods

# %% Initialization ---------------------------------------------
# These are the models used for N-1 analysis
Nm1_models = dataset_creation.NSWPH_Initialize_Nm1_Models()


def create_input_OPs(N_points_per_dimension: int = 5, sampling_method: str = 'Grid', seed: int = None):
    """
    Basic sampler of input data, i.e., the operating points (OPs). At first sampling in the unscaled domain, i.e.,
    [0;1] and then scaling using the value of the domain bounds from definitions.py.

    :param N_points_per_dimension: Governing the number of samples, i.e., 4**N. For non-grid sampling, simply 4**N
    datapoints are sampled.
    :param sampling_method: Defining the sampling method - valid methods in definitions.py
    :param seed: Controlling the random sampling of the input OPs (if set)
    :return:
    """
    assert sampling_method in valid_sampling_methods, \
        f'Please choose a valid sampling method.\nAvailable: {valid_sampling_methods}'

    num_samples = N_points_per_dimension ** 4
    if seed is not None:
        np.random.seed(seed=seed)

    if sampling_method == 'Grid':
        base_linspace = np.linspace(0, 1, N_points_per_dimension)
        base_grid = np.meshgrid(base_linspace,
                                base_linspace,
                                base_linspace,
                                base_linspace)

        unscaled_samples = np.hstack([grid_dimension.reshape((-1, 1)) for grid_dimension in base_grid])
    elif sampling_method == 'LHC':
        unscaled_samples = lhs(n=4, samples=num_samples)
    elif sampling_method == 'Uniform':
        unscaled_samples = np.random.rand(num_samples, 4)
    else:
        raise Exception('Invalid sampling method.')

    input_data = input_lower_bound + unscaled_samples * (input_upper_bound - input_lower_bound)

    return input_data


def evaluate_input_OPs(input_data):
    """
    Evaluation of the physical model (NSWPH) for the given input operating points (OP).

    :param input_data: A numpy array with each row corresponding to an OP that consists of power set points and droop
    parameters.
    :return: The resulting values of the minimum damping ratio as well as the associated Jacobians and eigenvalues.
    """
    input_data = utils.ensure_numpy_array(input_data)

    output = [dataset_creation.NSWPH_Minimum_Data_Point(input_OP, Nm1_models) for
              input_OP in input_data]

    output_damping_ratios = np.vstack([output_tuple[0] for output_tuple in output])
    output_eigenvalues = np.vstack([output_tuple[1] for output_tuple in output])
    output_jacobians = np.vstack([output_tuple[2] for output_tuple in output])

    return output_damping_ratios, output_jacobians, output_eigenvalues


def create_and_save_dataset(N_points_per_dimension: int = 5, sampling_method: str = 'Grid', dataset_name: str = None,
                            seed: int = None) -> None:
    """
    Main dataset creation function which includes the sampling of the input operating points (OPs), their evaluation
    and saving the dataset.

    :param N_points_per_dimension: Governing the number of samples, i.e., 4**N. For non-grid sampling, simply 4**N
    datapoints are sampled.
    :param sampling_method: Defining the sampling method - valid methods in definitions.py
    :param dataset_name: Optional name of the resulting dataset.
    :param seed: Controlling the random sampling of the input OPs (if set)
    :return: None (but the newly created dataset is saved)
    """
    if dataset_name is None:
        dataset_name = f'{sampling_method}_{N_points_per_dimension}'

    print(f'Creating dataset {dataset_name}')

    input_data = create_input_OPs(N_points_per_dimension=N_points_per_dimension, sampling_method=sampling_method,
                                  seed=seed)

    output_damping_ratios, output_jacobians, output_eigenvalues = evaluate_input_OPs(input_data=input_data)

    utils.save_dataset(input_data=input_data,
                       output_data=output_damping_ratios,
                       jacobian_data=output_jacobians,
                       eigenvalue_data=output_eigenvalues,
                       dataset_name=dataset_name)

    print(f'Created the dataset "{dataset_name}".')

    pass


def create_and_save_DW_dataset(dataset_base_name: str) -> None:
    """
    Execute a directed walk (DW) on the points from the dataset 'dataset_base_name'

    :param dataset_base_name: the name of the dataset that serves as the starting point for the DWs
    :return: None (but the newly created dataset is saved)
    """
    print(f'Creating DW dataset for {dataset_base_name}')

    if '_5' not in dataset_base_name:
        raise Exception('DW intended only for datasets with 4**5 ("_5") due to the dataset naming convention.')

    input_data, output_data, jacobian_data, eigenvalues_data = utils.load_dataset(dataset_name=dataset_base_name)

    input_data_DW, output_data_DW, eigenvalues_data_DW, jacobian_data_DW = dataset_creation.NSWPH_Directed_Walk_Data(
        input_data, output_data,
        jacobian_data, Nm1_models)

    dataset_new_name = dataset_base_name.replace('_5', '_DW')
    utils.save_dataset(input_data=input_data_DW,
                       output_data=output_data_DW,
                       jacobian_data=jacobian_data_DW,
                       eigenvalue_data=eigenvalues_data_DW,
                       dataset_name=dataset_new_name)

    print(f'Created the dataset "{dataset_new_name}".')

    pass


if __name__ == "__main__":
    # Example of the 'Grid_5' dataset creation
    create_and_save_dataset(N_points_per_dimension=2,
                            sampling_method='Grid')
