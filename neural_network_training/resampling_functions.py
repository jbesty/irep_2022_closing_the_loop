import itertools

import numpy as np
import torch
import torch.utils.data
import wandb

import dataset_creation
import preprocessing
import utils
import verification
from definitions import N_network_informed_points, N_verification_informed_points, training_validation_split


def create_dataset_from_OPs(input_data):
    """
    Wrapper to turn input data into a dataset form usable for torch.
    :param input_data: np.array with a row for each operating points (OPs) and the columns represent the different variables of the OP
    :return: a dataset in form of the 'DampingRatioDataset' format
    """
    input_data = utils.ensure_numpy_array(input_data)
    output_damping_ratios, output_jacobians, _ = dataset_creation.evaluate_input_OPs(input_data=input_data)

    dataset = preprocessing.DampingRatioDataset(input_data=input_data,
                                                output_data=output_damping_ratios,
                                                jacobian_data=output_jacobians)

    return dataset


def create_additional_dataset(network, config):
    """
    Wrapper for the resampling action. Create the additional data by selection of the method and log the dataset
    characteristics.
    :param network: the neural network
    :param config: the general config file
    :return: the additional training and validation dataset
    """

    if '_NI' in config.dataset_name:
        additional_training_dataset, additional_validation_dataset = select_and_create_additional_dataset_NI(
            network=network, resampling_region=config.resampling_region)
    elif '_VI' in config.dataset_name:
        additional_training_dataset, additional_validation_dataset = select_and_create_additional_dataset_VI(
            network=network, resampling_region=config.resampling_region, n_threads=config.threads)
    else:
        raise Exception('Invalid additional dataset method.')

    training_added_classification_length_dict = utils.compute_region_classification_len(
        additional_training_dataset[:][1],
        dataset_type="training_added")
    validation_added_classification_length_dict = utils.compute_region_classification_len(
        additional_validation_dataset[:][1],
        dataset_type="validation_added")
    wandb.log(training_added_classification_length_dict, commit=False)
    wandb.log(validation_added_classification_length_dict, commit=True)

    return additional_training_dataset, additional_validation_dataset


def select_and_create_additional_dataset_NI(network, resampling_region):
    """
    Create an additional training and validation dataset according to the neural network informed procedure.
    :param network: the (trained) neural network
    :param resampling_region: the definition of the width of the margin
    :return: the additional training and validation datasets
    """

    print(f'Started the creation of {N_network_informed_points} additional data.')

    # evaluate and classify a large number of points using the neural network
    test_inputs = utils.scale_array_with_input_bounds(input_array=torch.rand((1000000, 4)))
    network.eval()
    with torch.no_grad():
        test_outputs = network(test_inputs)

    # lower_bound, upper_bound = utils.convert_resampling_region_to_bounds(resampling_region=resampling_region)
    bounds = utils.convert_resampling_region_to_bounds(resampling_region=resampling_region)
    bounds_tensor = utils.ensure_tensor_array(bounds)

    # indentify all indices corresponding to samples in the marginal region
    test_margin_indices = torch.where(torch.logical_and(torch.less_equal(bounds_tensor[0], test_outputs),
                                                        torch.less_equal(test_outputs, bounds_tensor[1])))[0]

    # sample from indices tensor and create the additional training and validation datasets accordingly
    N_sample_points = min(len(test_margin_indices), N_network_informed_points)
    sample_indices = torch.randint(low=0, high=len(test_margin_indices), size=(N_sample_points,))
    N_training_samples = int(training_validation_split * N_sample_points)
    training_indices, validation_indices = torch.split(test_margin_indices[sample_indices], [N_training_samples,
                                                                                             N_sample_points - N_training_samples])

    training_inputs = test_inputs[training_indices, :]
    validation_inputs = test_inputs[validation_indices, :]

    additional_training_dataset = create_dataset_from_OPs(training_inputs)
    additional_validation_dataset = create_dataset_from_OPs(validation_inputs)

    print('Finished the creation of additional data.')

    return additional_training_dataset, additional_validation_dataset


def select_and_create_additional_dataset_VI(network, resampling_region, n_threads):
    """
    Create an additional training and validation dataset according to the verification informed procedure.
    :param network: the (trained) neural network
    :param resampling_region: the definition of the width of the margin
    :param n_threads: number of threads used for solving the verification problem
    :return: the additional training and validation datasets
    """

    print(f'Started the creation of {N_verification_informed_points} additional data.')

    # lower_bound_resampling_region, upper_bound_resampling_region = \
    #     utils.convert_resampling_region_to_bounds(resampling_region=resampling_region)
    bounds_resampling = utils.convert_resampling_region_to_bounds(resampling_region=resampling_region)
    bounds_resampling_tensor = utils.ensure_tensor_array(bounds_resampling)

    # setup the verification model and perform a full bound tightening (can be replaced with a more efficient case
    # dependent bound tightening setup)
    verification_model = verification.VerificationModel(network=network,
                                                        margin=bounds_resampling,
                                                        n_threads=n_threads)

    verification_model.perform_full_bound_tightening()

    # define the reference points around which the neural network behaviour shall be verified
    # here: the corner points of the hypercube
    reference_points_unscaled_list = list(itertools.product(*zip([0] * network.n_input_neurons,
                                                                 [1] * network.n_input_neurons)))
    reference_points_unscaled = torch.vstack(
        [torch.tensor(corner_point).reshape((1, -1)) for corner_point in reference_points_unscaled_list])
    reference_points_scaled = utils.scale_array_with_input_bounds(reference_points_unscaled)

    with torch.no_grad():
        network.eval()
        reference_points_predictions = network(utils.ensure_tensor_array(reference_points_scaled))

    # perform a statistical estimation of the radius (for reference and as a fallback, also a good check for the
    # verification results)
    test_points_unscaled = torch.rand(1000000, 4)

    radii_min_unscaled_statistically = torch.vstack(
        [utils.get_radius_from_reference_point_statistical(network=network,
                                                           reference_point_unscaled=reference_point_unscaled,
                                                           resampling_region=resampling_region,
                                                           test_points_unscaled=test_points_unscaled)
         for reference_point_unscaled in reference_points_unscaled])

    # The computational effort is high for the cases below, hence the size of the verified region is estimated by a
    # statistical evaluation which gives an upper bound on the verified region.
    if (network.n_hidden_layers == 2 and network.hidden_layer_size >= 64) or (
            network.n_hidden_layers == 3 and network.hidden_layer_size >= 64) or (
            network.n_hidden_layers == 4 and network.hidden_layer_size >= 32):
        replace_verification_with_statistical_evaluation = True
    else:
        replace_verification_with_statistical_evaluation = False

    # compute the minimum radii around the
    if replace_verification_with_statistical_evaluation:
        radii_min_unscaled = torch.zeros((reference_points_unscaled.shape[0], 1))
        radii_min_unscaled_used = radii_min_unscaled_statistically
    else:
        radii_min_unscaled_np = np.vstack(
            [verification_model.solve_for_largest_region_around_reference_point(reference_point=reference_point_scaled)[
                 0]
             for reference_point_scaled in reference_points_scaled])

        radii_min_unscaled = utils.ensure_tensor_array(radii_min_unscaled_np)
        radii_min_unscaled_used = radii_min_unscaled

    # log the results, split into stable and unstable cases
    stable_indices = utils.return_index_if_value_in_region(reference_points_predictions[:, 0],
                                                           lower_bound=bounds_resampling_tensor[1])
    unstable_indices = utils.return_index_if_value_in_region(reference_points_predictions[:, 0],
                                                             upper_bound=bounds_resampling_tensor[0])

    wandb.log({"stable_radii_statistical": radii_min_unscaled_statistically * stable_indices.reshape((-1, 1)),
               "unstable_radii_statistical": radii_min_unscaled_statistically * unstable_indices.reshape((-1, 1)),
               "stable_radii_verification": radii_min_unscaled * stable_indices.reshape((-1, 1)),
               "unstable_radii_verification": radii_min_unscaled * unstable_indices.reshape((-1, 1)),
               },
              commit=True)

    # sample a number of input points and then disregard all within the verified ball (infinity norm)
    candidate_points_unscaled = torch.rand((100000, 4))

    for reference_points_unscaled, radius_min_unscaled in zip(reference_points_unscaled, radii_min_unscaled_used):
        candidate_points_unscaled = utils.disregard_points_within_ball(data_unscaled=candidate_points_unscaled,
                                                                       reference_point_unscaled=reference_points_unscaled,
                                                                       distance=radius_min_unscaled)

    # sample points from the candidate points and create the additional training and validation datasets accordingly
    N_sample_points = min(len(candidate_points_unscaled), N_verification_informed_points)
    sample_indices = torch.randint(low=0, high=len(candidate_points_unscaled), size=(N_sample_points,))
    N_training_samples = int(0.8 * N_sample_points)

    training_indices, validation_indices = torch.split(sample_indices, [N_training_samples,
                                                                        N_sample_points - N_training_samples])

    selected_points_scaled = utils.scale_array_with_input_bounds(candidate_points_unscaled)

    training_inputs = selected_points_scaled[training_indices, :]
    validation_inputs = selected_points_scaled[validation_indices, :]

    additional_training_dataset = create_dataset_from_OPs(training_inputs)
    additional_validation_dataset = create_dataset_from_OPs(validation_inputs)

    print('Finished the creation of additional data.')
    return additional_training_dataset, additional_validation_dataset
