import numpy as np

import preprocessing
import utils
from definitions import training_validation_split


def build_dataset(dataset_name, dataset_split_seed):
    """
    Returns a training and a validation dataset at the ratio 80/20 in the 'DampingRatioDataset' (-> see preprocessing) format.
    :param dataset_name: The name as save under 'data' or a combination with the additional suffix
    :param dataset_split_seed: governs the random split between training and validation dataset
    :return: training and validation dataset
    """
    if '_NI' in dataset_name:
        dataset_name_data = dataset_name.replace('_NI', '_5')
        print(f'Using {dataset_name_data} as base dataset.')
    elif '_VI' in dataset_name:
        dataset_name_data = dataset_name.replace('_VI', '_5')
        print(f'Using {dataset_name_data} as base dataset.')
    elif '_PR' in dataset_name:
        dataset_name_data = dataset_name.replace('_PR', '_5')
        print(f'Using {dataset_name_data} as base dataset.')
    else:
        dataset_name_data = dataset_name

    input_data, output_data, jacobian_data, eigenvalues_data = utils.load_dataset(dataset_name=dataset_name_data)

    if '_DW' in dataset_name:
        base_dataset_name = dataset_name.replace('_DW', '_5')
        input_data_base, output_data_base, jacobian_data_base, eigenvalues_data_base = utils.load_dataset(
            dataset_name=base_dataset_name)

        input_complete = np.concatenate([input_data, input_data_base], axis=0)
        output_complete = np.concatenate([output_data, output_data_base], axis=0)
        jacobian_complete = np.concatenate([jacobian_data, jacobian_data_base], axis=0)
        eigenvalues_complete = np.concatenate([eigenvalues_data, eigenvalues_data_base], axis=0)

    else:
        input_complete = input_data
        output_complete = output_data
        jacobian_complete = jacobian_data
        eigenvalues_complete = eigenvalues_data

    dataset = preprocessing.DampingRatioDataset(input_data=input_complete,
                                                output_data=output_complete,
                                                jacobian_data=jacobian_complete)

    training_data, validation_data, testing_data = preprocessing.split_dataset(dataset=dataset,
                                                                               dataset_split_seed=dataset_split_seed,
                                                                               training_share=training_validation_split,
                                                                               validation_share=1 - training_validation_split)

    return training_data, validation_data


def build_test_dataset():
    """
    Prepares the test dataset (saved under 'data/TEST')
    :return: test dataset
    """
    input_data, output_data, jacobian_data, eigenvalues_data = utils.load_dataset(dataset_name='TEST')

    dataset = preprocessing.DampingRatioDataset(input_data=input_data,
                                                output_data=output_data,
                                                jacobian_data=jacobian_data)

    return dataset
