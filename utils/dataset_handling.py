import pickle

import numpy as np

from definitions import PROJECT_PATH, available_datasets_data


def load_dataset(dataset_name: str):
    """
    Load the dataset from the 'data' folder by name.
    :param dataset_name: Dataset name
    :return: the elements associated with the dataset (input, output, jacobian, and eigenvalues)
    """
    assert dataset_name in available_datasets_data, f'Specified dataset is currently not implemented!' \
                                                    f' \nChoose among {available_datasets_data} '

    with open(PROJECT_PATH / 'data' / f'{dataset_name}' / 'Input_Data.pkl', "rb") as file_path:
        input_data = pickle.load(file_path)

    with open(PROJECT_PATH / 'data' / f'{dataset_name}' / 'Output_Data.pkl', "rb") as file_path:
        output_data = pickle.load(file_path)

    with open(PROJECT_PATH / 'data' / f'{dataset_name}' / 'Jacobian_Data.pkl', "rb") as file_path:
        jacobian_data = pickle.load(file_path)

    with open(PROJECT_PATH / 'data' / f'{dataset_name}' / 'Eigenvalue_Data.pkl', "rb") as file_path:
        eigenvalues_data = pickle.load(file_path)

    return input_data, output_data, jacobian_data, eigenvalues_data


def concatenate_datasets(dataset_names):
    """
    Concatenate datasets from a list and remove duplicated data points
    :param dataset_names: List of dataset names
    :return: the elements associated with the concatenated dataset (input, output, jacobian, and eigenvalues)
    """
    input_data_list = list()
    output_data_list = list()
    jacobian_data_list = list()
    eigenvalue_data_list = list()

    for dataset_name in dataset_names:
        input_data, output_data, jacobian_data, eigenvalue_data = load_dataset(dataset_name=dataset_name)

        input_data_list.append(input_data)
        output_data_list.append(output_data)
        jacobian_data_list.append(jacobian_data)
        eigenvalue_data_list.append(eigenvalue_data)

    input_data_full = np.concatenate(input_data_list, axis=0)
    output_data_full = np.concatenate(output_data_list, axis=0)
    jacobian_data_full = np.concatenate(jacobian_data_list, axis=0)
    eigenvalue_data_full = np.concatenate(eigenvalue_data_list, axis=0)

    _, unique_indices = np.unique(input_data_full, axis=0, return_index=True)

    input_data_final = input_data_full[unique_indices, :]
    output_data_final = output_data_full[unique_indices, :]
    jacobian_data_final = jacobian_data_full[unique_indices, :]
    eigenvalue_data_final = eigenvalue_data_full[unique_indices, :]

    return input_data_final, output_data_final, jacobian_data_final, eigenvalue_data_final


def save_dataset(input_data, output_data, jacobian_data, eigenvalue_data, dataset_name):
    """
    Save a dataset consisting of (input, output, jacobian, and eigenvalues) to the 'data' folder and create the
    required folder (no overwritting if it exists)
    :param dataset_name: The name of the dataset and the associated folder
    """
    dataset_folder = PROJECT_PATH / 'data' / f'{dataset_name}'
    dataset_folder.mkdir(exist_ok=False)

    with open(dataset_folder / f'Input_Data.pkl', "wb") as file_path:
        pickle.dump(input_data, file_path)

    with open(dataset_folder / f'Output_Data.pkl', "wb") as file_path:
        pickle.dump(output_data, file_path)

    with open(dataset_folder / f'Jacobian_Data.pkl', "wb") as file_path:
        pickle.dump(jacobian_data, file_path)

    with open(dataset_folder / f'Eigenvalue_Data.pkl', "wb") as file_path:
        pickle.dump(eigenvalue_data, file_path)

    print(f'Created the dataset "{dataset_name}".')

    pass
