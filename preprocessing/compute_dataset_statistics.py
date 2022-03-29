import numpy as np
import torch

import utils
from definitions import available_datasets_data


def compute_dataset_statistics() -> None:
    """
    A simple pre-processing function to compute the statistics of the available datasets.

    Prints the results
    :return: None
    """

    lengths_list = list()
    print('dataset_name - stable - marginal stable - marginal - marginal unstable - unstable')
    for dataset_name in available_datasets_data:
        _, output_data, _, _ = utils.load_dataset(dataset_name)
        region_len_dict = utils.compute_region_classification_len(torch.tensor(output_data), "dataset")

        lengths_list.append(torch.concat(list(region_len_dict.values())))

    N_data_classes = torch.vstack(lengths_list).numpy()
    N_data_total = np.sum(N_data_classes, axis=1, keepdims=True)
    share_classes = N_data_classes / N_data_total * 100

    for dataset_name, N_points, shares in zip(available_datasets_data, N_data_total, share_classes):
        print(f'{dataset_name} - {N_points} - {np.round(shares, decimals=2)}')

    pass


if __name__ == '__main__':
    compute_dataset_statistics()
