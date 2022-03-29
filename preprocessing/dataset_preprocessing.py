import torch.utils.data


def split_dataset(dataset,
                  dataset_split_seed: int = 2,
                  training_share: float = 0.8,
                  validation_share: float = 0.2):
    """
    Function to split the dataset into training, validation, and testing subsets.
    In this implementation the dataset is only split into training and validation datasets as a separate testing dataset is used.
    :param dataset: Dataset (instance of a torch.utils.data.Dataset class)
    :param dataset_split_seed: random seed to govern the split
    :param training_share: [0; 1] share of training examples
    :param validation_share: [0; 1] share of validation examples
    :return: three datasets
    """

    n_training_points = int(len(dataset) * training_share)
    n_validation_points = int(len(dataset) * validation_share)
    n_testing_points = len(dataset) - n_training_points - n_validation_points

    training_data, validation_data, testing_data = torch.utils.data.random_split(dataset, [n_training_points,
                                                                                           n_validation_points,
                                                                                           n_testing_points],
                                                                                 generator=torch.Generator().manual_seed(
                                                                                     dataset_split_seed))

    return training_data, validation_data, testing_data


class DampingRatioDataset(torch.utils.data.Dataset):
    """
    Simple class that defines the dataset structure in the training process.
    Can the be used with DataLoader.
    :param input_data - operating point
    :param output_data - minimum damping ratio
    :param jacobian_data - jacobian of damping ratio wrt. the dimensions of the operating point

    """

    def __init__(self, input_data, output_data, jacobian_data):
        n_data_points = input_data.shape[0]
        n_input = input_data.shape[1]
        n_output = output_data.shape[1]
        self.input_data = torch.from_numpy(input_data).float()
        self.output_data = torch.from_numpy(output_data).float()
        self.jacobian = torch.from_numpy(jacobian_data.reshape((n_data_points, n_output, n_input))).float()

    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, idx):
        return self.input_data[idx, :], self.output_data[idx, :], self.jacobian[idx, :]
