import torch
import torch.nn.functional as F
import wandb

import neural_network_training
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_network(hidden_layer_size, n_hidden_layers, jacobian_regulariser, pytorch_init_seed):
    """
    Instantiate the neural network based on basic configuration parameters
    :param hidden_layer_size: number of neurons per hidden layers
    :param n_hidden_layers: number of hidden layers
    :param jacobian_regulariser: the strength of the regulariser on the Jacobian
    :param pytorch_init_seed: Initialisation seed for the weights and biases
    :return: the neural network
    """
    model = neural_network_training.NeuralNetwork(hidden_layer_size=hidden_layer_size,
                                                  n_hidden_layers=n_hidden_layers,
                                                  jacobian_regulariser=jacobian_regulariser,
                                                  pytorch_init_seed=pytorch_init_seed)

    return model.to(device)


def standardise_network(model, training_data):
    """
    Use the input and output statistics to adjust the standardisation constants (initially no transformation)
    :param model: neural network model
    :param training_data: the training data on which the standardisation is based
    :return: neural network model
    """
    input_statistics = torch.std_mean(training_data[:][0], dim=0, unbiased=False)
    output_statistics = torch.std_mean(training_data[:][1], dim=0, unbiased=False)

    model.standardise_input(input_statistics=input_statistics)
    model.scale_output(output_statistics=output_statistics)

    return model


def build_optimiser(network, learning_rate):
    """
    Instantiate the optimiser
    :param network: neural network for accessing the trainable parameters
    :param learning_rate: initial learning rate
    :return: optimiser instance
    """
    optimiser = torch.optim.Adam(network.parameters(),
                                 lr=learning_rate)

    return optimiser


def build_scheduler(optimiser, learning_rate_decay):
    """
    Instantiate the scheduler of the learning rate
    :param optimiser: the associated optimiser
    :param learning_rate_decay: value of the learning rate decay
    :return: scheduler instance
    """
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=learning_rate_decay)

    return scheduler


def compute_regional_loss(loss, target):
    """
    Compute the loss separated by the 5 defined regions
    :param loss: loss per data point
    :param target: prediction (indicates the region)
    :return: loss values per region (5 elements)
    """
    stable_region_indices, marginal_stable_region_indices, marginal_region_indices, \
    marginal_unstable_region_indices, unstable_region_indices = utils.compute_regions_belongings(
        value=target)

    loss_stable_region = torch.mean(loss[stable_region_indices])
    loss_marginal_stable_region = torch.mean(loss[marginal_stable_region_indices])
    loss_marginal_region = torch.mean(loss[marginal_region_indices])
    loss_marginal_unstable_region = torch.mean(loss[marginal_unstable_region_indices])
    loss_unstable_region = torch.mean(loss[unstable_region_indices])

    return loss_stable_region, loss_marginal_stable_region, loss_marginal_region, loss_marginal_unstable_region, loss_unstable_region


def train_epoch(network, loader, optimiser):
    """
    Single training epoch
    :param network: the neural network to be trained
    :param loader: the dataloader instance that contains the training data (potentially batched)
    :param optimiser: the optimiser specified for the provided neural network
    :return: dictionary containing the loss values
    """
    network.train()
    cumulative_loss = 0
    cumulative_loss_prediction = 0
    cumulative_loss_jacobian = 0
    cumulative_loss_stable = 0
    cumulative_loss_marginal_stable = 0
    cumulative_loss_marginal = 0
    cumulative_loss_marginal_unstable = 0
    cumulative_loss_unstable = 0

    # for loop to allow for batched data, len(loader) indicates the number of batches
    for _, (data, target, jacobian) in enumerate(loader):
        data, target, jacobian = data.to(device), target.to(device), jacobian.to(device)
        optimiser.zero_grad()

        # ➡ Forward pass
        prediction, jacobian_prediction = network.sensitivities(data)

        loss_full_prediction = F.mse_loss(prediction, target, reduction='none')
        loss_prediction = torch.mean(loss_full_prediction)

        with torch.no_grad():
            loss_stable_region, loss_marginal_stable_region, loss_marginal_region, loss_marginal_unstable_region, loss_unstable_region = compute_regional_loss(
                loss=loss_full_prediction,
                target=target)

        loss_jacobian_full = F.mse_loss(jacobian_prediction, jacobian[:, 0, :], reduction='none')
        loss_jacobian_point_wise = torch.linalg.norm(loss_jacobian_full, dim=1)
        loss_jacobian = torch.mean(loss_jacobian_point_wise)

        cumulative_loss += loss_prediction.item() + network.jacobian_regulariser * loss_jacobian.item()
        cumulative_loss_prediction += loss_prediction.item()
        cumulative_loss_jacobian += loss_jacobian.item()
        cumulative_loss_stable += loss_stable_region.item()
        cumulative_loss_marginal_stable += loss_marginal_stable_region.item()
        cumulative_loss_marginal += loss_marginal_region.item()
        cumulative_loss_marginal_unstable += loss_marginal_unstable_region.item()
        cumulative_loss_unstable += loss_unstable_region.item()

        loss = loss_prediction + network.jacobian_regulariser * loss_jacobian
        # ⬅ Backward pass + weight update
        loss.backward()
        optimiser.step()

    training_loss_dict = {"training_loss": cumulative_loss / len(loader),
                          "training_loss_prediction": cumulative_loss_prediction / len(loader),
                          "training_loss_jacobian": cumulative_loss_jacobian / len(loader),
                          "training_loss_stable_region": cumulative_loss_stable / len(loader),
                          "training_loss_marginal_stable_region": cumulative_loss_marginal_stable / len(loader),
                          "training_loss_marginal_region": cumulative_loss_marginal / len(loader),
                          "training_loss_marginal_unstable_region": cumulative_loss_marginal_unstable / len(loader),
                          "training_loss_unstable_region": cumulative_loss_unstable / len(loader),
                          }
    return training_loss_dict


def validate_epoch(network, loader, dict_prefix: str = "validation"):
    """
    Single validation epoch
    :param network: the neural network to be evaluated
    :param loader: the dataloader instance that contains the validation or testing data (potentially batched)
    :param dict_prefix: string to be added to the loss dictionary for clear naming
    :return: dictionary containing the loss values
    """
    network.eval()
    cumulative_loss = 0
    cumulative_loss_stable = 0
    cumulative_loss_marginal_stable = 0
    cumulative_loss_marginal = 0
    cumulative_loss_marginal_unstable = 0
    cumulative_loss_unstable = 0

    # for loop to allow for batched data, len(loader) indicates the number of batches
    for _, (data, target, _) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        # ➡ Forward pass
        with torch.no_grad():
            prediction = network(data)

            loss_full = F.mse_loss(prediction, target, reduction='none')
            loss = torch.mean(loss_full)
            loss_stable_region, loss_marginal_stable_region, loss_marginal_region, loss_marginal_unstable_region, loss_unstable_region = compute_regional_loss(
                loss=loss_full,
                target=target)
            cumulative_loss += loss.item()
            cumulative_loss_stable += loss_stable_region.item()
            cumulative_loss_marginal_stable += loss_marginal_stable_region.item()
            cumulative_loss_marginal += loss_marginal_region.item()
            cumulative_loss_marginal_unstable += loss_marginal_unstable_region.item()
            cumulative_loss_unstable += loss_unstable_region.item()

    loss_dict = {f"{dict_prefix}_loss": cumulative_loss / len(loader),
                 f"{dict_prefix}_loss_stable_region": cumulative_loss_stable / len(loader),
                 f"{dict_prefix}_loss_marginal_stable_region": cumulative_loss_marginal_stable / len(loader),
                 f"{dict_prefix}_loss_marginal_region": cumulative_loss_marginal / len(loader),
                 f"{dict_prefix}_loss_marginal_unstable_region": cumulative_loss_marginal_unstable / len(loader),
                 f"{dict_prefix}_loss_unstable_region": cumulative_loss_unstable / len(loader),
                 }

    return loss_dict


def execute_epoch(network, training_data_loader, validation_data_loader, optimiser, scheduler,
                  best_validation_loss, best_epoch, model_save_path):
    """
    Wrapper to execute one epoch including a training step, a validation step and relevant logging.
    """
    training_loss_dict = neural_network_training.train_epoch(network, training_data_loader, optimiser)
    validation_loss_dict = neural_network_training.validate_epoch(network, validation_data_loader,
                                                                  dict_prefix="validation")
    scheduler.step()

    if validation_loss_dict['validation_loss'] < best_validation_loss:
        best_epoch = network.epochs_total
        best_validation_loss = validation_loss_dict['validation_loss'].copy()
        torch.save(network.state_dict(), model_save_path)

    wandb.log(training_loss_dict, commit=False)
    wandb.log(validation_loss_dict, commit=False)
    wandb.log({"learning_rate": scheduler.get_last_lr()[0],
               "epoch": network.epochs_total,
               })

    network.epochs_total += 1

    return network, optimiser, scheduler, best_validation_loss, best_epoch
