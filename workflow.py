#!/usr/bin/env python3
import pathlib
import shutil
from types import SimpleNamespace

import numpy as np
import torch
import torch.utils.data

import neural_network_training
import utils
import verification
import wandb
from definitions import wandb_project, wandb_entity, valid_dataset_names


def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config, project=wandb_project, entity=wandb_entity) as run:
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        torch.set_num_threads(config.threads)

        # Setup of the logging
        model_artifact = wandb.Artifact(f'model_{run.id}', type='model')
        model_save_path = f'{run.dir}\\model.pth'
        best_validation_loss = 10000.0
        best_epoch = 0

        # %% Dataset generation ---------------------------------------------
        # In this case skipped by saving datasets (use utils.load_data_by_name for loading them)
        # If desired add a function that creates a dataset
        assert config.dataset_name in valid_dataset_names, 'The dataset specified in the config file is not ' \
                                                           'implemented. Check "valid_dataset_names" in ' \
                                                           '"definitions.py" '

        # %% Dataset preprocessing ---------------------------------------------
        training_data, validation_data = neural_network_training.build_dataset(dataset_name=config.dataset_name,
                                                                               dataset_split_seed=config.dataset_split_seed)

        training_data_loader = torch.utils.data.DataLoader(training_data, batch_size=len(training_data))
        validation_data_loader = torch.utils.data.DataLoader(validation_data, batch_size=len(validation_data))

        # %% Model training ---------------------------------------------
        # Basic setup of the model, optimizer and scheduler
        # (includes the standardisation  of the dataset, which could/should be moved to the pre-processing stage)

        network = neural_network_training.build_network(hidden_layer_size=config.hidden_layer_size,
                                                        n_hidden_layers=config.n_hidden_layers,
                                                        jacobian_regulariser=config.jacobian_regulariser,
                                                        pytorch_init_seed=config.pytorch_init_seed)
        network = neural_network_training.standardise_network(network, training_data)

        optimiser = neural_network_training.build_optimiser(network, config.learning_rate)
        scheduler = neural_network_training.build_scheduler(optimiser=optimiser, learning_rate_decay=config.lr_decay)

        # Training loop - with an interrupt if resampling is desired
        for epoch in range(config.epochs):

            if epoch == config.epochs_initial_training and (
                    '_NI' in config.dataset_name or '_VI' in config.dataset_name):
                network.load_state_dict(torch.load(model_save_path))
                network.eval()

                additional_training_dataset, additional_validation_dataset = \
                    neural_network_training.create_additional_dataset(network=network,
                                                                      config=config)

                # Combine the additional datasets with the existing dataset
                training_data = torch.utils.data.ConcatDataset([training_data, additional_training_dataset])
                validation_data = torch.utils.data.ConcatDataset([validation_data, additional_validation_dataset])

                training_data_loader = torch.utils.data.DataLoader(training_data,
                                                                   batch_size=len(training_data))
                validation_data_loader = torch.utils.data.DataLoader(validation_data,
                                                                     batch_size=len(validation_data))

            # executing a full epoch including a train, validation, logging
            network, optimiser, scheduler, best_validation_loss, best_epoch = \
                neural_network_training.execute_epoch(network, training_data_loader, validation_data_loader, optimiser,
                                                      scheduler, best_validation_loss, best_epoch, model_save_path)

        # %% Model assessment ---------------------------------------------
        # Only statistical evaluation here, could potentially be using verification
        # Logging of additionally interesting values (best validation loss and the corresponding training loss)
        network.load_state_dict(torch.load(model_save_path))
        network.eval()

        test_dataset = neural_network_training.build_test_dataset()
        test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset))

        testing_loss_dict = neural_network_training.validate_epoch(network, test_data_loader, dict_prefix="test")
        best_validation_loss_dict = neural_network_training.validate_epoch(network,
                                                                           validation_data_loader,
                                                                           dict_prefix="best_validation")

        best_training_loss_dict = neural_network_training.validate_epoch(network,
                                                                         training_data_loader,
                                                                         dict_prefix="best_training")

        wandb.log(testing_loss_dict, commit=False)
        wandb.log(best_validation_loss_dict, commit=False)
        wandb.log(best_training_loss_dict, commit=False)
        wandb.log({"best_epoch": best_epoch}, commit=True)

        # log the model
        model_artifact.add_file(model_save_path)
        run.log_artifact(model_artifact)
        run.finish()

    # clean the directory
    logs_directory = pathlib.Path(run.dir).parent
    shutil.rmtree(logs_directory)

    return run


def create_config():
    """
    A basic configuration file to train a single NN with the specified parameters.
    :return: config dictionary
    """
    parameters_dict = {
        'hidden_layer_size': 32,
        'n_hidden_layers': 2,
        'epochs': 3000,
        'learning_rate': 0.05,
        'lr_decay': 1.0,
        'dataset_name': 'Uniform_DW',
        'dataset_split_seed': 10,
        'pytorch_init_seed': 1,
        'epochs_initial_training': 1000,
        'jacobian_regulariser': 0.0,
        'resampling_region': 'marginal_tight',
        'threads': 16,
    }

    config = SimpleNamespace(**parameters_dict)

    return config


if __name__ == '__main__':
    wandb.login()
    run_config = create_config()
    run = train(config=run_config)

    network, config = utils.rebuild_trained_model_from_cloud(run_id=run.id)

    # set up the verification process
    verification_model = verification.VerificationModel(network=network, margin=np.array([2.75, 3.25]))
    verification_model.perform_full_bound_tightening()

    # obtain verified radii from the corner points
    corner_points_scaled, verified_radii, corner_predictions, resulting_point, resulting_point_prediction = \
        verification_model.solve_regions_around_corner_points()

    # test for a fixed power set point
    power_set_point = np.array([[2.0, 0.5]])
    control_reference_points = np.array([[25.0, 30.0],
                                         [40.0, 10.0]])
    control_plot = verification.plot_verified_region_power_set_point(verification_model=verification_model,
                                                                     power_set_point=power_set_point,
                                                                     control_reference_points=control_reference_points)
    control_plot.show()

    # test for a fixed control set point
    control_set_point = np.array([[60.0, 40.0]])
    power_reference_points = np.array([[0.0, 0.0],
                                       [1.0, 0.2]])
    power_plot = verification.plot_verified_region_control_set_point(verification_model=verification_model,
                                                                     control_set_point=control_set_point,
                                                                     power_reference_points=power_reference_points)

    power_plot.show()
