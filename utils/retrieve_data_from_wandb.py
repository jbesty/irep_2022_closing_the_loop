import pathlib
from types import SimpleNamespace

import torch
import wandb

import neural_network_training
from definitions import wandb_entity, wandb_project, PROJECT_PATH


def rebuild_trained_model_from_cloud(run_id):
    api = wandb.Api()
    run = api.run(f'{wandb_entity}/{wandb_project}/{run_id}')
    config_dict = run.config

    config = SimpleNamespace(**config_dict)

    artifact = api.artifact(f'{wandb_entity}/{wandb_project}/model_{run_id}:v0', type='model')
    artifact_path = pathlib.Path(artifact.file(PROJECT_PATH))

    model = neural_network_training.NeuralNetwork(hidden_layer_size=config.hidden_layer_size,
                                                  n_hidden_layers=config.n_hidden_layers,
                                                  jacobian_regulariser=config.jacobian_regulariser,
                                                  pytorch_init_seed=config.pytorch_init_seed)

    model.load_state_dict(torch.load(artifact_path))
    model.eval()

    artifact_path.unlink()

    return model, config
