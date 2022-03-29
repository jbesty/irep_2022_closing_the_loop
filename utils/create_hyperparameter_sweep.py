import numpy as np
import wandb


def create_config_tuned_hyperparameters(dataset_name):
    """
    Create the config file with the best hyper-parameters for a specific dataset as found in the hyper-parameter tuning
    process and presented in the results of the paper. The resulting sweep contains 100 runs with different initialisations.
    """
    if dataset_name in ['LHC_NI']:
        n_hidden_layers = 2
    elif dataset_name in ['Grid_5', 'Grid_7', 'Grid_VI', 'LHC_6', 'LHC_7', 'LHC_VI', 'Uniform_DW',
                          'Uniform_NI', 'Uniform_VI', 'Uniform_PR']:
        n_hidden_layers = 3
    elif dataset_name in ['Grid_6', 'Grid_DW', 'Grid_NI', 'Grid_PR', 'LHC_5', 'LHC_DW', 'LHC_PR', 'Uniform_5',
                          'Uniform_6', 'Uniform_7']:
        n_hidden_layers = 4
    else:
        raise Exception(f'Please specify n_hidden_layers for {dataset_name}!')

    if dataset_name in ['Grid_VI']:
        hidden_layer_size = 16
    elif dataset_name in ['Grid_5', 'Grid_DW', 'Grid_NI', 'LHC_5', 'LHC_6', 'LHC_DW', 'LHC_NI', 'LHC_VI', 'Uniform_5',
                          'Uniform_6', 'Uniform_7', 'Uniform_DW', 'Uniform_VI', 'Uniform_PR']:
        hidden_layer_size = 32
    elif dataset_name in ['Grid_6', 'Grid_7', 'Grid_PR', 'LHC_7', 'LHC_PR', 'Uniform_NI']:
        hidden_layer_size = 64
    else:
        raise Exception(f'Please specify hidden_layer_size for {dataset_name}!')

    if dataset_name in ['LHC_PR']:
        learning_rate = 0.01
    elif dataset_name in ['Grid_6', 'Grid_7', 'Grid_PR', 'LHC_7', 'Uniform_PR', 'Uniform_NI']:
        learning_rate = 0.02
    elif dataset_name in ['Grid_5', 'Grid_DW', 'Grid_NI', 'Grid_VI', 'LHC_5', 'LHC_6', 'LHC_DW', 'LHC_NI', 'LHC_VI',
                          'Uniform_5',
                          'Uniform_6', 'Uniform_7', 'Uniform_DW', 'Uniform_VI']:
        learning_rate = 0.05
    else:
        raise Exception(f'Please specify learning_rate for {dataset_name}!')

    if '_PR' in dataset_name:
        jacobian_regulariser = 0.1
    else:
        jacobian_regulariser = 0.0

    if dataset_name in ['Grid_VI', 'LHC_VI', 'Uniform_NI']:
        resampling_region = 'marginal_tight'
    elif dataset_name in ['Grid_NI', 'LHC_NI', 'Uniform_VI']:
        resampling_region = 'marginal_wide'
    else:
        resampling_region = None

    pytorch_init_seeds = np.arange(100).tolist()

    sweep_config = {'program': 'workflow.py',
                    'method': 'grid',
                    'name': f'{dataset_name}_tuned'}

    metric = {
        'name': 'best_validation_loss',
        'goal': 'minimize'
    }

    parameters_dict = {
        'epochs': {'value': 3000},
        'epochs_initial_training': {'value': 1000},
        'hidden_layer_size': {'value': hidden_layer_size},
        'n_hidden_layers': {'value': n_hidden_layers},
        'learning_rate': {'value': learning_rate},
        'lr_decay': {'value': 1.0},
        'jacobian_regulariser': {'value': jacobian_regulariser},
        'resampling_region': {'value': resampling_region},
        'dataset_name': {'value': dataset_name},
        'dataset_split_seed': {'value': 10},
        'pytorch_init_seed': {'values': pytorch_init_seeds},
        'threads': {'value': 1},
    }

    sweep_config['parameters'] = parameters_dict
    sweep_config['metric'] = metric

    return sweep_config


def create_config_hyperparameter_search(dataset_name):
    """
    Create the config file for the hyper-parameter tuning given a dataset.
    """
    hidden_layer_size = [16, 32, 64]
    n_hidden_layers = [2, 3, 4]
    learning_rate = [0.005, 0.01, 0.02, 0.05]
    lr_decay = [0.99, 0.995, 0.999, 1.0]

    pytorch_init_seeds = np.arange(10).tolist()

    sweep_config = {'program': 'workflow.py',
                    'method': 'grid',
                    'name': f'{dataset_name}_tuning'}

    metric = {
        'name': 'best_validation_loss',
        'goal': 'minimize'
    }

    parameters_dict = {
        'epochs': {'value': 3000},
        'epochs_initial_training': {'value': 1000},
        'hidden_layer_size': {'values': hidden_layer_size},
        'n_hidden_layers': {'values': n_hidden_layers},
        'learning_rate': {'values': learning_rate},
        'lr_decay': {'values': lr_decay},
        'dataset_name': {'value': dataset_name},
        'dataset_split_seed': {'value': 10},
        'pytorch_init_seed': {'values': pytorch_init_seeds},
        'threads': {'value': 1},
    }

    if '_PR' in dataset_name:
        parameters_dict['jacobian_regulariser'] = {'values': [0.001, 0.01, 0.1, 1.0]}
    else:
        parameters_dict['jacobian_regulariser'] = {'value': 0.0}

    if '_NI' in dataset_name or '_VI' in dataset_name:
        parameters_dict['resampling_region'] = {'values': ['marginal_tight', 'marginal_wide']}
    else:
        parameters_dict['resampling_region'] = {'value': None}

    sweep_config['parameters'] = parameters_dict
    sweep_config['metric'] = metric

    return sweep_config


def setup_sweep(dataset_name: str, hyperparameters: str = 'tuned') -> str:
    """
    Create the sweep on WandB for hyper-parameter tuning or the tuned case.
    :param dataset_name:
    :param hyperparameters:
    :return:
    """
    wandb.login()
    if hyperparameters == 'tuned':
        sweep_config = create_config_tuned_hyperparameters(dataset_name=dataset_name)
    elif hyperparameters == 'search':
        sweep_config = create_config_hyperparameter_search(dataset_name=dataset_name)
    else:
        raise Exception('Please specify a valid hyperparameter setting routine.')

    sweep_id = wandb.sweep(sweep_config)
    return sweep_id


if __name__ == '__main__':
    dataset_names = [
        'Grid_5',
        # 'Grid_6',
        # 'Grid_7',
        # 'Grid_DW',
        'Grid_NI',
        # 'Grid_VI',
        # 'Grid_PR',
        # 'LHC_5',
        # 'LHC_6',
        # 'LHC_7',
        'LHC_DW',
        # 'LHC_NI',
        # 'LHC_VI',
        # 'LHC_PR',
        # 'Uniform_5',
        # 'Uniform_6',
        # 'Uniform_7',
        # 'Uniform_DW',
        # 'Uniform_NI',
        'Uniform_VI',
        'Uniform_PR',
    ]

    sweep_id_list = list()
    for dataset_name in dataset_names:
        sweep_id = setup_sweep(dataset_name=dataset_name, hyperparameters='tuned')
        sweep_id_list.append(sweep_id)
