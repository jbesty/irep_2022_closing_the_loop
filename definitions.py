import pathlib

import numpy as np

# This file contains the relevant parameters that would be either hardcoded or that are not subject to change in the
# study. Furthermore, variables of 'global' interest are stored here.

PROJECT_PATH = pathlib.Path(__file__).parent
wandb_entity = 'jbest'
wandb_project = 'uncategorized'

input_upper_bound = np.array([[2, 0.5, 75, 50]])
input_lower_bound = np.array([[0, -0.5, 0, 0]])

# datasets that are stored in a folder - ideally writing a double check that these folders exist and contain all
# necessary files
available_datasets_data = ['Grid_5',
                           'Grid_6',
                           'Grid_7',
                           'Grid_8',
                           'Grid_DW',
                           'LHC_5',
                           'LHC_6',
                           'LHC_7',
                           'LHC_DW',
                           'TEST',
                           'Uniform_5',
                           'Uniform_6',
                           'Uniform_7',
                           'Uniform_DW']

# includes available_datasets_data and 'advanced' options that build on the basic dataset
valid_dataset_names = ['Grid_5',
                       'Grid_6',
                       'Grid_7',
                       'Grid_8',
                       'Grid_DW',
                       'Grid_PR',
                       'Grid_NI',
                       'Grid_VI',
                       'LHC_5',
                       'LHC_6',
                       'LHC_7',
                       'LHC_DW',
                       'LHC_PR',
                       'LHC_NI',
                       'LHC_VI',
                       'TEST',
                       'Uniform_5',
                       'Uniform_6',
                       'Uniform_7',
                       'Uniform_DW',
                       'Uniform_PR',
                       'Uniform_NI',
                       'Uniform_VI',
                       ]

valid_sampling_methods = ['Grid', 'LHC', 'Uniform']
training_validation_split: float = 0.8  # 80 % for training and 20 % for validation

# resampling specifications
N_network_informed_points: int = 200
N_verification_informed_points: int = 200

# NSWPH parameters
N_SC: int = 1  # number of synchronous condensers
N_OFF: int = 2  # number of offshore converters (HVDC links)
N_WF: int = 5  # number of wind farms

# Directed Walk information
min_damping = 3  # What is the stability margin?
walk_margin = 5  # How close should we be to take a walk?
damp_tol = 0.2  # When do we terminate the DW?
jac_tol = 0.01  # If gradient is smaller than this, don't walk
max_its = 25  # Maximum number of steps
