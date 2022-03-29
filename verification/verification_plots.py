import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import utils
from definitions import input_lower_bound, input_upper_bound


def compute_plotting_grid(reference_point, N_grid_points, x_dimension_idx, y_dimension_idx):
    """
    Create the basic grid for contour plots.
    :param reference_point: A row vector containing the reference value that will be repeated. The columns of
    x_dimension_idx and y_dimension_idx will be replaced by an equally spaced grid
    :param N_grid_points: number of equally spaced values on the x- and y-axis
    :param x_dimension_idx: index of the column whose value serve as the x-axis
    :param y_dimension_idx: index of the column whose value serve as the y-axis
    :return: the input data in form of one data point per row (input_grid) and the 2D meshgrid for x and y
    """
    input_grid = np.ones((N_grid_points ** 2, reference_point.shape[1])) * reference_point

    grid_values_x = np.linspace(input_lower_bound[0, x_dimension_idx], input_upper_bound[0, x_dimension_idx],
                                N_grid_points)
    grid_values_y = np.linspace(input_lower_bound[0, y_dimension_idx], input_upper_bound[0, y_dimension_idx],
                                N_grid_points)

    input_2D_grid_x, input_2D_grid_y = np.meshgrid(grid_values_x, grid_values_y)

    input_grid[:, x_dimension_idx:x_dimension_idx + 1] = input_2D_grid_x.reshape((-1, 1))
    input_grid[:, y_dimension_idx:y_dimension_idx + 1] = input_2D_grid_y.reshape((-1, 1))

    return input_grid, input_2D_grid_x, input_2D_grid_y


def plot_verified_region_for_set_point(input_2D_grid_x, input_2D_grid_y, network_prediction_grid,
                                       reference_points, epsilon_values, resulting_points,
                                       x_dimension_idx, y_dimension_idx, x_label, y_label, plot_title):
    """
    Generic contour plot given adjustable values in two dimensions. It shows the network prediction as a coloured
    contour plot as well as the verified regions (dashed lines) around one or multiple reference point (white crosses)
    and the point that solved the optimisation problem (black dot).
    """
    plt.rcParams["font.family"] = "serif"
    plt.rcParams['text.usetex'] = True
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rc('font', size=12)  # controls default text size
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["figure.frameon"] = False

    fig0, ax0 = plt.subplots(1, 1, constrained_layout=True, figsize=(4, 3))
    cf0 = ax0.contourf(input_2D_grid_x,
                       input_2D_grid_y,
                       network_prediction_grid, np.arange(-3.0, 9.0, 0.05),
                       extend='both', cmap='seismic_r')
    cbar0 = plt.colorbar(cf0, )

    for reference_point, epsilon, resulting_point in zip(reference_points, epsilon_values, resulting_points):
        region_width = epsilon * 2 * (input_upper_bound[0, x_dimension_idx] - input_lower_bound[0, x_dimension_idx])
        region_height = epsilon * 2 * (input_upper_bound[0, y_dimension_idx] - input_lower_bound[0, y_dimension_idx])

        ax0.scatter(reference_point[x_dimension_idx], reference_point[y_dimension_idx], c='white', s=20.0, marker="x")
        ax0.add_patch(Rectangle((reference_point[x_dimension_idx] - region_width / 2,
                                 reference_point[y_dimension_idx] - region_height / 2),
                                region_width,
                                region_height,
                                edgecolor='white',
                                fill=False,
                                lw=1,
                                linestyle='dashed'))

        ax0.scatter(resulting_point[x_dimension_idx],
                    resulting_point[y_dimension_idx],
                    c='black', s=30.0, marker="o")

    ax0.set_xlabel(f'{x_label}')
    ax0.set_ylabel(f'{y_label}')
    ax0.set_title(f'{plot_title}')
    ax0.set_xlim(xmin=input_lower_bound[0, x_dimension_idx], xmax=input_upper_bound[0, x_dimension_idx])
    ax0.set_ylim(ymin=input_lower_bound[0, y_dimension_idx], ymax=input_upper_bound[0, y_dimension_idx])
    cbar0.set_label('Damping ratio')

    return fig0


def plot_verified_region_power_set_point(verification_model, power_set_point, control_reference_points):
    """
    Plot for a given power set point and (multiple) control reference points.
    :param verification_model: The verification model
    :param power_set_point: A fixed power set point
    :param control_reference_points: An array of one or mulitple reference points for the verification
    :return: a figure
    """
    assert power_set_point.shape[0] == 1 and power_set_point.shape[1] == 2, \
        'Please specify the power set point as a row vector.'
    assert control_reference_points.shape[0] >= 1 and control_reference_points.shape[1] == 2, \
        'Please check the control reference point definition.'

    N_grid_points = 201
    x_dimension_idx = 2
    y_dimension_idx = 3

    reference_points = np.hstack(
        [np.ones((control_reference_points.shape[0], power_set_point.shape[1])) * power_set_point,
         control_reference_points])
    reference_point = reference_points[:1, :]

    input_grid, input_2D_grid_x, input_2D_grid_y = compute_plotting_grid(reference_point=reference_point,
                                                                         N_grid_points=N_grid_points,
                                                                         x_dimension_idx=x_dimension_idx,
                                                                         y_dimension_idx=y_dimension_idx)

    network_prediction = verification_model.predict(utils.ensure_tensor_array(input_array=input_grid))
    network_prediction_grid = utils.ensure_numpy_array(input_array=network_prediction).reshape(
        (N_grid_points, N_grid_points))

    verification_results = [verification_model.solve_for_maximum_control_parameters(reference_point=reference_point) for
                            reference_point in reference_points]
    epsilon_values = np.vstack([result[0] for result in verification_results])
    resulting_points = np.vstack([result[1] for result in verification_results])

    x_label = r'$K_{p,f}$'
    y_label = r'$K_v$'
    plot_title = fr'$P^*$, $Q^*$ = ({power_set_point[0, 0]}, {power_set_point[0, 1]}) p.u.'

    fig0 = plot_verified_region_for_set_point(input_2D_grid_x, input_2D_grid_y, network_prediction_grid,
                                              reference_points, epsilon_values, resulting_points,
                                              x_dimension_idx, y_dimension_idx, x_label, y_label, plot_title)
    return fig0


def plot_verified_region_control_set_point(verification_model, control_set_point, power_reference_points):
    """
    Plot for a given control set point and (multiple) power reference points.
    :param verification_model: The verification model
    :param control_set_point: A fixed control set point
    :param power_reference_points: An array of one or mulitple reference points for the verification
    :return: a figure
    """
    assert control_set_point.shape[0] == 1 and control_set_point.shape[1] == 2, \
        'Please specify the control set point as a row vector.'
    assert power_reference_points.shape[0] >= 1 and power_reference_points.shape[1] == 2, \
        'Please check the power reference point definition.'

    N_grid_points = 201
    x_dimension_idx = 0
    y_dimension_idx = 1

    reference_points = np.hstack(
        [power_reference_points,
         np.ones((power_reference_points.shape[0], control_set_point.shape[1])) * control_set_point])

    reference_point = reference_points[:1, :]

    input_grid, input_2D_grid_x, input_2D_grid_y = compute_plotting_grid(reference_point=reference_point,
                                                                         N_grid_points=N_grid_points,
                                                                         x_dimension_idx=x_dimension_idx,
                                                                         y_dimension_idx=y_dimension_idx)

    network_prediction = verification_model.predict(utils.ensure_tensor_array(input_array=input_grid))
    network_prediction_grid = utils.ensure_numpy_array(input_array=network_prediction).reshape(
        (N_grid_points, N_grid_points))

    verification_results = [verification_model.solve_for_maximum_power_parameters(reference_point=reference_point) for
                            reference_point in reference_points]
    epsilon_values = np.vstack([result[0] for result in verification_results])
    resulting_points = np.vstack([result[1] for result in verification_results])

    x_label = r'$P\,[\rm{p.u.}]$'
    y_label = r'$Q\,[\rm{p.u.}]$'
    K_pf_star = r'$K_{p,f}^*$'
    K_v_star = r'$K_{v}^*$'
    plot_title = f'{K_pf_star}, {K_v_star} = ({control_set_point[0, 0]}, {control_set_point[0, 1]}).'

    fig0 = plot_verified_region_for_set_point(input_2D_grid_x, input_2D_grid_y, network_prediction_grid,
                                              reference_points, epsilon_values, resulting_points,
                                              x_dimension_idx, y_dimension_idx, x_label, y_label, plot_title)
    return fig0
