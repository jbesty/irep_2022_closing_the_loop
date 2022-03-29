import copy

import numpy as np
import scipy

import dataset_creation.NSWPH_initialize_state_vector as isv
from dataset_creation.NSWPH_models_linear import calc_state_matrix
from dataset_creation.NSWPH_system_models import nswph_models
from definitions import N_SC, N_WF, N_OFF, min_damping, walk_margin, damp_tol, jac_tol, max_its, input_upper_bound, \
    input_lower_bound


# %% Computation of damping ratios
def compute_eig_damping(eig_val):
    sigma = np.real(eig_val)
    omega = np.abs(np.imag(eig_val))

    # Compute the damping ratio
    damping = -100 * sigma / np.sqrt(sigma ** 2 + omega ** 2)

    # Output
    return damping


def NSWPH_Eigen_Decomposition(A):
    # Compute the eigen decomposition
    Lambda, Left, Right = scipy.linalg.eig(A, left=True)

    # Now, we have     np.diag(Lambda).dot(np.transpose(Left.conj())) == (np.transpose(Left).conj()).dot(A)
    # Now, we have     A.dot(Right)                                   == Right.dot(np.diag(Lambda))

    # First, make the left eigenvector matrix "conventional"
    Left_H = np.transpose(Left.conj())

    # Normalize Correctly!
    # W*A = Lam*W  =>  V*W*A = V*Lam*W  => V*W*A = A if wi*vi = 1, wj*vi = 0
    for ii in range(Lambda.size):
        Left_H[ii, :] = Left_H[ii, :] / (Left_H[ii, :].dot(Right[:, ii]))

    # Now, A = Right.dot(np.diag(Lambda)).dot(Left_H)
    Left = Left_H

    # Ouput
    return Lambda, Left, Right


def NSWPH_Data_Point(npr, models, M, input_OP, N_WF_OUT):
    WF_Power_Reference_Point, WF_Reactive_Reference_Point, WF_Droop_Kpf_Point, WF_Droop_Kv_Point = input_OP

    # Step 0: Multiply each point by a vector
    if N_WF_OUT == 0:
        N_WF_Total = N_WF - N_WF_OUT
    else:
        N_WF_Total = N_WF - 1
    unity_vec = np.ones(N_WF_Total, dtype='f')
    WF_Power_References = WF_Power_Reference_Point * unity_vec
    WF_Reactive_References = WF_Reactive_Reference_Point * unity_vec
    WF_Droop_Kpf = WF_Droop_Kpf_Point * unity_vec
    WF_Droop_Kv = WF_Droop_Kv_Point * unity_vec

    # Step 1: set Wind Farm active and reactive power
    Loss_factor = 0.95
    total_power_absolute = 0
    for kk in range(N_WF_Total):
        # This is defined in local per unit! Therefore, in the power flow solver,
        # it is transformed into a system per unit: WF_Power_Ref*S_local/S_system
        models[M['WF{:d}'.format(kk + 1)]].Qref = WF_Reactive_References[kk]
        models[M['WF{:d}'.format(kk + 1)]].Pref = WF_Power_References[kk]
        total_power_absolute += WF_Power_References[kk] * models[M['WF{:d}'.format(kk + 1)]].Sn

    # Step 2: set Off Shore power to balance the WF
    for kk in range(N_OFF):
        # This is defined in local per unit! Therefore, we should transform the
        # wind turbine power into the off-shore converter base :)
        models[M['OFF{:d}'.format(kk + 1)]].Pref = -Loss_factor * (total_power_absolute / N_OFF) / models[
            M['OFF{:d}'.format(kk + 1)]].Sn

    # Step 3: adjust the frequency droop parameters
    for kk in range(N_WF_Total):
        models[M['WF{:d}'.format(kk + 1)]].Kpf = WF_Droop_Kpf[kk]

    # Step 4: adjust voltage droop
    for kk in range(N_WF_Total):
        models[M['WF{:d}'.format(kk + 1)]].Kv = WF_Droop_Kv[kk]

    # Compute the state matrix
    x0, success = isv.initialize_state_vector(npr, models, pr=True)
    Amat_0 = calc_state_matrix(x0, npr, models)
    eigval, Left, Right = NSWPH_Eigen_Decomposition(Amat_0)

    # We only want to track the mode with the minimum damping ratio
    #
    # Find the zero mode corresponding eigenvalue
    ZMF = 2 * np.pi * 50
    eps_window = 1e-2
    T1 = np.imag(eigval) > (ZMF - eps_window)
    T2 = np.imag(eigval) < (ZMF + eps_window)
    T3 = np.real(eigval) < eps_window
    T4 = np.real(eigval) > -eps_window
    ZM_Eig = np.logical_and(np.logical_and(np.logical_and(T1, T2), T3), T4)
    eigval_noZM = eigval[np.logical_not(ZM_Eig)]

    # Looking in narrow frequency range
    LF = 2 * np.pi * 0.01
    HF = 2 * np.pi * 500

    # Track the LF mode
    eigval_LD_vec = copy.copy(eigval_noZM)
    eigval_LD_vec = eigval_LD_vec[np.imag(eigval_LD_vec) > LF]
    eigval_LD_vec = eigval_LD_vec[np.imag(eigval_LD_vec) < HF]
    eigval_LD_damping = compute_eig_damping(eigval_LD_vec)
    damping_local_ind = np.argmin(eigval_LD_damping)
    eigval_LD_damping = eigval_LD_damping[damping_local_ind]
    eigval_LD = eigval_LD_vec[damping_local_ind]
    eigval_LD_ind = np.argwhere(eigval == eigval_LD)[0][0]

    # Output
    Damping_Ratio = np.array([eigval_LD_damping])

    # Finally, compute the sensitivity of the damping ratio to the parameters
    Right_eigvec = Right
    Left_eigvec = Left

    # Grab the correct right eigenvectors
    Right_eigvec_LD = Right_eigvec[:, eigval_LD_ind]

    # Grab the correct left eigenvectors
    Left_eigvec_LD = Left_eigvec[eigval_LD_ind, :]

    # Base eigenvalues
    w_LD = np.imag(eigval_LD)
    sig_LD = np.real(eigval_LD)

    # Get matrix sensitivity:
    eps = 1e-3

    # Also, return the eigenvalues
    Eigenvalues = np.array([sig_LD, w_LD])

    Jacobian_deta_dp = np.zeros((1, 4))  # 1 damping ratios, 4 degrees of freedom
    esp_vec = eps * np.ones(N_WF_Total, dtype='f')

    # Loop and perturb!
    for ii in range(4):

        # List "models" cannot be deep copied, so let's just define all
        # parameters during each call.
        WF_Power_References_Pertubed = copy.deepcopy(WF_Power_References)
        WF_Reactive_References_Pertubed = copy.deepcopy(WF_Reactive_References)
        WF_Droop_Kpf_Perturbed = copy.deepcopy(WF_Droop_Kpf)
        WF_Droop_Kv_Perturbed = copy.deepcopy(WF_Droop_Kv)

        # Test index
        if ii == 0:
            # Perturb active power, in this case
            WF_Power_References_Pertubed += esp_vec
        elif ii == 1:
            # Perturb reactive power, in this case
            WF_Reactive_References_Pertubed += esp_vec
        elif ii == 2:
            # Perturb Kpf droop parameters, in this case
            WF_Droop_Kpf_Perturbed += esp_vec
        else:
            # Perturb Kv droop parameters, in this case
            WF_Droop_Kv_Perturbed += esp_vec

        # Update models
        total_power_absolute = 0
        for kk in range(N_WF_Total):
            models[M['WF{:d}'.format(kk + 1)]].Pref = WF_Power_References_Pertubed[kk]
            total_power_absolute += WF_Power_References_Pertubed[kk] * models[M['WF{:d}'.format(kk + 1)]].Sn
        for kk in range(N_OFF):
            models[M['OFF{:d}'.format(kk + 1)]].Pref = -Loss_factor * (total_power_absolute / N_OFF) / models[
                M['OFF{:d}'.format(kk + 1)]].Sn
        for kk in range(N_WF_Total):
            models[M['WF{:d}'.format(kk + 1)]].Qref = WF_Reactive_References_Pertubed[kk]
        for kk in range(N_WF_Total):
            models[M['WF{:d}'.format(kk + 1)]].Kpf = WF_Droop_Kpf_Perturbed[kk]
        for kk in range(N_WF_Total):
            models[M['WF{:d}'.format(kk + 1)]].Kv = WF_Droop_Kv_Perturbed[kk]

        # Compute the state matrix
        x0, success = isv.initialize_state_vector(npr, models, pr=True)
        Amat_Pert = calc_state_matrix(x0, npr, models)

        # Estimate matrix derivative
        dA_dp = (Amat_Pert - Amat_0) / eps

        # Compute the eigenvalue sensitivity
        dlamLD_dp = Left_eigvec_LD.dot(dA_dp).dot(Right_eigvec_LD) / (Left_eigvec_LD.dot(Right_eigvec_LD))

        # Compute the damping ratio sensitivity
        dw_LD = np.imag(dlamLD_dp)
        dsig_LD = np.real(dlamLD_dp)

        detaLD_dp = 100 * w_LD * (sig_LD * dw_LD - w_LD * dsig_LD) / np.power(sig_LD ** 2 + w_LD ** 2, 3 / 2)

        # Append into a matrix
        Jacobian_deta_dp[:, ii] = np.array([detaLD_dp])

        ''' ------ Direct numerical comparison ------ 

        # --- Written for LF/HF eigenmode tracking ---

        eigval,Left,Right = NSWPH_Eigen_Decomposition(Amat_Pert)

        # Find the zero mode corresponding eigenvalue
        ZMF         = 2*np.pi*50
        eps_ZM      = 1e-4
        T1          = np.imag(eigval)>(ZMF-eps_ZM)
        T2          = np.imag(eigval)<(ZMF+eps_ZM)
        T3          = np.real(eigval)<eps_ZM
        T4          = np.real(eigval)>-eps_ZM
        ZM_Eig      = np.logical_and(np.logical_and(np.logical_and(T1,T2),T3),T4)
        eigval_noZM = eigval[np.logical_not(ZM_Eig)]

        # Define low and high frequency modes
        eigval_LF = copy.copy(eigval_noZM)
        eigval_HF = copy.copy(eigval_noZM)

        # Track the LF mode
        eigval_LF         = eigval_LF[np.imag(eigval_LF)>LF]
        eigval_LF         = eigval_LF[np.imag(eigval_LF)<MF]
        eigval_LF_damping_NEW = compute_eig_damping(eigval_LF)
        damping_local_ind = np.argmin(eigval_LF_damping_NEW)
        eigval_LF_damping_NEW = eigval_LF_damping_NEW[damping_local_ind]

        # Track the HF mode
        eigval_HF         = eigval_HF[np.imag(eigval_HF)>MF]
        eigval_HF         = eigval_HF[np.imag(eigval_HF)<HF]
        eigval_HF_damping_NEW = compute_eig_damping(eigval_HF)
        damping_local_ind     = np.argmin(eigval_HF_damping_NEW)
        eigval_HF_damping_NEW = eigval_HF_damping_NEW[damping_local_ind]

        # Comapre
        Jac_row0_entry  = (eigval_LF_damping_NEW - eigval_LF_damping)/eps
        Jac_row1_entry  = (eigval_HF_damping_NEW - eigval_HF_damping)/eps'''

    return Damping_Ratio, Eigenvalues, Jacobian_deta_dp


# %% Initialize Models
def NSWPH_Initialize_Nm1_Models():
    # Initialize
    Nm1_models = []

    # Loop over all N-1 contingencies
    for ii in range(N_WF + 1):
        # This turns off each turbine (ii=0, all on)
        N_WF_OUT = ii

        #####################################################
        # Each time N_WF_OUT changes, you must re-initialize!
        #####################################################
        npr, models, M = nswph_models(N_SC, N_OFF, N_WF, N_WF_OUT, Pwf=0.0, onshore=False)

        # Now, add these models to a list of
        NSWPH = [npr, models, M]

        # Now add this list to a list of lists
        Nm1_models.append(NSWPH)

    # Output
    return Nm1_models


# %% Return the smallest damping ratio across all trials
def NSWPH_Minimum_Data_Point(input_OP, Nm1_models):
    # Loop over all N-1 contingencies (loss of a turbine)
    # This turns off each turbine (ii=0, all on)
    # N_WF_OUT = ii

    #####################################################
    # Each time N_WF_OUT changes, you must re-initialize!
    #####################################################
    #
    # Option 1: call the following code and directly build the models:
    #    npr, models, M = nswph_models(N_SC, N_OFF, N_WF, N_WF_OUT, Pwf=0.0, onshore=False)
    #
    # Option 2: call the pre-built models

    # Compute damping etc.
    results_list = [NSWPH_Data_Point(Nm1_models[N_WF_OUT][0],
                                     Nm1_models[N_WF_OUT][1], Nm1_models[N_WF_OUT][2],
                                     input_OP, N_WF_OUT) for N_WF_OUT in range(N_WF + 1)]

    minimum_damping_ratio_contingencies = np.vstack([results_element[0] for results_element in results_list])

    # Now, only save the data associated with the smallest DR
    smallest_damping_ratio_index = np.argmin(minimum_damping_ratio_contingencies)

    return results_list[smallest_damping_ratio_index]


# %% Directed Walk
def NSWPH_Directed_Walk(OP, Damping, Jacobian, Nm1_models):
    Parameter_Upper_Bound = input_upper_bound.flatten()
    Parameter_Lower_Bound = input_lower_bound.flatten()
    # Directed walk
    #
    # Things learned:
    #   1. Don't step control parameters and power at the same time
    #   2. Step sizes have to be small -- Newton *definitely* doesn't work
    #
    # Define maximum steps
    decay_rate = 0.85
    max_step_far = 0.05  # when outside of 0.5%
    max_step_close = max_step_far / 4  # when within     0.5%
    #####################################################
    #####################################################
    # Original sampling scheme: take a true gradient step
    # with a single scaling factor (mag_val)
    #    max_step_far   = 10/100     # when outside of 0.5%
    #    max_step_close = 2.5/100    # when within     0.5%
    #
    # Now, scale the step so that it touches 10% in some direction
    #    mag_val  = min(mag)
    #    step_val = -mag_val*sub_jac
    # Take a step
    #    OP_DW_step[param_indices] = OP_DW[param_indices] + damp_dist*step_val
    #####################################################
    #####################################################
    #
    # Is the LF damping close enough to the stability margin?
    mag = np.array([0.0, 0.0])
    num_additions = 0
    Data_Power_Droop, Data_Damping, Data_Eigenvalues, Data_Jacobian = np.empty((0, 4)), np.empty((0, 1)), np.empty(
        (0, 2)), np.empty((0, 4))

    if abs(Damping - min_damping) < walk_margin:  # Is the damping mode close enough to the margin?
        for jj in range(2):  # Loop over both sets of parameters
            # Initialize decay factors
            Decay_factors_far = 1
            Decay_factors_close = 1

            # Initialize
            OP_DW = copy.deepcopy(OP)  # Start from the original operating point!
            param_indices = range(2 * jj, 2 + 2 * jj)  # Grab the parameter indices (power, or droop?)
            damping_value = copy.copy(Damping)  # "damping_value" is the new, DW damping (initialized here)
            while_idx = 0  # Don't take more than "max_its" DW steps
            add_point = 0  # Prove that a point must be added
            Jacobian_DW = copy.deepcopy(Jacobian)  # Initialize
            while abs(damping_value - min_damping) > damp_tol:  # Can we stop yet?
                break_flag = 0  # Set to 0 initially
                while_idx += 1  # Increment for each DW
                sub_jac = Jacobian_DW.flatten()[param_indices]
                OP_DW_step = copy.copy(OP_DW)
                if np.linalg.norm(sub_jac, 2) < jac_tol:
                    break_flag = 1
                else:
                    # Define bounds
                    if jj == 0:  # Power
                        if abs(damping_value - min_damping) < 0.5:  # within 1
                            LB = Decay_factors_close * max_step_close * (
                                    Parameter_Lower_Bound[param_indices] - OP_DW[param_indices])
                            UB = Decay_factors_close * max_step_close * (
                                    Parameter_Upper_Bound[param_indices] - OP_DW[param_indices])
                            Decay_factors_close = Decay_factors_close * decay_rate
                        else:
                            LB = Decay_factors_far * max_step_far * (
                                    Parameter_Lower_Bound[param_indices] - OP_DW[param_indices])
                            UB = Decay_factors_far * max_step_far * (
                                    Parameter_Upper_Bound[param_indices] - OP_DW[param_indices])
                            Decay_factors_far = Decay_factors_close * decay_rate
                    else:  # Control
                        if abs(damping_value - min_damping) < 0.5:  # within 1
                            LB = Decay_factors_close * max_step_close * (
                                    Parameter_Lower_Bound[param_indices] - OP_DW[param_indices])
                            UB = Decay_factors_close * max_step_close * (
                                    Parameter_Upper_Bound[param_indices] - OP_DW[param_indices])
                            Decay_factors_close = Decay_factors_close * decay_rate
                        else:
                            LB = Decay_factors_far * max_step_far * (
                                    Parameter_Lower_Bound[param_indices] - OP_DW[param_indices])
                            UB = Decay_factors_far * max_step_far * (
                                    Parameter_Upper_Bound[param_indices] - OP_DW[param_indices])
                            Decay_factors_far = Decay_factors_close * decay_rate

                    # Damping distance -- if positive, damping should decrease
                    damp_dist = np.sign(damping_value - min_damping)

                    # Innocent step
                    raw_step = -damp_dist * sub_jac
                    for kk in range(2):
                        if raw_step[kk] > 0:
                            mag[kk] = UB[kk] / raw_step[kk]
                        else:
                            mag[kk] = LB[kk] / raw_step[kk]

                    # Step!
                    OP_DW_step[param_indices] = OP_DW[param_indices] - damp_dist * mag * sub_jac

                    # Clip to max/min values
                    OP_DW_clip = NSWPH_Clip_Step(OP_DW_step, Parameter_Upper_Bound, Parameter_Lower_Bound)

                    # If BOTH elements have been clipped, STOP.
                    if (while_idx > max_its) or (np.count_nonzero(OP_DW_clip - OP_DW_step) == 2) or np.all(
                            OP_DW == OP_DW_clip):
                        break_flag = 1
                    else:
                        # Re-set
                        OP_DW = OP_DW_clip

                        # Take a step and save the data
                        Damping_DW, Eigenvalues_DW, Jacobian_DW = NSWPH_Minimum_Data_Point(OP_DW, Nm1_models)
                        print(Damping_DW)

                        # Since we have taken at least one step, set the "add_point" flag
                        add_point = 1

                        # Define the new damping value
                        damping_value = Damping_DW

                if break_flag == 1:
                    break

            # Add only the last collected point
            if add_point == 1:
                # num_additions += 1
                print("--------")
                Data_Power_Droop, Data_Damping, Data_Eigenvalues, Data_Jacobian = OP_DW.reshape(
                    (1, -1)), Damping_DW.reshape((1, -1)), Eigenvalues_DW.reshape((1, -1)), Jacobian_DW.reshape((1, -1))

    return Data_Power_Droop, Data_Damping, Data_Eigenvalues, Data_Jacobian


# %% Clip gradient steps
def NSWPH_Clip_Step(Parameters, Parameter_Upper_Bound, Parameter_Lower_Bound):
    Parameters_Clipped = np.maximum(Parameters, Parameter_Lower_Bound)
    Parameters_Clipped = np.minimum(Parameters_Clipped, Parameter_Upper_Bound)

    return Parameters_Clipped


def NSWPH_Directed_Walk_Data(Data_Power_Droop, Data_Damping, Data_Jacobian, Nm1_models):
    results = [NSWPH_Directed_Walk(Operating_Point, Damping, Jacobian, Nm1_models)
               for Operating_Point, Damping, Jacobian
               in zip(Data_Power_Droop, Data_Damping, Data_Jacobian)]

    input_OPs = np.vstack([output_tuple[0] for output_tuple in results])
    output_damping_ratios = np.vstack([output_tuple[1] for output_tuple in results])
    output_eigenvalues = np.vstack([output_tuple[2] for output_tuple in results])
    output_jacobians = np.vstack([output_tuple[3] for output_tuple in results])

    return input_OPs, output_damping_ratios, output_eigenvalues, output_jacobians
