import numpy as np

from dataset_creation.NSWPH_models import ModelType


def LoadNetworkData(npr, models):
    """
    Updates the Network parameters with the Ybus matrix and bus types according
    to the system models.

    Parameters
    ----------
    npr : object
        Network parameters.
    models : list
        List of system models.

    Returns
    -------
    None.

    """
    ######################

    # TODO this part should be more generic

    bus_inds = {'Nr': 0, 'Type': 1, 'Pld': 2, 'Qld': 3, 'Gsh': 4, 'Bsh': 5, 'Vm': 6, 'Va': 7}

    bus = np.empty(shape=(0, 8))

    for i in range(npr.n_bus):
        b = np.array([[i, 1, 0, 0, 0, 0, 1, 0]])
        bus = np.vstack((bus, b))
    # for i in range(npr.n_bus_dc):
    #     b = np.array([[npr.n_bus+i, 5, 0, 0, 0, 0, 1, 0]])
    #     bus = np.vstack((bus,b))

    branch = np.empty(shape=(0, 5))
    n = 0
    for model in models:
        if model.type == ModelType.GEN_ORD_6 or model.type == ModelType.GEN_2_2:
            if not hasattr(model, 'Vref') or model.Vref == 0:
                Vref = 1
            else:
                Vref = model.Vref
            if n == 0:
                b = np.array([[model.bus_ind, 3, 0, 0, 0, 0, Vref, 0]])
                n += 1
            else:
                b = np.array([[model.bus_ind, 2, model.Pm, 0, 0, 0, Vref, 0]])

            bus[model.bus_ind] = b

        elif model.type == ModelType.VSC_1:

            if model.Kq > 0 and model.Kv > 0:
                bus[model.bus_ind, bus_inds['Type']] = 4
            elif model.Kv > 0:
                bus[model.bus_ind, bus_inds['Type']] = 2
            else:
                bus[model.bus_ind, bus_inds['Type']] = 1

            # if not model.bus_ind_dc == -1:
            #     bus[model.bus_ind_dc,bus_inds['Type']] = 5

            bus[model.bus_ind, bus_inds['Pld']] += model.Pref * model.Sn / npr.Sb
            bus[model.bus_ind, bus_inds['Qld']] += model.Qref * model.Sn / npr.Sb

        elif model.type == ModelType.VS:
            bus[model.bus_ind, bus_inds['Type']] = 3
            bus[model.bus_ind, bus_inds['Vm']] = model.V0
            n += 1

    bus[:npr.n_bus, bus_inds['Bsh']] += npr.Csh

    br_inds = {'From': 0, 'To': 1, 'R': 2, 'X': 3, 'B': 4}

    branch = np.c_[npr.f, npr.t, npr.Rbr, npr.Lbr, npr.Cbr]

    ##########################

    n_bus = len(bus[:, 0])
    n_br = len(branch[:, 0])

    Ys = 1 / (branch[:, br_inds['R']] + 1j * branch[:, br_inds['X']])
    Yp = 1j * branch[:, br_inds['B']]

    Yff = Ys + Yp / 2
    Yft = - Ys
    Ytf = - Ys

    Ysh = (bus[:, bus_inds['Gsh']] + 1j * bus[:, bus_inds['Bsh']])

    f = branch[:, br_inds['From']].astype(int)
    t = branch[:, br_inds['To']].astype(int)

    Ybus = np.zeros((n_bus, n_bus), dtype=complex)
    for i in range(n_br):
        Ybus[f[i], f[i]] += Yff[i]
        Ybus[f[i], t[i]] += Yft[i]
        Ybus[t[i], f[i]] += Ytf[i]
        Ybus[t[i], t[i]] += Yff[i]

    for i in range(n_bus):
        Ybus[i, i] += Ysh[i]

    npr.Sbus = bus[:, bus_inds['Pld']] + 1j * bus[:, bus_inds['Qld']]

    npr.V0 = bus[:, bus_inds['Vm']] * np.exp(1j * np.deg2rad(bus[:, bus_inds['Va']]))

    buscode = bus[:, bus_inds['Type']]

    npr.pq_index = np.where(buscode == 1)[0]  # Find indices for all PQ-busses
    npr.pv_index = np.where(buscode == 2)[0]  # Find indices for all PV-busses
    npr.pqv_index = np.where(buscode == 4)[0]  # Find indices for all Converter-busses with voltage droop
    npr.ref = np.where(buscode == 3)[0]  # Find index for ref bus

    # Create Branch Matrices
    # Create the two branch admittance matrices
    Y_from = np.zeros((n_br, n_bus), dtype=np.complex)
    Y_to = np.zeros((n_br, n_bus), dtype=np.complex)
    # br_f = f# The from busses (python indices start at 0)
    # br_t = t# The to busses
    # br_Y = Ys# The series admittance of each branch
    for k in range(0, len(f)):  # Fill in the matrices
        Y_from[k, f[k]] = Ys[k]
        Y_from[k, t[k]] = -Ys[k]
        Y_to[k, f[k]] = -Ys[k]
        Y_to[k, t[k]] = Ys[k]

    npr.Ybus = Ybus
    npr.Y_from = Y_from
    npr.Y_to = Y_to

    return
