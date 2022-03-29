# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 12:21:57 2021

@author: brysa
"""
import numpy as np

import dataset_creation.NSWPH_load_network_data as lnd
import dataset_creation.NSWPH_load_parameters as lpr
from dataset_creation.NSWPH_models import network_parameters


def nswph_models(n_sc, n_off, n_wf, n_wf_out, Pwf=0.0, Pl1=0.0, onshore=False, nswph=None, test=1):
    """
    Constructs the North Sea Wind Power Hub system

    Parameters
    ----------
    n_sc : int
        DESCRIPTION.
    n_off : int
        DESCRIPTION.
    n_wf : int
        DESCRIPTION.
    n_wf_out : int
        DESCRIPTION. which turbine is out?
    Pwf : float, optional
        Initial power output of the wind farms in pu. The default is 0.
    Pl1 : float, optional
        Initial power output of link 1 in pu. The default is 0.
    onshore : bool, optional
        DESCRIPTION. The default is False.
    nswph : object, optional
        Modified network parameters. The default is None.
    test : int, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    npr : TYPE
        DESCRIPTION.
    models : TYPE
        DESCRIPTION.
    M : TYPE
        DESCRIPTION.

    """

    if nswph is None:
        nswph = lpr.network_parameters_nswph()
    npr = network_parameters()
    models = []

    nb = 1
    nbr = 0
    nb_dc = 0
    nbr_dc = 0

    f = []
    t = []
    Rbr = []
    Lbr = []
    Cbr = []
    Csh = [nswph.Chub]

    M = {}
    k = 0
    for i in range(n_sc):
        m = lpr.syn_con()

        m.bus_ind = nb
        m.ID = k

        models.append(m)

        f.append(0)
        t.append(nb)
        Rbr.append(nswph.Rtsc)
        Lbr.append(nswph.Ltsc)
        Cbr.append(0)

        Csh.append(nswph.Csc)
        nb += 1
        nbr += 1

        M['SC{:d}'.format(i + 1)] = k
        k += 1

    Psum = 0

    if n_wf_out == 0:
        # No turbine is out
        n_wf_outage = n_wf
        length_vec = np.linspace(1, n_wf, n_wf) - 1

    else:
        # A turbine is out
        n_wf_outage = n_wf - 1

        # New, construct an array used for defining line lengths
        length_vec = np.linspace(1, n_wf, n_wf)
        length_vec = np.delete(length_vec, n_wf_out - 1) - 1

    for i in range(n_wf_outage):

        f.append(0)
        t.append(nb)
        Rbr.append(nswph.Rtwf)
        Lbr.append(nswph.Ltwf)
        Cbr.append(0)
        Csh.append(0)

        for b in range(test):
            Csh.append(0)
            f.append(nb + b)
            t.append(nb + b + 1)
            l = (nswph.Cblen + length_vec[i] * nswph.Cbinc) / test
            Rbr.append(nswph.Rcb * l)
            Lbr.append(nswph.Lcb * l)
            Cbr.append(nswph.Ccb * l)

        nb += test
        nbr += 1 + test
        # m = lpr.wind_farm_converter_parameters()
        m = lpr.wind_farm_converter()
        m.Pref = Pwf
        m.bus_ind = nb
        m.ID = k

        models.append(m)

        nb += 1

        M['WF{:d}'.format(i + 1)] = k
        k += 1
        Psum += Pwf * m.Sn

    for i in range(n_off):
        m = lpr.offshore_converter()

        if i == 0:  # TODO this is not finished. Set power transfer through the hub. Initialization doesn't apply this yet
            Pl = Pl1
        else:
            Pl = -Pl1

        m.Pref = -(Psum / m.Sn) / n_off + Pl
        m.bus_ind = 0
        m.ID = k

        models.append(m)

        M['OFF{:d}'.format(i + 1)] = k
        k += 1

        ###
        if onshore:
            m2 = lpr.onshore_converter()

            m2.bus_ind = nb
            m2.Pref = (Psum / m.Sn) / n_off
            m2.ID = k

            models.append(m2)

            M['ON{:d}'.format(i + 1)] = k

            Csh.append(0.01)

            k += 1

            ###
            mcb = lpr.dc_cable()
            mcb.f = k - 2
            mcb.t = k - 1
            mcb.ID = k

            models.append(mcb)

            M['DCcb{:d}'.format(i + 1)] = k

            k += 1

            mvs = lpr.voltage_source()
            mvs.bus_ind = nb
            models.append(mvs)
            mvs.ID = k

            M['VS{:d}'.format(i + 1)] = k
            k += 1

            nb += 1

            m.bus_ind_dc = nb_dc
            m2.bus_ind_dc = nb_dc + 1
            mcb.bus_f_dc = nb_dc
            mcb.bus_t_dc = nb_dc + 1
            nb_dc += 2
            nbr_dc += 1

    npr.n_bus = nb
    npr.n_br = nbr

    npr.n_bus_dc = nb_dc
    npr.n_br_dc = nbr_dc

    npr.f = np.array(f)
    npr.t = np.array(t)
    npr.Rbr = np.array(Rbr)
    npr.Lbr = np.array(Lbr)
    npr.Cbr = np.array(Cbr)
    npr.Csh = np.array(Csh)

    lnd.LoadNetworkData(npr, models)

    return npr, models, M
