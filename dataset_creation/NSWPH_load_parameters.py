import numpy as np

import dataset_creation.NSWPH_models as mdls

S_BASE = 100
SC_MVA = 300
OFFSHORE_MVA = 2100
WF_MVA = 800
FREQ = 50
WN = 2 * np.pi * FREQ


def wind_farm_converter():
    wf = mdls.vsc_1()

    wf.Sn = WF_MVA
    wf.Sb = S_BASE
    wf.wn = WN

    wf.Rt = 0.01
    wf.Lt = 0.15
    wf.Kp_pll = 0.83
    wf.Ki_pll = 31.84
    wf.Kpf = 0
    wf.Kif = 0
    wf.Kpp = 0
    wf.Kip = 15.7
    wf.Kpc = 0.477
    wf.Kic = 0.6
    wf.Kq = 1
    wf.Kv = 0
    wf.Kpq = 0.1
    wf.Kiq = 40.9
    wf.Kad = 1
    wf.Tad = 0.2
    wf.Pref = 0.0
    wf.Qref = 0.0
    wf.Vref = 1.0
    wf.Tpm = 0.001
    wf.Tvm = 0.001

    wf.ctrl = mdls.CtrlMode.P_Q
    wf.label = 'wind_farm'

    return wf


def offshore_converter():
    off = mdls.vsc_1()

    off.Sn = OFFSHORE_MVA
    off.Sb = S_BASE
    off.wn = WN

    off.Rt = 0.01
    off.Lt = 0.15
    off.Kp_pll = 0.64
    off.Ki_pll = 31.84
    off.Kpf = 2.5
    off.Kif = 3.55
    off.Kpp = 0.0
    off.Kip = 35.4
    off.Kic = 0.8
    off.Kpc = 0.977
    off.Kq = 0.0
    off.Kv = 1
    off.Kpq = 7.59
    off.Kiq = 10
    off.Kad = 1
    off.Tad = 0.2
    off.Pref = 0.0
    off.Qref = 0.0
    off.Vref = 1.0
    off.Tpm = 0.001
    off.Tvm = 0.001

    off.ctrl = mdls.CtrlMode.P_VAC

    off.label = 'offshore'

    return off


def onshore_converter():
    on = mdls.vsc_1()

    on.Sn = OFFSHORE_MVA
    on.Sb = S_BASE
    on.wn = WN

    on.Rt = 0.01
    on.Lt = 0.15
    on.Kp_pll = 0.83
    on.Ki_pll = 31.84
    on.Kpf = 0
    on.Kif = 0
    on.Kpp = 1.3
    on.Kip = 70
    on.Kic = 30.
    on.Kpc = 1.43
    on.Kq = 1
    on.Kv = 0
    on.Kpq = 0.1
    on.Kiq = 10
    on.Kad = 0
    on.Tad = 0.05

    on.Pref = 0.0
    on.Qref = 0.0
    on.Vref = 1.0

    on.Tpm = 0.001
    on.Tvm = 0.001

    on.ctrl = mdls.CtrlMode.VDC_Q

    on.label = 'onshore'

    return on


def syn_con():
    sc = mdls.gen_2_2()

    sc.Sn = SC_MVA
    sc.Sb = S_BASE
    sc.ID = -1
    sc.Pm = 0
    sc.Qm = 0
    sc.cosn = 1
    sc.freq = FREQ

    # input parameters
    sc.ra = 0.05

    sc.xd = 2.2
    sc.xdp = 0.3
    sc.xdpp = 0.2

    sc.xq = 2
    sc.xqp = 0.4
    sc.xqpp = 0.2

    sc.xrld = 0
    sc.xrlq = 0
    sc.dkd = 0
    sc.dpe = 0

    sc.xl = 0.15

    sc.H = 2  # rated to gen_MVA
    sc.Td0p = 7
    sc.Td0pp = 0.05
    sc.Tq0p = 1.5
    sc.Tq0pp = 0.05

    sc.calc_machine_parameters()

    # # AVR TODO move
    sc.Pe0 = 0
    sc.Efd0 = 1

    sc.Vref = 1.0
    sc.Kc = 16
    sc.Tc = 2.67
    sc.Te = 0.3
    sc.Tm = 0.005

    sc.label = 'synchronous_condenser'

    return sc


def dc_cable():
    cable = mdls.dc_line()
    cable.wn = WN
    cable.ID = -1
    cable.Length = 200
    Vb = 1050
    Sb = 100
    Zb = Vb ** 2 / Sb
    r = 0.011
    l = 1e-5
    c = (2 * 54.78516e-6 * WN) * 1

    cable.R = r * cable.Length / Zb
    cable.L = l * cable.Length / Zb
    cable.C = c * Zb
    cable.G = 0.0 * cable.Length

    cable.label = 'dc_cable'
    cable.bus_f_dc = -1
    cable.bus_t_dc = -1
    cable.Sf = 100
    cable.St = 100
    cable.x_If = -1
    cable.x_It = -1

    return cable


def voltage_source():
    vs = mdls.voltage_source()
    vs.wn = WN
    vs.ID = -1
    vs.R = 0.0001
    vs.L = 0.002
    vs.X = vs.L
    vs.V0 = 1

    vs.label = 'voltage_source'

    return vs


class network_parameters_nswph():
    Sb = S_BASE
    wn = WN
    Chub = 0.01 * 1
    Csc = 0.01 * 1

    n_sc = 1
    n_wf = 1
    n_off = 1

    vb = 66  # kV
    zb = vb ** 2 / Sb
    Rcb = 0.006416666667  # Ohm/km
    Lcb = 0.009162975  # Ohm/km
    Xcb = 0.001020223994  # MOhm*km (shunt cap. reactance)

    Cblen = 10 * 1  # Cable length (km) of first Wind farm
    Cbinc = 5 * 1  # Length increment for each subsequent Wind farm
    Ccb = zb / (Xcb * 1e6)
    Rcb = Rcb / zb
    Lcb = Lcb / zb

    Rtsc = 0.01 / (SC_MVA / S_BASE)
    Ltsc = 0.15 / (SC_MVA / S_BASE)

    Rtwf = 0.01 / (WF_MVA / S_BASE)
    Ltwf = 0.15 / (WF_MVA / S_BASE)
