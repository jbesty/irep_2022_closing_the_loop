# from numba import njit
from enum import IntEnum

import numpy as np
# import numba
# from numba.experimental import jitclass
from numba import int32, int64, float64, complex128, typed
from numba.core import types

kv_ty = (types.unicode_type, types.int64)


class ModelType(IntEnum):
    """
    Identification of different Model types.
    """
    GEN_ORD_6 = 0  # 6th order model
    VSC_1 = 1
    DC_LINE = 2
    VS = 3
    SAVR = 4
    GEN_2_2 = 5  # model 2.2


class CtrlMode(IntEnum):
    """
    Identification of converter control modes.
    """
    P_VAC = 0
    P_Q = 1
    VDC_Q = 2


# @njit
def d_vsc_dt(xm, um, model):
    """
    Voltage Source Converter differential equations

    Parameters
    ----------
    xm : ndarray
        State vector.
    um : ndarray
        Input vector.
    model : object
        Model parameters.

    Returns
    -------
    dx : ndarray
        State derivatives.

    """

    i_d = xm[model.x_idx['Id']]
    i_q = xm[model.x_idx['Iq']]
    # i_dc = x[model.x_idx['Idc']]
    Md = xm[model.x_idx['Md']]
    Mq = xm[model.x_idx['Mq']]
    Madd = xm[model.x_idx['Madd']]
    Madq = xm[model.x_idx['Madq']]
    Theta_pll = xm[model.x_idx['Theta']]
    Xpll = xm[model.x_idx['Xpll']]
    Xf = xm[model.x_idx['Xf']]
    Xp = xm[model.x_idx['Xp']]
    Xq = xm[model.x_idx['Xq']]
    Pm = xm[model.x_idx['Pm']]
    Qm = xm[model.x_idx['Qm']]
    Vm = xm[model.x_idx['Vm']]

    vx = um[0]
    vy = um[1]
    Vdc = um[2]
    Pref = um[3]
    Qref = um[4]
    Vref = um[5]

    vd = (vx * np.cos(Theta_pll) + vy * np.sin(Theta_pll))
    vq = (-vx * np.sin(Theta_pll) + vy * np.cos(Theta_pll))

    wpll = model.Kp_pll * vq + model.Ki_pll * Xpll

    # wpll = np.clip(wpll, 0.8, 1.2) # TODO check the limits and make them part of the model

    Pac = vd * i_d + vq * i_q
    Qac = (vq * i_d - vd * i_q)
    Vac = np.sqrt(vd ** 2 + vq ** 2)

    if model.Tpm == 0:
        Pm = Pac
        Qm = Qac
    if model.Tvm == 0:
        Vm = Vac

    if model.ctrl == CtrlMode.VDC_Q:  # TODO seperate the control modes to avoid mixup (Vref is used for both ac and dc)
        dP = Vdc / Vref - 1
    else:
        dP = Pref - Pm + model.Kpf * (1 - wpll) + model.Kif * Xf

    id_ref = model.Kpp * dP + Xp * model.Kip

    dQ = (model.Kq * (Qm - Qref) + model.Kv * (Vm - Vref))
    iq_ref = dQ * model.Kpq + Xq * model.Kiq

    # id_max = 1
    # id_ref = np.clip(id_ref, -id_max, id_max)

    # iq_max = np.sqrt(max(0,1-id_ref**2))
    # iq_ref = np.clip(iq_ref, -iq_max, iq_max)

    vmd = (Madd - wpll * model.Lt * i_q + model.Kpc * (id_ref - i_d) + model.Kic * Md) / Vdc
    vmq = (Madq + wpll * model.Lt * i_d + model.Kpc * (iq_ref - i_q) + model.Kic * Mq) / Vdc

    dx = np.zeros(len(xm))
    dx[model.x_idx['Id']] = model.wn / model.Lt * (vmd - vd - model.Rt * i_d + wpll * model.Lt * i_q)  # di_d
    dx[model.x_idx['Iq']] = model.wn / model.Lt * (vmq - vq - model.Rt * i_q - wpll * model.Lt * i_d)  # di_q
    # dx[model.x_idx['Idc']]= (model.wn/(model.Ldc)*(Pac/Vdc-i_dc)) # TODO find a propper equation assuming power
    #  balance between AC and DC sides
    dx[model.x_idx['Md']] = (id_ref - i_d)  # dMd
    dx[model.x_idx['Mq']] = (iq_ref - i_q)  # dMq
    dx[model.x_idx['Madd']] = (-Madd + vd) / model.Tad  # dMadd
    dx[model.x_idx['Madq']] = (-Madq + vq) / model.Tad  # dMadq
    dx[model.x_idx['Theta']] = (wpll - 1) * model.wn  # dTheta_pll
    dx[model.x_idx['Xpll']] = vq  # dXpll
    dx[model.x_idx['Xf']] = (1 - wpll)  # dMf
    dx[model.x_idx['Xp']] = dP  # dMp
    dx[model.x_idx['Xq']] = dQ  # dMq

    if model.Tpm > 0:
        dx[model.x_idx['Pm']] = (Pac - Pm) / model.Tpm
        dx[model.x_idx['Qm']] = (Qac - Qm) / model.Tpm
    if model.Tvm > 0:
        dx[model.x_idx['Vm']] = (Vac - Vm) / model.Tvm

    return dx


# @njit
def d_dcline_dt(xm, um, model):
    """
    DC line differential equations

    Parameters
    ----------
    xm : ndarray
        State vector.
    u : ndarray
        Input vector.
    model : object
        Model parameters.

    Returns
    -------
    dx : ndarray
        State derivatives.
    """

    Il = xm[model.x_idx['Il']]
    Vf = xm[model.x_idx['Vf']]
    Vt = xm[model.x_idx['Vt']]

    If = um[0]
    It = um[1]

    dx = np.zeros(len(xm))
    dx[model.x_idx['Il']] = model.wn * 1 / (model.L + 1e-6) * (Vf - Vt - model.R * Il)
    dx[model.x_idx['Vf']] = model.wn * 2 / model.C * (If - Il - model.G / 2 * Vf)
    dx[model.x_idx['Vt']] = model.wn * 2 / model.C * (Il - It - model.G / 2 * Vt)

    return dx


# @njit
def d_vs_dt(xm, um, model):
    """
    Voltage source differential equations

    Parameters
    ----------
    xm : ndarray
        State vector.
    um : ndarray
        Input vector.
    model : object
        Model parameters.

    Returns
    -------
    dx : ndarray
        State derivatives.
    """
    phi = xm[model.x_idx['phi']]
    Ix = xm[model.x_idx['Ix']]
    Iy = xm[model.x_idx['Iy']]

    Vx = um[0]
    Vy = um[1]
    # f = um[2]
    fpu = 1  # TODO this should be the measured grid frequency
    dphi = 2 * np.pi * 50 * (fpu - 1)

    ux_setp = model.V0 * np.cos(phi + dphi)
    uy_setp = model.V0 * np.sin(phi + dphi)

    dIx = model.wn / model.L * (ux_setp - Vx - model.R * Ix + model.L * Iy)
    dIy = model.wn / model.L * (uy_setp - Vy - model.R * Iy - model.L * Ix)

    dx = np.zeros(len(xm))
    dx[model.x_idx['phi']] = dphi
    dx[model.x_idx['Ix']] = dIx
    dx[model.x_idx['Iy']] = dIy

    return dx


# @njit
def d_gen_ord_6_rms_dt(xm, um, model):
    """
    Sixth order generator differential equations.
    Generator current is not a state variable and
    is calculated from the terminal and subtransient
    voltage.

    Parameters
    ----------
    xm : ndarray
        State vector.
    um : ndarray
        Input vector.
    model : object
        Model parameters.

    Returns
    -------
    dx : ndarray
        State derivatives.
    """
    d = xm[model.x_idx['d']]
    w = xm[model.x_idx['w']]
    Eqp = xm[model.x_idx['Eqp']]
    Eqpp = xm[model.x_idx['Eqpp']]
    Edp = xm[model.x_idx['Edp']]
    Edpp = xm[model.x_idx['Edpp']]

    Efq = xm[model.x_idx['Efq']]  # TODO seperate the avr from the generator to simplify using different avr models
    Vf = xm[model.x_idx['Vf']]
    X_avr = xm[model.x_idx['Xavr']]

    Efq = max(Efq, 0)  # TODO add limits

    vx = um[0]
    vy = um[1]
    Vref = um[2]

    Vac = np.sqrt(vx ** 2 + vy ** 2)

    Vd = (vx * np.cos(d) + vy * np.sin(d))
    Vq = (-vx * np.sin(d) + vy * np.cos(d))

    Id = -(-model.ra * (Vd - Edpp) - model.xqpp * (Vq - Eqpp)) / (model.ra ** 2 + model.xqpp * model.xdpp)
    Iq = -(model.xdpp * (Vd - Edpp) - model.ra * (Vq - Eqpp)) / (model.ra ** 2 + model.xqpp * model.xdpp)

    Pe = -(Vd * Id + Vq * Iq) + (Id ** 2 + Iq ** 2) * model.ra
    # Pe = (Edpp*Id+Eqpp*Iq)+(model.xdpp-model.xqpp)*Id*Iq

    delta_w = model.wn * (w - 1)
    dx = np.zeros(len(xm))
    dx[model.x_idx['d']] = delta_w
    dx[model.x_idx['w']] = 1 / (model.Tj) * (model.Pm - Pe - model.D * w)  # dw
    dx[model.x_idx['Eqp']] = 1 / model.Tdp * (Efq - Eqp + Id * (model.xd - model.xdp))
    dx[model.x_idx['Eqpp']] = 1 / model.Tdpp * (Eqp - Eqpp + Id * (model.xdp - model.xdpp))
    dx[model.x_idx['Edp']] = 1 / model.Tqp * (-Edp - Iq * (model.xq - model.xqp))
    dx[model.x_idx['Edpp']] = 1 / model.Tqpp * (Edp - Edpp - Iq * (model.xqp - model.xqpp))

    dEfq = 1 / model.Te * (-Efq + model.Kc * (Vref - Vf) + model.Kc / model.Tc * X_avr)

    dx[model.x_idx['Efq']] = dEfq
    dx[model.x_idx['Vf']] = 1 / model.Tm * (-Vf + Vac)
    dx[model.x_idx['Xavr']] = (Vref - Vf)

    return dx


# @njit
def d_gen_ord_6_emt_dt(xm, um, model):
    """
    Sixth order generator differential equations.
    Generator current is included as a state variable.

    Parameters
    ----------
    xm : ndarray
        State vector.
    um : ndarray
        Input vector.
    model : object
        Model parameters.

    Returns
    -------
    dx : ndarray
        State derivatives.
    """
    Id = xm[model.x_idx['Id']]
    Iq = xm[model.x_idx['Iq']]

    d = xm[model.x_idx['d']]
    w = xm[model.x_idx['w']]
    Eqp = xm[model.x_idx['Eqp']]
    Eqpp = xm[model.x_idx['Eqpp']]
    Edp = xm[model.x_idx['Edp']]
    Edpp = xm[model.x_idx['Edpp']]

    Efq = xm[model.x_idx['Efq']]  # TODO seperate the avr from the generator to simplify using different avr models
    Vf = xm[model.x_idx['Vf']]
    X_avr = xm[model.x_idx['Xavr']]

    # Efq = np.clip(Efq, 0.0, 5.0)

    vx = um[0]
    vy = um[1]
    Vref = um[2]

    Vac = np.sqrt(vx ** 2 + vy ** 2)

    Vd = vx * np.cos(d) + vy * np.sin(d)
    Vq = -vx * np.sin(d) + vy * np.cos(d)

    Pe = (Edpp * Id + Eqpp * Iq) + (model.xdpp - model.xqpp) * Id * Iq

    delta_w = model.wn * (w - 1)
    dx = np.zeros(len(xm))
    dx[model.x_idx['d']] = delta_w
    dx[model.x_idx['w']] = (1 / model.Tj) * (model.Pm - Pe - model.D * w)
    dx[model.x_idx['Eqp']] = (1 / model.Tdp) * (Efq - Eqp - Id * (model.xd - model.xdp))
    dx[model.x_idx['Eqpp']] = (1 / model.Tdpp) * (Eqp - Eqpp - Id * (model.xdp - model.xdpp))
    dx[model.x_idx['Edp']] = (1 / model.Tqp) * (-Edp + Iq * (model.xq - model.xqp))
    dx[model.x_idx['Edpp']] = (1 / model.Tqpp) * (Edp - Edpp + Iq * (model.xqp - model.xqpp))

    dEfq = 1 / model.Te * (-Efq + model.Kc * (Vref - Vf) + model.Kc / model.Tc * X_avr)

    dx[model.x_idx['Efq']] = dEfq
    dx[model.x_idx['Vf']] = 1 / model.Tm * (-Vf + Vac)
    dx[model.x_idx['Xavr']] = (Vref - Vf)

    # TODO check the equations for w*E''
    dx[model.x_idx['Id']] = model.wn / model.xdpp * (w * Edpp - Vd - model.ra * Id + w * model.xqpp * Iq)
    dx[model.x_idx['Iq']] = model.wn / model.xqpp * (w * Eqpp - Vq - model.ra * Iq - w * model.xdpp * Id)

    return dx


# @njit
def d_gen_model_2_2_dt(xm, um, model):
    """
    Generator model 2.2 differential equations.
    Generator current is included as a state variable.

    Parameters
    ----------
    xm : ndarray
        State vector.
    um : ndarray
        Input vector.
    model : object
        Model parameters.

    Returns
    -------
    dx : ndarray
        State derivatives.
    """

    Id = xm[model.x_idx['Id']]
    Iq = xm[model.x_idx['Iq']]

    d = xm[model.x_idx['d']]
    w = xm[model.x_idx['w']]
    psi_d = xm[model.x_idx['psi_d']]
    psi_q = xm[model.x_idx['psi_q']]
    psi_fd = xm[model.x_idx['psi_fd']]
    psi_1d = xm[model.x_idx['psi_1d']]
    psi_1q = xm[model.x_idx['psi_1q']]
    psi_2q = xm[model.x_idx['psi_2q']]

    Efd = xm[model.x_idx['Efd']]  # TODO seperate the avr from the generator to simplify using different avr models
    Vf = xm[model.x_idx['Vf']]
    X_avr = xm[model.x_idx['Xavr']]

    # Efd = np.clip(Efd, 0.0, 5.0)

    vx = um[0]
    vy = um[1]
    Vref = um[2]
    # Efd = um[3]

    Vac = np.sqrt(vx ** 2 + vy ** 2)

    Vd = (vx * np.cos(d) + vy * np.sin(d))
    Vq = (-vx * np.sin(d) + vy * np.cos(d))

    vfd = model.rfd / model.xadu * Efd

    te = (Iq * psi_d - Id * psi_q) / model.cosn
    tm = 0  # TODO include torque input
    tdkd = model.dkd * (w - 1)
    tdpe = model.dpe / w * (w - 1)

    ifd = model.kfd * Id + (model.x1d_loop * psi_fd - (model.xad + model.xrld) * psi_1d) / model.xdet_d
    i1d = model.k1d * Id + (model.xfd_loop * psi_1d - (model.xad + model.xrld) * psi_fd) / model.xdet_d
    i1q = model.k1q * Iq + (model.x2q_loop * psi_1q - (model.xaq + model.xrlq) * psi_2q) / model.xdet_q
    i2q = model.k2q * Iq + (model.x1q_loop * psi_2q - (model.xaq + model.xrlq) * psi_1q) / model.xdet_q

    dpsi_fd = model.wn * (vfd - model.rfd * ifd)
    dpsi_1d = model.wn * (-model.r1d * i1d)
    dpsi_1q = model.wn * (-model.r1q * i1q)
    dpsi_2q = model.wn * (-model.r2q * i2q)

    Edpp = -w * (model.k1q * psi_1q + model.k2q * psi_2q) + (
            model.kfd / model.wn * dpsi_fd + model.k1d / model.wn * dpsi_1d)
    Eqpp = w * (model.kfd * psi_fd + model.k1d * psi_1d) + (
            model.k1q / model.wn * dpsi_1q + model.k2q / model.wn * dpsi_2q)

    delta_w = model.wn * (w - 1)
    dx = np.zeros(len(xm))
    dx[model.x_idx['d']] = delta_w
    dx[model.x_idx['w']] = (1 / model.Tj) * (tm - te - tdkd - tdpe)
    dx[model.x_idx['psi_d']] = model.wn * (Vd + model.ra * Id + w * psi_q)
    dx[model.x_idx['psi_q']] = model.wn * (Vq + model.ra * Iq - w * psi_d)
    dx[model.x_idx['psi_fd']] = dpsi_fd
    dx[model.x_idx['psi_1d']] = dpsi_1d
    dx[model.x_idx['psi_1q']] = dpsi_1q
    dx[model.x_idx['psi_2q']] = dpsi_2q

    dx[model.x_idx['Id']] = (model.wn / model.xdpp * (Edpp - Vd - model.ra * Id + w * model.xqpp * Iq))
    dx[model.x_idx['Iq']] = (model.wn / model.xqpp * (Eqpp - Vq - model.ra * Iq - w * model.xdpp * Id))

    dEfd = 1 / model.Te * (-Efd + model.Kc * (Vref - Vf) + model.Kc / model.Tc * X_avr)

    dx[model.x_idx['Efd']] = dEfd
    dx[model.x_idx['Vf']] = 1 / model.Tm * (-Vf + Vac)
    dx[model.x_idx['Xavr']] = (Vref - Vf)

    return dx


# @njit
def d_avr_dt(xm, um, model):
    """
    Simple AVR differential equations.

    Parameters
    ----------
    xm : ndarray
        State vector.
    um : ndarray
        Input vector.
    model : object
        Model parameters.

    Returns
    -------
    dx : ndarray
        State derivatives.
    """

    Efd = xm[model.x_idx['Efd']]
    Vf = xm[model.x_idx['Vf']]
    X_avr = xm[model.x_idx['Xavr']]

    Vpu = um[0]
    Vref = um[2]

    dx = np.zeros(len(xm))
    dx[model.x_idx['Efd']] = 1 / model.Te * (-Efd + model.Kc * (Vref - Vf) + model.Kc / model.Tc * X_avr)
    dx[model.x_idx['Vf']] = 1 / model.Tm * (-Vf + Vpu)
    dx[model.x_idx['Xavr']] = (Vref - Vf)

    return dx


# @njit
def d_network_dt(xn, un, npr):
    """
    Differential equations of the network. Assuming non-zero capacitance
    at every bus. Zero capacitance buses are not yet fully tested.

    Parameters
    ----------
    xn : ndarray
        State vector.
    un : ndarray
        Input vector.
    npr : object
        Network parameters.

    Returns
    -------
    dx : ndarray
        State derivatives.
    """

    n_bus = npr.n_bus

    ishx = np.zeros(n_bus)
    ishy = np.zeros(n_bus)

    for k in range(n_bus):
        ishx[k] += un[k * 2]
        ishy[k] += un[k * 2 + 1]

    for k, (f, t) in enumerate(zip(npr.f, npr.t)):
        ishx[f] -= xn[n_bus * 2 + k * 2]
        ishy[f] -= xn[n_bus * 2 + k * 2 + 1]

        ishx[t] += xn[n_bus * 2 + k * 2]
        ishy[t] += xn[n_bus * 2 + k * 2 + 1]

    dx = np.zeros(len(xn))

    for i in range(n_bus):
        Csh = np.imag(npr.Ybus[i].sum())
        vx = xn[i * 2]
        vy = xn[i * 2 + 1]
        if Csh > 1e-9:
            dx[i * 2] = npr.wn / Csh * (ishx[i] + Csh * vy)
            dx[i * 2 + 1] = npr.wn / Csh * (ishy[i] - Csh * vx)
        else:  # TODO if Csh == 0
            dx[i * 2] = npr.wn * ishx[i]
            dx[i * 2 + 1] = npr.wn * ishy[i]

    for i in range(npr.n_br):
        f = npr.f[i]
        t = npr.t[i]

        vfx = xn[f * 2]
        vfy = xn[f * 2 + 1]
        vtx = xn[t * 2]
        vty = xn[t * 2 + 1]

        ix = xn[n_bus * 2 + i * 2]
        iy = xn[n_bus * 2 + i * 2 + 1]

        R = np.real(-1 / npr.Ybus[f, t])
        L = np.imag(-1 / npr.Ybus[f, t])

        dx[n_bus * 2 + i * 2] = (npr.wn / L * (vfx - vtx - R * ix + L * iy))
        dx[n_bus * 2 + i * 2 + 1] = (npr.wn / L * (vfy - vty - R * iy - L * ix))

    return dx


# # @njit
def d_sys_dt(x, u, npr, models):
    # TODO # @njit can't be used with the models in a list.
    # I should find a way around that because it's much slower
    dx = np.zeros(len(x))

    unw = np.zeros(npr.n_bus * 2)

    for model in models:
        vx = x[npr.x_ind + model.bus_ind * 2]
        vy = x[npr.x_ind + model.bus_ind * 2 + 1]

        if model.type == ModelType.GEN_ORD_6:
            xm = x[model.x_ind:model.x_ind + model.nx]
            um = np.array([vx, vy, model.Vref])
            dx[model.x_ind:model.x_ind + model.nx] = d_gen_ord_6_emt_dt(xm, um, model)

            d = x[model.x_ind + model.x_idx['d']]

            Id = x[model.x_ind + model.x_idx['Id']]
            Iq = x[model.x_ind + model.x_idx['Iq']]

            ix = (Id * np.cos(d) - Iq * np.sin(d)) * model.Sn / npr.Sb
            iy = (Id * np.sin(d) + Iq * np.cos(d)) * model.Sn / npr.Sb

            unw[model.bus_ind * 2] += ix
            unw[model.bus_ind * 2 + 1] += iy

        if model.type == ModelType.GEN_2_2:
            xm = x[model.x_ind:model.x_ind + model.nx]
            um = np.array([vx, vy, model.Vref])
            dx[model.x_ind:model.x_ind + model.nx] = d_gen_model_2_2_dt(xm, um, model)

            d = x[model.x_ind + model.x_idx['d']]
            # Vd = vx*np.cos(d)+vy*np.sin(d)
            # Vq = -vx*np.sin(d)+vy*np.cos(d)

            # Eqpp = x[model.x_ind+model.x_idx['Eqpp']]
            # Edpp = x[model.x_ind+model.x_idx['Edpp']]

            # IdIq = model.Zg_inv.copy()@np.array([[Vd-Edpp],[Vq-Eqpp]])

            # Id = IdIq[0,0]
            # Iq = IdIq[1,0]

            Id = x[model.x_ind + model.x_idx['Id']]
            Iq = x[model.x_ind + model.x_idx['Iq']]

            ix = (Id * np.cos(d) - Iq * np.sin(d)) * model.Sn / npr.Sb
            iy = (Id * np.sin(d) + Iq * np.cos(d)) * model.Sn / npr.Sb

            # unw = np.hstack((unw,ix,iy))
            unw[model.bus_ind * 2] += ix
            unw[model.bus_ind * 2 + 1] += iy
        elif model.type == ModelType.VSC_1:
            ix = (x[model.x_ind + model.x_idx['Id']] * np.cos(x[model.x_ind + model.x_idx['Theta']]) - x[
                model.x_ind + model.x_idx['Iq']] * np.sin(x[model.x_ind + model.x_idx['Theta']])) * model.Sn / npr.Sb
            iy = (x[model.x_ind + model.x_idx['Id']] * np.sin(x[model.x_ind + model.x_idx['Theta']]) + x[
                model.x_ind + model.x_idx['Iq']] * np.cos(x[model.x_ind + model.x_idx['Theta']])) * model.Sn / npr.Sb

            xm = x[model.x_ind:model.x_ind + model.nx]
            um = np.array([vx, vy, 1, model.Pref, model.Qref, model.Vref])

            dx[model.x_ind:model.x_ind + model.nx] = d_vsc_dt(xm, um, model)

            # unw = np.hstack((unw,ix,iy))
            unw[model.bus_ind * 2] += ix
            unw[model.bus_ind * 2 + 1] += iy

    xnw = x[npr.x_ind:]

    dx[npr.x_ind:] = d_network_dt(xnw, unw, npr)

    return dx


# @njit
def calc_gen_dx(x, npr, model):
    vx = x[npr.x_ind + model.bus_ind * 2]
    vy = x[npr.x_ind + model.bus_ind * 2 + 1]

    xm = x[model.x_ind:model.x_ind + model.nx]
    um = np.array([vx, vy, model.Vref])

    dx = model.dx_dt(xm, um)

    d = x[model.x_ind + model.x_idx['d']]

    Id = x[model.x_ind + model.x_idx['Id']]
    Iq = x[model.x_ind + model.x_idx['Iq']]

    ix = (Id * np.cos(d) - Iq * np.sin(d)) * model.Sn / npr.Sb
    iy = (Id * np.sin(d) + Iq * np.cos(d)) * model.Sn / npr.Sb

    return dx, ix, iy


# @njit
def calc_vsc_dx(x, npr, model):
    if not model.x_dc == -1:
        vdc = x[model.x_dc]
    else:
        vdc = 1
    vx = x[npr.x_ind + model.bus_ind * 2]
    vy = x[npr.x_ind + model.bus_ind * 2 + 1]

    ix = (x[model.x_ind + model.x_idx['Id']] * np.cos(x[model.x_ind + model.x_idx['Theta']]) - x[
        model.x_ind + model.x_idx['Iq']] * np.sin(x[model.x_ind + model.x_idx['Theta']])) * model.Sn / npr.Sb
    iy = (x[model.x_ind + model.x_idx['Id']] * np.sin(x[model.x_ind + model.x_idx['Theta']]) + x[
        model.x_ind + model.x_idx['Iq']] * np.cos(x[model.x_ind + model.x_idx['Theta']])) * model.Sn / npr.Sb

    xm = x[model.x_ind:model.x_ind + model.nx]
    um = np.array([vx, vy, vdc, model.Pref, model.Qref, model.Vref])

    dx = model.dx_dt(xm, um)

    return dx, ix, iy


# @njit
def calc_dc_cable_dx(x, npr, model, vsc_f, vsc_t):
    If = -x[vsc_f.x_ind + vsc_f.x_idx['Idc']] * vsc_f.Sn / npr.Sb
    It = x[vsc_t.x_ind + vsc_t.x_idx['Idc']] * vsc_t.Sn / npr.Sb

    xm = x[model.x_ind:model.x_ind + model.nx]
    um = np.array([If, It])

    dx = model.dx_dt(xm, um)

    return dx


# @njit
def calc_dc_cable2_dx(x, npr, model):
    If = -x[model.x_If] * model.Sf / npr.Sb
    It = x[model.x_It] * model.St / npr.Sb

    xm = x[model.x_ind:model.x_ind + model.nx]
    um = np.array([If, It])

    dx = model.dx_dt(xm, um)

    return dx


# @njit
def calc_vs_dx(x, npr, model):
    vx = x[npr.x_ind + model.bus_ind * 2]
    vy = x[npr.x_ind + model.bus_ind * 2 + 1]

    xm = x[model.x_ind:model.x_ind + model.nx]

    um = np.array([vx, vy])

    dx = model.dx_dt(xm, um)

    ix = x[model.x_ind + model.x_idx['Ix']]
    iy = x[model.x_ind + model.x_idx['Iy']]

    return dx, ix, iy


def d_sys_2_dt(x, npr, models):
    # TODO @njit can't be used with the models in a list.
    # I should find a way around that because it's much slower
    dx = np.zeros(len(x))

    unw = np.zeros(npr.n_bus * 2)

    for model in models:
        if model.type in [ModelType.GEN_ORD_6, ModelType.GEN_2_2]:
            dx[model.x_ind:model.x_ind + model.nx], ix, iy = calc_gen_dx(x, npr, model)

            unw[model.bus_ind * 2] += ix
            unw[model.bus_ind * 2 + 1] += iy

        elif model.type == ModelType.VSC_1:
            dx[model.x_ind:model.x_ind + model.nx], ix, iy = calc_vsc_dx(x, npr, model)

            unw[model.bus_ind * 2] += ix
            unw[model.bus_ind * 2 + 1] += iy

        elif model.type == ModelType.DC_LINE:
            dx[model.x_ind:model.x_ind + model.nx] = calc_dc_cable_dx(x, npr, model, models[model.f], models[model.t])

        elif model.type == ModelType.VS:
            dx[model.x_ind:model.x_ind + model.nx], ix, iy = calc_vs_dx(x, npr, model)

            unw[model.bus_ind * 2] += ix
            unw[model.bus_ind * 2 + 1] += iy

    xnw = x[npr.x_ind:]

    dx[npr.x_ind:] = d_network_dt(xnw, unw, npr)

    return dx


# @njit
def d_sys_nswph_dt(x, u, npr, OFF, WF, SC):
    dx = np.zeros(len(x))

    unw = np.zeros(npr.n_bus * 2)

    for model in OFF.models:
        vx = x[npr.x_ind + model.bus_ind * 2]
        vy = x[npr.x_ind + model.bus_ind * 2 + 1]

        if not model.x_dc == -1:
            vdc = x[model.x_dc]
        else:
            vdc = 1

        ix = (x[model.x_ind + model.x_idx['Id']] * np.cos(x[model.x_ind + model.x_idx['Theta']]) - x[
            model.x_ind + model.x_idx['Iq']] * np.sin(x[model.x_ind + model.x_idx['Theta']])) * model.Sn / npr.Sb
        iy = (x[model.x_ind + model.x_idx['Id']] * np.sin(x[model.x_ind + model.x_idx['Theta']]) + x[
            model.x_ind + model.x_idx['Iq']] * np.cos(x[model.x_ind + model.x_idx['Theta']])) * model.Sn / npr.Sb

        xm = x[model.x_ind:model.x_ind + model.nx]
        um = np.array([vx, vy, vdc, model.Pref, model.Qref, model.Vref])

        dx[model.x_ind:model.x_ind + model.nx] = model.dx_dt(xm, um)

        unw[model.bus_ind * 2] += ix
        unw[model.bus_ind * 2 + 1] += iy

    for model in WF.models:
        vx = x[npr.x_ind + model.bus_ind * 2]
        vy = x[npr.x_ind + model.bus_ind * 2 + 1]

        ix = (x[model.x_ind + model.x_idx['Id']] * np.cos(x[model.x_ind + model.x_idx['Theta']]) - x[
            model.x_ind + model.x_idx['Iq']] * np.sin(x[model.x_ind + model.x_idx['Theta']])) * model.Sn / npr.Sb
        iy = (x[model.x_ind + model.x_idx['Id']] * np.sin(x[model.x_ind + model.x_idx['Theta']]) + x[
            model.x_ind + model.x_idx['Iq']] * np.cos(x[model.x_ind + model.x_idx['Theta']])) * model.Sn / npr.Sb

        xm = x[model.x_ind:model.x_ind + model.nx]
        um = np.array([vx, vy, 1, u[0], u[1], u[2]])

        dx[model.x_ind:model.x_ind + model.nx] = model.dx_dt(xm, um)

        unw[model.bus_ind * 2] += ix
        unw[model.bus_ind * 2 + 1] += iy

    for model in SC.models:
        vx = x[npr.x_ind + model.bus_ind * 2]
        vy = x[npr.x_ind + model.bus_ind * 2 + 1]

        xm = x[model.x_ind:model.x_ind + model.nx]
        um = np.array([vx, vy, model.Vref])

        d = x[model.x_ind + model.x_idx['d']]

        dx[model.x_ind:model.x_ind + model.nx] = model.dx_dt(xm, um)

        Id = x[model.x_ind + model.x_idx['Id']]
        Iq = x[model.x_ind + model.x_idx['Iq']]

        ix = (Id * np.cos(d) - Iq * np.sin(d)) * model.Sn / npr.Sb
        iy = (Id * np.sin(d) + Iq * np.cos(d)) * model.Sn / npr.Sb

        unw[model.bus_ind * 2] += ix
        unw[model.bus_ind * 2 + 1] += iy

    xnw = x[npr.x_ind:]

    dx[npr.x_ind:] = d_network_dt(xnw, unw, npr)

    return dx


# @njit
def d_sys_nswph_full_dt(x, u, npr, OFF, WF, SC, ON, CB, VS):
    dx = np.zeros(len(x))

    unw = np.zeros(npr.n_bus * 2)
    P_model = np.zeros(VS.models[-1].ID)
    for model in OFF.models:
        vx = x[npr.x_ind + model.bus_ind * 2]
        vy = x[npr.x_ind + model.bus_ind * 2 + 1]

        if not model.x_dc == -1:
            vdc = x[model.x_dc]
            # vdc = 1
        else:
            vdc = 1

        ix = (x[model.x_ind + model.x_idx['Id']] * np.cos(x[model.x_ind + model.x_idx['Theta']]) - x[
            model.x_ind + model.x_idx['Iq']] * np.sin(x[model.x_ind + model.x_idx['Theta']])) * model.Sn / npr.Sb
        iy = (x[model.x_ind + model.x_idx['Id']] * np.sin(x[model.x_ind + model.x_idx['Theta']]) + x[
            model.x_ind + model.x_idx['Iq']] * np.cos(x[model.x_ind + model.x_idx['Theta']])) * model.Sn / npr.Sb

        xm = x[model.x_ind:model.x_ind + model.nx]
        um = np.array([vx, vy, vdc, model.Pref, model.Qref, model.Vref])

        dx[model.x_ind:model.x_ind + model.nx] = model.dx_dt(xm, um)

        unw[model.bus_ind * 2] += ix
        unw[model.bus_ind * 2 + 1] += iy

        P_model[model.ID] = vx * ix + vy * iy

    for model in WF.models:
        vx = x[npr.x_ind + model.bus_ind * 2]
        vy = x[npr.x_ind + model.bus_ind * 2 + 1]

        ix = (x[model.x_ind + model.x_idx['Id']] * np.cos(x[model.x_ind + model.x_idx['Theta']]) - x[
            model.x_ind + model.x_idx['Iq']] * np.sin(x[model.x_ind + model.x_idx['Theta']])) * model.Sn / npr.Sb
        iy = (x[model.x_ind + model.x_idx['Id']] * np.sin(x[model.x_ind + model.x_idx['Theta']]) + x[
            model.x_ind + model.x_idx['Iq']] * np.cos(x[model.x_ind + model.x_idx['Theta']])) * model.Sn / npr.Sb

        xm = x[model.x_ind:model.x_ind + model.nx]
        um = np.array([vx, vy, 1, model.Pref, model.Qref, model.Vref])

        dx[model.x_ind:model.x_ind + model.nx] = model.dx_dt(xm, um)

        unw[model.bus_ind * 2] += ix
        unw[model.bus_ind * 2 + 1] += iy

    for model in SC.models:
        vx = x[npr.x_ind + model.bus_ind * 2]
        vy = x[npr.x_ind + model.bus_ind * 2 + 1]

        xm = x[model.x_ind:model.x_ind + model.nx]
        um = np.array([vx, vy, model.Vref])
        dx[model.x_ind:model.x_ind + model.nx] = model.dx_dt(xm, um)

        d = x[model.x_ind + model.x_idx['d']]  # +np.pi/2
        # Vd = vx*np.cos(d)+vy*np.sin(d)
        # Vq = -vx*np.sin(d)+vy*np.cos(d)
        # Vd = vx*np.sin(d)-vy*np.cos(d)
        # Vq = vx*np.cos(d)+vy*np.sin(d)

        # Eqpp = x[model.x_ind+model.x_idx['Eqpp']]
        # Edpp = x[model.x_ind+model.x_idx['Edpp']]

        # IdIq = model.Zg_inv.copy()@np.array([[Vd-Edpp],[Vq-Eqpp]])

        # Id = IdIq[0,0]
        # Iq = IdIq[1,0]

        # Id = (-model.ra*(Vd-Edpp)-model.xqpp*(Vq-Eqpp))/(model.ra**2+model.xqpp*model.xdpp)
        # Iq = (model.xdpp*(Vd-Edpp)-model.ra*(Vq-Eqpp))/(model.ra**2+model.xqpp*model.xdpp)

        # Id = (model.ra*(Edpp-Vd)-model.xqpp*(Eqpp-Vq))/(model.ra**2+model.xqpp*model.xdpp)
        # Iq = (model.xdpp*(Edpp-Vd)+model.ra*(Eqpp-Vq))/(model.ra**2+model.xqpp*model.xdpp)
        Id = x[model.x_ind + model.x_idx['Id']]
        Iq = x[model.x_ind + model.x_idx['Iq']]

        ix = (Id * np.cos(d) - Iq * np.sin(d)) * model.Sn / npr.Sb
        iy = (Id * np.sin(d) + Iq * np.cos(d)) * model.Sn / npr.Sb
        # ix = (Id*np.sin(d)+Iq*np.cos(d))*model.Sn/npr.Sb
        # iy = (-Id*np.cos(d)+Iq*np.sin(d))*model.Sn/npr.Sb

        unw[model.bus_ind * 2] += ix
        unw[model.bus_ind * 2 + 1] += iy

    for model in ON.models:
        vx = x[npr.x_ind + model.bus_ind * 2]
        vy = x[npr.x_ind + model.bus_ind * 2 + 1]

        if not model.x_dc == -1:
            vdc = x[model.x_dc]
        else:
            vdc = 1

        ix = (x[model.x_ind + model.x_idx['Id']] * np.cos(x[model.x_ind + model.x_idx['Theta']]) - x[
            model.x_ind + model.x_idx['Iq']] * np.sin(x[model.x_ind + model.x_idx['Theta']])) * model.Sn / npr.Sb
        iy = (x[model.x_ind + model.x_idx['Id']] * np.sin(x[model.x_ind + model.x_idx['Theta']]) + x[
            model.x_ind + model.x_idx['Iq']] * np.cos(x[model.x_ind + model.x_idx['Theta']])) * model.Sn / npr.Sb

        xm = x[model.x_ind:model.x_ind + model.nx]
        um = np.array([vx, vy, vdc, model.Pref, model.Qref, model.Vref])

        dx[model.x_ind:model.x_ind + model.nx] = model.dx_dt(xm, um)

        unw[model.bus_ind * 2] += ix
        unw[model.bus_ind * 2 + 1] += iy

        P_model[model.ID] = vx * ix + vy * iy

    for model in CB.models:
        # def test1(model,OFF):
        for m1 in OFF.models:
            if m1.ID == model.f:
                If = -x[m1.x_ind + m1.x_idx['Idc']] * m1.Sn / npr.Sb
                break
        for m2 in ON.models:
            if m2.ID == model.t:
                It = x[m2.x_ind + m2.x_idx['Idc']] * m2.Sn / npr.Sb
                break
        #   return If,It
        # def test2(model,OFF):
        #     m1 = OFF.models[np.flatnonzero(np.asarray([m.ID for m in OFF.models])==model.f)[0]]
        #     If = -x[m1.x_ind+m1.x_idx['Idc']]*m1.Sn/npr.Sb
        #     m2 = ON.models[np.flatnonzero(np.asarray([m.ID for m in ON.models])==model.t)[0]]
        #     It = x[m2.x_ind+m2.x_idx['Idc']]*m2.Sn/npr.Sb
        #     return If,It

        Pf = -P_model[model.f]

        Pt = P_model[model.t]

        Vfdc = x[model.x_ind + model.x_idx['Vf']]
        Vtdc = x[model.x_ind + model.x_idx['Vt']]

        # If = Pf/Vfdc
        # It = Pt/Vtdc

        # If = x[model.x_ind+model.x_idx['Idc']]
        xm = x[model.x_ind:model.x_ind + model.nx]
        um = np.array([If, It])

        dx[model.x_ind:model.x_ind + model.nx] = model.dx_dt(xm, um)

    for model in VS.models:
        vx = x[npr.x_ind + model.bus_ind * 2]
        vy = x[npr.x_ind + model.bus_ind * 2 + 1]

        xm = x[model.x_ind:model.x_ind + model.nx]
        um = np.array([vx, vy])

        dx[model.x_ind:model.x_ind + model.nx] = model.dx_dt(xm, um)

        unw[model.bus_ind * 2] += x[model.x_ind + model.x_idx['Ix']]
        unw[model.bus_ind * 2 + 1] += x[model.x_ind + model.x_idx['Iy']]

    xnw = x[npr.x_ind:]

    dx[npr.x_ind:] = d_network_dt(xnw, unw, npr)

    return dx


# %% Component models

spec_vsc_1 = [
    ('Sn', float64),
    ('Sb', float64),
    ('wn', float64),
    ('Rt', float64),
    ('Lt', float64),
    ('Ldc', float64),
    ('Kp_pll', float64),
    ('Ki_pll', float64),
    ('Kpf', float64),
    ('Kif', float64),
    ('Kpp', float64),
    ('Kip', float64),
    ('Kic', float64),
    ('Kpc', float64),
    ('Kq', float64),
    ('Kv', float64),
    ('Kpq', float64),
    ('Kiq', float64),
    ('Kad', float64),
    ('Tad', float64),
    ('Pref', float64),
    ('Qref', float64),
    ('Vref', float64),
    ('Tpm', float64),
    ('Tvm', float64),
    ('x_idx', types.DictType(*kv_ty)),
    ('nx', types.int64),
    ('states', types.ListType(types.unicode_type)),
    ('bus_ind', int64),
    ('bus_ind_dc', int64),
    ('x_ind', int64),
    ('type', int64),
    ('label', types.string),
    ('ctrl', int64),
    ('ID', int64),
    ('x_dc', int64),
]


# @jitclass(spec_vsc_1)
class vsc_1():
    def __init__(self):

        self.Sn = 100
        self.Sb = 100
        self.wn = 2 * np.pi * 50
        self.ID = -1

        self.Rt = 0.01
        self.Lt = 0.15
        self.Ldc = 0.01
        self.Kp_pll = 0.83
        self.Ki_pll = 31.84
        self.Kpf = 0
        self.Kif = 0
        self.Kpp = 0.0
        self.Kip = 39
        self.Kic = 0.6
        self.Kpc = 0.477
        self.Kq = 0.0
        self.Kv = 1
        self.Kpq = 5
        self.Kiq = 10
        self.Kad = 1
        self.Tad = 0.2
        self.Pref = 0.0
        self.Qref = 0.0
        self.Vref = 1.0
        self.Tpm = 0.001
        self.Tvm = 0.001

        self.x_idx = typed.Dict.empty(*kv_ty)
        self.states = typed.List.empty_list(types.unicode_type)
        self.states.extend(
            ['Id', 'Iq', 'Idc', 'Md', 'Mq', 'Madd', 'Madq', 'Theta', 'Xpll', 'Xf', 'Xp', 'Xq', 'Pm', 'Qm', 'Vm'])
        # self.states.extend(['Id','Iq','Md','Mq','Madd','Madq','Theta','Xpll','Xf','Xp','Xq','Pm','Qm','Vm'])
        self.nx = len(self.states)
        for i, state in enumerate(self.states):
            self.x_idx[state] = i

        self.type = ModelType.VSC_1
        self.label = 'vsc'
        self.ctrl = CtrlMode.P_VAC
        self.x_dc = -1
        self.bus_ind = -1
        self.bus_ind_dc = -1

    def dx_dt(self, xm, um):
        """
        Voltage Source Converter differential equations

        Parameters
        ----------
        xm : ndarray
            State vector.
        um : ndarray
            Input vector.
        model : object
            Model parameters.

        Returns
        -------
        dx : ndarray
            State derivatives.

        """

        i_d = xm[self.x_idx['Id']]
        i_q = xm[self.x_idx['Iq']]
        i_dc = xm[self.x_idx['Idc']]
        Md = xm[self.x_idx['Md']]
        Mq = xm[self.x_idx['Mq']]
        Madd = xm[self.x_idx['Madd']]
        Madq = xm[self.x_idx['Madq']]
        Theta_pll = xm[self.x_idx['Theta']]
        Xpll = xm[self.x_idx['Xpll']]
        Xf = xm[self.x_idx['Xf']]
        Xp = xm[self.x_idx['Xp']]
        Xq = xm[self.x_idx['Xq']]
        Pm = xm[self.x_idx['Pm']]
        Qm = xm[self.x_idx['Qm']]
        Vm = xm[self.x_idx['Vm']]

        vx = um[0]
        vy = um[1]
        Vdc = um[2]
        Pref = um[3]
        Qref = um[4]
        Vref = um[5]

        vd = (vx * np.cos(Theta_pll) + vy * np.sin(Theta_pll))
        vq = (-vx * np.sin(Theta_pll) + vy * np.cos(Theta_pll))

        wpll = self.Kp_pll * vq + self.Ki_pll * Xpll

        wpll = np.clip(np.array([wpll]), 0.8, 1.2)[0]  # TODO check the limits and make them part of the model

        Pac = vd * i_d + vq * i_q
        Qac = (vq * i_d - vd * i_q)
        Vac = np.sqrt(vd ** 2 + vq ** 2)

        if self.Tpm == 0:
            Pm = Pac
            Qm = Qac
        if self.Tvm == 0:
            Vm = Vac

        if self.ctrl == CtrlMode.VDC_Q:  # TODO seperate the control modes to avoid mixup (Vref is used for both ac and dc)
            dP = Vdc / Vref - 1
        else:
            dP = Pref - Pm + self.Kpf * (1 - wpll) + self.Kif * Xf

        id_ref = self.Kpp * dP + Xp * self.Kip

        dQ = (self.Kq * (Qm - Qref) + self.Kv * (Vm - Vref))
        iq_ref = dQ * self.Kpq + Xq * self.Kiq

        id_max = 1
        id_ref = np.clip(np.array([id_ref]), -id_max, id_max)[0]

        iq_max = np.sqrt(max(0, 1 - id_ref ** 2))
        iq_ref = np.clip(np.array([iq_ref]), -iq_max, iq_max)[0]

        vmd = (Madd - wpll * self.Lt * i_q + self.Kpc * (id_ref - i_d) + self.Kic * Md) / Vdc
        vmq = (Madq + wpll * self.Lt * i_d + self.Kpc * (iq_ref - i_q) + self.Kic * Mq) / Vdc

        dx = np.zeros(len(xm))
        dx[self.x_idx['Id']] = self.wn / self.Lt * (vmd - vd - self.Rt * i_d + wpll * self.Lt * i_q)  # di_d
        dx[self.x_idx['Iq']] = self.wn / self.Lt * (vmq - vq - self.Rt * i_q - wpll * self.Lt * i_d)  # di_q
        dx[self.x_idx['Idc']] = (self.wn / (self.Ldc) * ((
                                                                 i_d * vmd + i_q * vmq) / Vdc - i_dc))  # TODO find a propper equation assuming power balance between AC and DC sides
        # dx[self.x_idx['Idc']]= self.wn*((i_d*vmd+i_q*vmq)/Vdc-i_dc)
        dx[self.x_idx['Md']] = (id_ref - i_d)  # dMd
        dx[self.x_idx['Mq']] = (iq_ref - i_q)  # dMq
        dx[self.x_idx['Madd']] = (-Madd + vd) / (self.Tad)  # dMadd
        dx[self.x_idx['Madq']] = (-Madq + vq) / (self.Tad)  # dMadq
        dx[self.x_idx['Theta']] = (wpll - 1) * self.wn  # dTheta_pll
        dx[self.x_idx['Xpll']] = vq  # dXpll
        dx[self.x_idx['Xf']] = (1 - wpll)  # dMf
        dx[self.x_idx['Xp']] = dP  # dMp
        dx[self.x_idx['Xq']] = dQ  # dMq

        if self.Tpm > 0:
            dx[self.x_idx['Pm']] = (Pac - Pm) / self.Tpm
            dx[self.x_idx['Qm']] = (Qac - Qm) / self.Tpm
        if self.Tvm > 0:
            dx[self.x_idx['Vm']] = (Vac - Vm) / self.Tvm

        return dx

    def abcd_linear(self, xo, uo):
        Id = xo[self.x_idx['Id']]
        Iq = xo[self.x_idx['Iq']]
        theta = xo[self.x_idx['Theta']]
        Xpll = xo[self.x_idx['Xpll']]
        vd = uo[0]
        vq = uo[1]
        if len(uo) > 2:
            vdc = uo[2]
        else:
            vdc = 1.0

        Ac = np.array([[self.wn * (-self.Kpc / vdc - self.Rt) / self.Lt, self.wn * (
                self.Lt * (Xpll * self.Ki_pll + self.Kp_pll * vq) - self.Lt * (
                Xpll * self.Ki_pll + self.Kp_pll * vq) / vdc) / self.Lt, 0.0,
                        self.Kic * self.wn / (self.Lt * vdc), 0.0, self.wn / (self.Lt * vdc), 0.0, 0.0, self.wn * (
                                Iq * self.Ki_pll * self.Lt + (
                                -Iq * self.Ki_pll * self.Lt - self.Ki_pll * self.Kpc * self.Kpf * self.Kpp) / vdc) / self.Lt,
                        self.Kif * self.Kpc * self.Kpp * self.wn / (self.Lt * vdc),
                        self.Kip * self.Kpc * self.wn / (self.Lt * vdc), 0.0,
                        -self.Kpc * self.Kpp * self.wn / (self.Lt * vdc), 0.0, 0.0],
                       [self.wn * (-self.Lt * (Xpll * self.Ki_pll + self.Kp_pll * vq) + self.Lt * (
                               Xpll * self.Ki_pll + self.Kp_pll * vq) / vdc) / self.Lt,
                        self.wn * (-self.Kpc / vdc - self.Rt) / self.Lt, 0.0, 0.0, self.Kic * self.wn / (self.Lt * vdc),
                        0.0, self.wn / (self.Lt * vdc), 0.0,
                        self.wn * (-Id * self.Ki_pll * self.Lt + Id * self.Ki_pll * self.Lt / vdc) / self.Lt, 0.0, 0.0,
                        self.Kiq * self.Kpc * self.wn / (self.Lt * vdc), 0.0,
                        self.Kpc * self.Kpq * self.Kq * self.wn / (self.Lt * vdc),
                        self.Kpc * self.Kpq * self.Kv * self.wn / (self.Lt * vdc)],
                       [self.wn * vd / (self.Ldc * vdc), self.wn * vq / (self.Ldc * vdc), -self.wn / self.Ldc, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -self.Ki_pll * self.Kpf * self.Kpp,
                        self.Kif * self.Kpp, self.Kip, 0.0, -self.Kpp, 0.0, 0.0],
                       [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.Kiq, 0.0, self.Kpq * self.Kq,
                        self.Kpq * self.Kv],
                       [0.0, 0.0, 0.0, 0.0, 0.0, -1.0 / self.Tad, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0 / self.Tad, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.Ki_pll * self.wn, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -self.Ki_pll, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -self.Ki_pll * self.Kpf, self.Kif, 0.0, 0.0, -1.0, 0.0,
                        0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.Kq, self.Kv],
                       [vd / self.Tpm, vq / self.Tpm, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0 / self.Tpm,
                        0.0, 0.0],
                       [vq / self.Tpm, -vd / self.Tpm, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        -1.0 / self.Tpm, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0 / self.Tvm]
                       ], dtype=float)

        Bvc = np.array([[-self.wn / self.Lt, self.wn * (Iq * self.Kp_pll * self.Lt + (
                -Iq * self.Kp_pll * self.Lt - self.Kp_pll * self.Kpc * self.Kpf * self.Kpp) / vdc) / self.Lt],
                        [0.0,
                         self.wn * (-Id * self.Kp_pll * self.Lt + Id * self.Kp_pll * self.Lt / vdc - 1.0) / self.Lt],
                        [Id * self.wn / (self.Ldc * vdc), Iq * self.wn / (self.Ldc * vdc)],
                        [0.0, -self.Kp_pll * self.Kpf * self.Kpp],
                        [0.0, 0.0],
                        [1.0 / self.Tad, 0.0],
                        [0.0, 1.0 / self.Tad],
                        [0.0, self.Kp_pll * self.wn],
                        [0.0, 1.0],
                        [0.0, -self.Kp_pll],
                        [0.0, -self.Kp_pll * self.Kpf],
                        [0.0, 0.0],
                        [Id / self.Tpm, Iq / self.Tpm],
                        [-Iq / self.Tpm, Id / self.Tpm],
                        [vd / (self.Tvm * np.sqrt(vd ** 2 + vq ** 2)), vq / (self.Tvm * np.sqrt(vd ** 2 + vq ** 2))],
                        ], dtype=float)

        Tc = np.array([[np.cos(theta), np.sin(theta)],
                       [-np.sin(theta), np.cos(theta)]], dtype=float)

        Rvc = np.zeros((2, self.nx), dtype=float)
        Rvc[0, self.x_idx['Theta']] = vq
        Rvc[1, self.x_idx['Theta']] = -vd

        Pc = np.zeros((2, self.nx), dtype=float)
        Pc[0, self.x_idx['Id']] = 1
        Pc[1, self.x_idx['Iq']] = 1
        Pc[0, self.x_idx['Theta']] = -Iq
        Pc[1, self.x_idx['Theta']] = Id

        Cc = Tc.T @ Pc

        Dc = np.zeros((2, 2), dtype=float)

        A = Ac + Bvc @ Rvc
        B = Bvc @ Tc
        C = Cc * self.Sn / self.Sb
        D = Dc

        return A, B, C, D


spec_dcline = [
    ('wn', float64),
    ('ID', int64),
    ('R', float64),
    ('L', float64),
    ('C', float64),
    ('G', float64),
    ('Length', int64),
    ('f', int64),
    ('t', int64),
    ('Sf', int64),
    ('St', int64),
    ('x_If', int64),
    ('x_It', int64),
    ('bus_f_dc', int64),
    ('bus_t_dc', int64),
    ('x_idx', types.DictType(types.unicode_type, types.int64)),
    ('nx', types.int64),
    ('states', types.ListType(types.unicode_type)),
    ('x_ind', int64),
    ('type', int64),
    ('label', types.string)

]


# @jitclass(spec_dcline)
class dc_line():
    def __init__(self):
        self.wn = 2 * np.pi * 50
        self.ID = -1
        self.Length = 100

        self.R = 0.001
        self.L = 0.01
        self.C = 0.01
        self.G = 0.0

        self.x_idx = typed.Dict.empty(*kv_ty)
        self.states = typed.List.empty_list(types.unicode_type)
        self.states.extend(['Il', 'Vf', 'Vt'])
        self.nx = len(self.states)
        for i, state in enumerate(self.states):
            self.x_idx[state] = i

        self.type = ModelType.DC_LINE
        self.label = 'dc_line'
        self.bus_f_dc = -1
        self.bus_t_dc = -1
        self.Sf = 100
        self.St = 100
        self.x_If = -1
        self.x_It = -1

    def dx_dt(self, xm, um):
        Il = xm[self.x_idx['Il']]
        Vf = xm[self.x_idx['Vf']]
        Vt = xm[self.x_idx['Vt']]

        If = um[0]
        It = um[1]

        dx = np.zeros(len(xm))
        dx[self.x_idx['Il']] = (self.wn * 1 / (self.L + 1e-6) * (Vf - Vt - self.R * Il)) / self.wn
        dx[self.x_idx['Vf']] = self.wn * 2 / self.C * (If - Il - self.G / 2 * Vf)
        dx[self.x_idx['Vt']] = self.wn * 2 / self.C * (Il - It - self.G / 2 * Vt)

        return dx


spec_vs = [
    ('wn', float64),
    ('ID', int64),
    ('R', float64),
    ('L', float64),
    ('X', float64),
    ('V0', float64),
    ('x_idx', types.DictType(types.unicode_type, types.int64)),
    ('nx', types.int64),
    ('states', types.ListType(types.unicode_type)),
    ('bus_ind', int64),
    ('x_ind', int64),
    ('type', int64),
    ('label', types.string)
]


# @jitclass(spec_vs)
class voltage_source():
    def __init__(self):
        self.wn = 2 * np.pi * 50
        self.ID = -1
        self.R = 0.0001
        self.L = 0.001
        self.X = self.L

        self.V0 = 1

        self.x_idx = typed.Dict.empty(*kv_ty)
        self.states = typed.List.empty_list(types.unicode_type)
        self.states.extend(['phi', 'Ix', 'Iy'])
        # self.states.extend(['phi'])
        self.nx = len(self.states)
        for i, state in enumerate(self.states):
            self.x_idx[state] = i

        self.type = ModelType.VS
        self.label = 'voltage_source'

    def dx_dt(self, xm, um):
        phi = xm[self.x_idx['phi']]
        Ix = xm[self.x_idx['Ix']]
        Iy = xm[self.x_idx['Iy']]

        Vx = um[0]
        Vy = um[1]
        # f = um[2]
        fpu = 1  # TODO this should be the measured grid frequency
        dphi = 2 * np.pi * 50 * (fpu - 1)

        ux_setp = self.V0 * np.cos(phi + dphi)
        uy_setp = self.V0 * np.sin(phi + dphi)

        dIx = self.wn / self.L * (ux_setp - Vx - self.R * Ix + self.L * Iy)
        dIy = self.wn / self.L * (uy_setp - Vy - self.R * Iy - self.L * Ix)

        dx = np.zeros(len(xm))
        dx[self.x_idx['phi']] = dphi
        dx[self.x_idx['Ix']] = dIx
        dx[self.x_idx['Iy']] = dIy

        return dx

    def abcd_linear(self, xo, uo):
        phi = xo[self.x_idx['phi']]
        Ix = xo[self.x_idx['Ix']]
        Iy = xo[self.x_idx['Iy']]
        Vx = uo[0]
        Vy = uo[1]
        Ex = self.V0 * np.cos(phi)
        Ey = self.V0 * np.sin(phi)

        A = np.array([[-1.0 + ((Ey - Ex) * (Ix * self.R - Iy * self.R - Iy * self.X + Vx + Ey - Ex) / (
                -Ix * self.R - Ix * self.X - Iy * self.R - Vy + Ey + Ex) ** 2 + (Ey + Ex) / (
                                       -Ix * self.R - Ix * self.X - Iy * self.R - Vy + Ey + Ex)) / (
                               1.0 + (Ix * self.R - Iy * self.R - Iy * self.X + Vx + Ey - Ex) ** 2 / (
                               -Ix * self.R - Ix * self.X - Iy * self.R - Vy + Ey + Ex) ** 2), (
                               self.R / (-Ix * self.R - Ix * self.X - Iy * self.R - Vy + Ey + Ex) + (
                               self.R + self.X) * (Ix * self.R - Iy * self.R - Iy * self.X + Vx + Ey - Ex) / (
                                       -Ix * self.R - Ix * self.X - Iy * self.R - Vy + Ey + Ex) ** 2) / (
                               1.0 + (Ix * self.R - Iy * self.R - Iy * self.X + Vx + Ey - Ex) ** 2 / (
                               -Ix * self.R - Ix * self.X - Iy * self.R - Vy + Ey + Ex) ** 2), (
                               self.R * (Ix * self.R - Iy * self.R - Iy * self.X + Vx + Ey - Ex) / (
                               -Ix * self.R - Ix * self.X - Iy * self.R - Vy + Ey + Ex) ** 2 + (
                                       -self.R - self.X) / (
                                       -Ix * self.R - Ix * self.X - Iy * self.R - Vy + Ey + Ex)) / (
                               1.0 + (Ix * self.R - Iy * self.R - Iy * self.X + Vx + Ey - Ex) ** 2 / (
                               -Ix * self.R - Ix * self.X - Iy * self.R - Vy + Ey + Ex) ** 2)],
                      [-Ey * self.wn / self.X, -self.R * self.wn / self.X, self.wn],
                      [Ex * self.wn / self.X, -self.wn, -self.R * self.wn / self.X],
                      ])

        B = np.array([[1.0 / ((1.0 + (Ix * self.R - Iy * self.R - Iy * self.X + Vx + Ey - Ex) ** 2 / (
                -Ix * self.R - Ix * self.X - Iy * self.R - Vy + Ey + Ex) ** 2) * (
                                      -Ix * self.R - Ix * self.X - Iy * self.R - Vy + Ey + Ex)),
                       (Ix * self.R - Iy * self.R - Iy * self.X + Vx + Ey - Ex) / ((1.0 + (
                               Ix * self.R - Iy * self.R - Iy * self.X + Vx + Ey - Ex) ** 2 / (
                                                                                            -Ix * self.R - Ix * self.X - Iy * self.R - Vy + Ey + Ex) ** 2) * (
                                                                                           -Ix * self.R - Ix * self.X - Iy * self.R - Vy + Ey + Ex) ** 2)],
                      [-self.wn / self.X, 0.0],
                      [0.0, -self.wn / self.X],
                      ])

        C = np.zeros((2, self.nx))
        C[0, self.x_idx['Ix']] = 1.0
        C[1, self.x_idx['Iy']] = 1.0
        D = np.zeros((2, 2))

        return A, B, C, D


spec_netw = [
    ('Sb', float64),
    ('wn', float64),
    ('n_bus', int64),
    ('n_br', int64),
    ('n_bus_dc', int64),
    ('n_br_dc', int64),
    ('f', int32[:]),
    ('t', int32[:]),
    ('x_ind', int64),
    ('Ybus', complex128[:, :]),
    ('Y_from', complex128[:, :]),
    ('Y_to', complex128[:, :]),
    ('Rbr', float64[:]),
    ('Lbr', float64[:]),
    ('Cbr', float64[:]),
    ('Csh', float64[:]),
    ('Sbus', complex128[:]),
    ('V0', complex128[:]),
    ('pq_index', int64[:]),
    ('pv_index', int64[:]),
    ('pqv_index', int64[:]),
    ('ref', int64[:]),
    ('x_ind', int64),
]


# @jitclass(spec_netw)
class network_parameters():
    def __init__(self):
        self.Sb = 100
        self.wn = 2 * np.pi * 50
        self.x_ind = -1


spec_gen_ord_6 = [
    ('Sn', float64),
    ('Sb', float64),
    ('wn', float64),
    ('Pm', float64),
    ('Qm', float64),
    ('ra', float64),
    ('xd', float64),
    ('xq', float64),
    ('xdp', float64),
    ('xqp', float64),
    ('xdpp', float64),
    ('xqpp', float64),
    ('xl', float64),
    ('H', float64),
    ('D', float64),
    ('Tdp', float64),
    ('Tdpp', float64),
    ('Tqp', float64),
    ('Tqpp', float64),
    ('Tj', float64),
    ('kd', float64),
    ('kq', float64),
    ('Zg', float64[:, :]),
    ('Zg_inv', float64[:, :]),
    ('Vref', float64),
    ('Kc', float64),
    ('Tc', float64),
    ('Te', float64),
    ('Tm', float64),
    ('Pe0', float64),
    ('Efq0', float64),
    ('x_idx', types.DictType(types.unicode_type, types.int64)),
    ('nx', types.int64),
    ('states', types.ListType(types.unicode_type)),
    ('bus_ind', int64),
    ('x_ind', int64),
    ('type', int64),
    ('label', types.string),
    ('ID', int64),
]


# @jitclass(spec_gen_ord_6)
class gen_ord_6_rms():
    def __init__(self):
        self.Sn = 100
        self.Sb = 100
        self.wn = 2 * np.pi * 50
        self.ID = -1
        self.Pm = 0
        self.Qm = 0

        self.ra = 0.05

        self.xd = 2.2
        self.xdp = 0.3
        self.xdpp = 0.2

        self.xq = 2
        self.xqp = 0.4
        self.xqpp = 0.2

        self.xl = 0.15

        self.H = 2  # rated to gen_MVA
        self.D = 0
        self.Tdp = 7
        self.Tdpp = 0.05
        self.Tqp = 1.5
        self.Tqpp = 0.05
        self.Tj = 2 * self.H

        self.kd = (self.xd - self.xdpp) / (self.xdp - self.xdpp)
        self.kq = (self.xq - self.xqpp) / (self.xqp - self.xqpp)

        self.Zg = np.array([[-self.ra, self.xqpp],
                            [-self.xdpp, -self.ra]])

        self.Zg_inv = np.linalg.inv(self.Zg)

        self.Pe0 = 0
        self.Efq0 = 1

        self.Vref = 1.0
        self.Kc = 16
        self.Tc = 2.67
        self.Te = 0.3
        self.Tm = 0.005

        self.x_idx = typed.Dict.empty(*kv_ty)
        self.states = typed.List.empty_list(types.unicode_type)
        self.states.extend(['d', 'w', 'Eqp', 'Eqpp', 'Edp', 'Edpp', 'Efq', 'Vf', 'Xavr'])
        self.nx = len(self.states)
        for i, state in enumerate(self.states):
            self.x_idx[state] = i

        self.type = ModelType.GEN_ORD_6
        self.label = 'Generator'

    def dx_dt(self, xm, um):
        """
        Sixth order generator differential equations.
        Generator current is not a state variable and
        is calculated from the terminal and subtransient
        voltage.

        Parameters
        ----------
        xm : ndarray
            State vector.
        um : ndarray
            Input vector.

        Returns
        -------
        dx : ndarray
            State derivatives.
        """
        d = xm[self.x_idx['d']]
        w = xm[self.x_idx['w']]
        Eqp = xm[self.x_idx['Eqp']]
        Eqpp = xm[self.x_idx['Eqpp']]
        Edp = xm[self.x_idx['Edp']]
        Edpp = xm[self.x_idx['Edpp']]

        Efq = xm[self.x_idx['Efq']]  # TODO seperate the avr from the generator to simplify using different avr models
        Vf = xm[self.x_idx['Vf']]
        X_avr = xm[self.x_idx['Xavr']]

        Efq = np.clip(np.array([Efq]), 0.0, 5.0)[0]

        vx = um[0]
        vy = um[1]
        Vref = um[2]

        Vac = np.sqrt(vx ** 2 + vy ** 2)

        Vd = (vx * np.cos(d) + vy * np.sin(d))
        Vq = (-vx * np.sin(d) + vy * np.cos(d))

        Id = -(-self.ra * (Vd - Edpp) - self.xqpp * (Vq - Eqpp)) / (self.ra ** 2 + self.xqpp * self.xdpp)
        Iq = -(self.xdpp * (Vd - Edpp) - self.ra * (Vq - Eqpp)) / (self.ra ** 2 + self.xqpp * self.xdpp)

        Pe = -(Vd * Id + Vq * Iq) + (Id ** 2 + Iq ** 2) * self.ra
        # Pe = (Edpp*Id+Eqpp*Iq)+(self.xdpp-self.xqpp)*Id*Iq

        delta_w = self.wn * (w - 1)
        dx = np.zeros(len(xm))
        dx[self.x_idx['d']] = delta_w
        dx[self.x_idx['w']] = 1 / (self.Tj) * (self.Pm - Pe - self.D * w)  # dw
        dx[self.x_idx['Eqp']] = 1 / self.Tdp * (Efq - Eqp + Id * (self.xd - self.xdp))
        dx[self.x_idx['Eqpp']] = 1 / self.Tdpp * (Eqp - Eqpp + Id * (self.xdp - self.xdpp))
        dx[self.x_idx['Edp']] = 1 / self.Tqp * (-Edp - Iq * (self.xq - self.xqp))
        dx[self.x_idx['Edpp']] = 1 / self.Tqpp * (Edp - Edpp - Iq * (self.xqp - self.xqpp))

        dEfq = 1 / self.Te * (-Efq + self.Kc * (Vref - Vf) + self.Kc / self.Tc * X_avr)

        dx[self.x_idx['Efq']] = dEfq
        dx[self.x_idx['Vf']] = 1 / self.Tm * (-Vf + Vac)
        dx[self.x_idx['Xavr']] = (Vref - Vf)

        return dx


# @jitclass(spec_gen_ord_6)
class gen_ord_6_emt():
    def __init__(self):
        self.Sn = 100
        self.Sb = 100
        self.wn = 2 * np.pi * 50
        self.ID = -1
        self.Pm = 0
        self.Qm = 0

        self.ra = 0.05 * 1

        self.xd = 2.2
        self.xdp = 0.3
        self.xdpp = 0.2

        self.xq = 2
        self.xqp = 0.4
        self.xqpp = 0.2

        self.xl = 0.15

        self.H = 2  # rated to gen_MVA
        self.D = 0
        self.Tdp = 7
        self.Tdpp = 0.05
        self.Tqp = 1.5
        self.Tqpp = 0.05

        self.Tj = 2 * self.H

        self.kd = (self.xd - self.xdpp) / (self.xdp - self.xdpp)
        self.kq = (self.xq - self.xqpp) / (self.xqp - self.xqpp)

        self.Zg = np.array([[-self.ra, self.xqpp],
                            [-self.xdpp, -self.ra]])

        self.Zg_inv = np.linalg.inv(self.Zg)

        self.Pe0 = 0
        self.Efq0 = 1

        self.Vref = 1.0
        self.Kc = 16
        self.Tc = 2.67
        self.Te = 0.3
        self.Tm = 0.005

        self.x_idx = typed.Dict.empty(*kv_ty)
        self.states = typed.List.empty_list(types.unicode_type)
        self.states.extend(['d', 'w', 'Eqp', 'Eqpp', 'Edp', 'Edpp', 'Efq', 'Vf', 'Xavr', 'Id', 'Iq'])
        self.nx = len(self.states)
        for i, state in enumerate(self.states):
            self.x_idx[state] = i

        self.type = ModelType.GEN_ORD_6
        self.label = 'synchronous_condenser'

    def dx_dt(self, xm, um):
        """
        Sixth order generator differential equations.
        Generator current is included as a state variable.

        Parameters
        ----------
        xm : ndarray
            State vector.
        um : ndarray
            Input vector.

        Returns
        -------
        dx : ndarray
            State derivatives.
        """
        Id = xm[self.x_idx['Id']]
        Iq = xm[self.x_idx['Iq']]

        d = xm[self.x_idx['d']]
        w = xm[self.x_idx['w']]
        Eqp = xm[self.x_idx['Eqp']]
        Eqpp = xm[self.x_idx['Eqpp']]
        Edp = xm[self.x_idx['Edp']]
        Edpp = xm[self.x_idx['Edpp']]

        Efq = xm[self.x_idx['Efq']]  # TODO seperate the avr from the generator to simplify using different avr models
        Vf = xm[self.x_idx['Vf']]
        X_avr = xm[self.x_idx['Xavr']]

        Efq = np.clip(np.array([Efq]), 0.0, 5.0)[0]

        vx = um[0]
        vy = um[1]
        Vref = um[2]

        Vac = np.sqrt(vx ** 2 + vy ** 2)

        Vd = vx * np.cos(d) + vy * np.sin(d)
        Vq = -vx * np.sin(d) + vy * np.cos(d)

        Pe = (Edpp * Id + Eqpp * Iq) + (self.xdpp - self.xqpp) * Id * Iq

        delta_w = self.wn * (w - 1)
        dx = np.zeros(len(xm))
        dx[self.x_idx['d']] = delta_w
        dx[self.x_idx['w']] = (1 / self.Tj) * (self.Pm - Pe - self.D * w)
        dx[self.x_idx['Eqp']] = (1 / self.Tdp) * (Efq - Eqp - Id * (self.xd - self.xdp))
        dx[self.x_idx['Eqpp']] = (1 / self.Tdpp) * (Eqp - Eqpp - Id * (self.xdp - self.xdpp))
        dx[self.x_idx['Edp']] = (1 / self.Tqp) * (-Edp + Iq * (self.xq - self.xqp))
        dx[self.x_idx['Edpp']] = (1 / self.Tqpp) * (Edp - Edpp + Iq * (self.xqp - self.xqpp))

        dEfq = 1 / self.Te * (-Efq + self.Kc * (Vref - Vf) + self.Kc / self.Tc * X_avr)

        dx[self.x_idx['Efq']] = dEfq
        dx[self.x_idx['Vf']] = 1 / self.Tm * (-Vf + Vac)
        dx[self.x_idx['Xavr']] = (Vref - Vf)

        # TODO check the equations for w*E''
        dx[self.x_idx['Id']] = self.wn / self.xdpp * (w * Edpp - Vd - self.ra * Id + w * self.xqpp * Iq)
        dx[self.x_idx['Iq']] = self.wn / self.xqpp * (w * Eqpp - Vq - self.ra * Iq - w * self.xdpp * Id)

        return dx


spec_gen_2_2 = [
    ('Sn', float64),
    ('Sb', float64),
    ('freq', float64),
    ('wn', float64),
    ('Pm', float64),
    ('Qm', float64),
    ('cosn', float64),
    ('ra', float64),
    ('xd', float64),
    ('xq', float64),
    ('xdp', float64),
    ('xqp', float64),
    ('xdpp', float64),
    ('xqpp', float64),
    ('xl', float64),
    ('xad', float64),
    ('xadu', float64),
    ('xaq', float64),
    ('xrld', float64),
    ('xrlq', float64),
    ('xfd', float64),
    ('x1d', float64),
    ('x1q', float64),
    ('x2q', float64),
    ('kfd', float64),
    ('k1d', float64),
    ('k1q', float64),
    ('k2q', float64),
    ('xdet_d', float64),
    ('xdet_q', float64),
    ('xfd_loop', float64),
    ('x1d_loop', float64),
    ('x1q_loop', float64),
    ('x2q_loop', float64),
    ('rfd', float64),
    ('r1d', float64),
    ('r1q', float64),
    ('r2q', float64),
    ('H', float64),
    ('dkd', float64),
    ('dpe', float64),
    ('Td0p', float64),
    ('Td0pp', float64),
    ('Tq0p', float64),
    ('Tq0pp', float64),
    ('Tdp', float64),
    ('Tdpp', float64),
    ('Tqp', float64),
    ('Tqpp', float64),
    ('Tj', float64),
    ('Vref', float64),
    ('Kc', float64),
    ('Tc', float64),
    ('Te', float64),
    ('Tm', float64),
    ('Pe0', float64),
    ('Efd0', float64),
    ('x_idx', types.DictType(types.unicode_type, types.int64)),
    ('nx', types.int64),
    ('states', types.ListType(types.unicode_type)),
    ('bus_ind', int64),
    ('x_ind', int64),
    ('type', int64),
    ('label', types.string),
    ('ID', int64),
    ('spec', types.ListType(types.unicode_type)),
]


# @jitclass(spec_gen_2_2)
class gen_2_2():
    def __init__(self):
        self.Sn = 100
        self.Sb = 100
        self.ID = -1
        self.Pm = 0
        self.Qm = 0
        self.cosn = 1

        self.freq = 50
        self.ra = 0.05
        self.xd = 2.2
        self.xq = 2
        self.xdp = 0.3
        self.xqp = 0.4
        self.xdpp = 0.2
        self.xqpp = 0.2
        self.xl = 0.15

        self.xrld = 0
        self.xrlq = 0

        self.dkd = 0
        self.dpe = 0

        self.H = 2  # rated to gen_MVA
        # self.D = 0
        self.Td0p = 7
        self.Td0pp = 0.05
        self.Tq0p = 1.5
        self.Tq0pp = 0.05

        self.calc_machine_parameters()

        # # AVR TODO move
        self.Pe0 = 0
        self.Efd0 = 1

        self.Vref = 1.0
        self.Kc = 16
        self.Tc = 2.67
        self.Te = 0.3
        self.Tm = 0.005

        #
        self.x_idx = typed.Dict.empty(*kv_ty)
        self.states = typed.List.empty_list(types.unicode_type)
        self.states.extend(['d', 'w', 'psi_d', 'psi_q', 'psi_fd', 'psi_1d', 'psi_1q', 'psi_2q', 'Id', 'Iq', 'Efd', 'Vf',
                            'Xavr'])  # The AVR is included
        #  TODO seperate the AVR from the generator model
        # self.states.extend(['d','w','psi_d','psi_q','psi_fd','psi_1d','psi_1q','psi_2q','Id','Iq'])
        self.nx = len(self.states)
        for i, state in enumerate(self.states):
            self.x_idx[state] = i

        self.type = ModelType.GEN_2_2
        self.label = 'generator'

    def calc_machine_parameters(self):
        """
        Calculate the Equivalent Circuit Parameters from the Short Circuit parameters

        Returns
        -------
        None.

        """

        wn = 2 * np.pi * self.freq

        # input parameters
        ra = 0.05

        xd = self.xd
        xdp = self.xdp
        xdpp = self.xdpp

        xq = self.xq
        xqp = self.xqp
        xqpp = self.xqpp

        xrld = self.xrld
        xrlq = self.xrlq

        xl = self.xl

        H = self.H  # rated to gen_MVA

        Td0p = self.Td0p
        Td0pp = self.Td0pp
        Tq0p = self.Tq0p
        Tq0pp = self.Tq0pp
        Tj = 2 * self.H

        Tdp = Td0p * xdp / xd
        Tdpp = Td0pp * xdpp / xdp
        Tqp = Tq0p * xqp / xq
        Tqpp = Tq0pp * xqpp / xqp

        # calculation d axis parameters
        xad = xd - xl

        x1 = xd - xl + xrld
        x2 = x1 - ((xd - xl) ** 2) / xd
        x3 = (x2 - x1 * xdpp / xd) / (1 - xdpp / xd)

        T1 = (xd / xdp) * Tdp + (1 - (xd / xdp) + (xd / xdpp)) * Tdpp
        T2 = Tdp + Tdpp

        a = (x2 * T1 - x1 * T2) / (x1 - x2)
        b = (x3 / (x3 - x2)) * Tdp * Tdpp

        Tsfd = -a / 2 + np.sqrt(a ** 2 / 4 - b)
        Ts1d = -a / 2 - np.sqrt(a ** 2 / 4 - b)

        xfd = (Tsfd - Ts1d) / ((T1 - T2) / (x1 - x2) + Ts1d / x3)
        x1d = (Ts1d - Tsfd) / ((T1 - T2) / (x1 - x2) + Tsfd / x3)

        rfd = xfd / (wn * Tsfd)
        r1d = x1d / (wn * Ts1d)

        # calculation q axis parameters
        xaq = xq - xl

        x1 = xq - xl + xrlq
        x2 = x1 - ((xq - xl) ** 2) / xq
        x3 = (x2 - x1 * xqpp / xq) / (1 - xqpp / xq)

        T1 = (xq / xqp) * Tqp + (1 - (xq / xqp) + (xq / xqpp)) * Tqpp
        T2 = Tqp + Tqpp

        a = (x2 * T1 - x1 * T2) / (x1 - x2)
        b = (x3 / (x3 - x2)) * Tqp * Tqpp

        Ts2q = -a / 2 + np.sqrt(a ** 2 / 4 - b)
        Ts1q = -a / 2 - np.sqrt(a ** 2 / 4 - b)

        x2q = (Ts2q - Ts1q) / ((T1 - T2) / (x1 - x2) + Ts1q / x3)
        x1q = (Ts1q - Ts2q) / ((T1 - T2) / (x1 - x2) + Ts2q / x3)

        r2q = x2q / (wn * Ts2q)
        r1q = x1q / (wn * Ts1q)

        #

        xdet_d = (xad + xrld) * (x1d + xfd) + xfd * x1d
        xdet_q = (xad + xrlq) * (x2q + x1q) + x2q * x1q

        kfd = (xad * x1d) / ((xad + xrld) * (x1d + xfd) + xfd * x1d)
        k1d = (xad * xfd) / ((xad + xrld) * (x1d + xfd) + xfd * x1d)

        k1q = (xaq * x2q) / ((xad + xrlq) * (x2q + x1q) + x2q * x1q)
        k2q = (xaq * x1q) / ((xad + xrlq) * (x2q + x1q) + x2q * x1q)

        xfd_loop = xad + xrld + xfd
        x1d_loop = xad + xrld + x1d
        x1q_loop = xaq + xrlq + x1q
        x2q_loop = xaq + xrlq + x2q

        self.wn = wn
        self.xad = xad
        self.xadu = xad
        self.xaq = xaq
        self.xfd = xfd
        self.x1d = x1d
        self.x1q = x1q
        self.x2q = x2q
        self.kfd = kfd
        self.k1d = k1d
        self.k1q = k1q
        self.k2q = k2q
        self.xdet_d = xdet_d
        self.xdet_q = xdet_q
        self.xfd_loop = xfd_loop
        self.x1d_loop = x1d_loop
        self.x1q_loop = x1q_loop
        self.x2q_loop = x2q_loop
        self.rfd = rfd
        self.r1d = r1d
        self.r1q = r1q
        self.r2q = r2q
        self.Tj = Tj

    def dx_dt(self, xm, um):
        """
        Generator model 2.2 differential equations.
        Generator current is included as a state variable.

        Parameters
        ----------
        xm : ndarray
            State vector.
        um : ndarray
            Input vector.

        Returns
        -------
        dx : ndarray
            State derivatives.
        """

        Id = xm[self.x_idx['Id']]
        Iq = xm[self.x_idx['Iq']]

        d = xm[self.x_idx['d']]
        w = xm[self.x_idx['w']]
        psi_d = xm[self.x_idx['psi_d']]
        psi_q = xm[self.x_idx['psi_q']]
        psi_fd = xm[self.x_idx['psi_fd']]
        psi_1d = xm[self.x_idx['psi_1d']]
        psi_1q = xm[self.x_idx['psi_1q']]
        psi_2q = xm[self.x_idx['psi_2q']]

        Efd = xm[self.x_idx['Efd']]  # TODO seperate the avr from the generator to simplify using different avr models
        Vf = xm[self.x_idx['Vf']]
        X_avr = xm[self.x_idx['Xavr']]

        Efd = np.clip(np.array([Efd]), 0.0, 5.0)[0]

        vx = um[0]
        vy = um[1]
        Vref = um[2]
        # Efd = um[3]

        Vac = np.sqrt(vx ** 2 + vy ** 2)

        Vd = (vx * np.cos(d) + vy * np.sin(d))
        Vq = (-vx * np.sin(d) + vy * np.cos(d))

        vfd = self.rfd / self.xadu * Efd

        te = (Iq * psi_d - Id * psi_q) / self.cosn
        tm = 0  # TODO include torque input
        tdkd = self.dkd * (w - 1)
        tdpe = self.dpe / w * (w - 1)

        ifd = self.kfd * Id + (self.x1d_loop * psi_fd - (self.xad + self.xrld) * psi_1d) / self.xdet_d
        i1d = self.k1d * Id + (self.xfd_loop * psi_1d - (self.xad + self.xrld) * psi_fd) / self.xdet_d
        i1q = self.k1q * Iq + (self.x2q_loop * psi_1q - (self.xaq + self.xrlq) * psi_2q) / self.xdet_q
        i2q = self.k2q * Iq + (self.x1q_loop * psi_2q - (self.xaq + self.xrlq) * psi_1q) / self.xdet_q

        dpsi_fd = self.wn * (vfd - self.rfd * ifd)
        dpsi_1d = self.wn * (-self.r1d * i1d)
        dpsi_1q = self.wn * (-self.r1q * i1q)
        dpsi_2q = self.wn * (-self.r2q * i2q)

        Edpp = -w * (self.k1q * psi_1q + self.k2q * psi_2q) + (
                self.kfd / self.wn * dpsi_fd + self.k1d / self.wn * dpsi_1d)
        Eqpp = w * (self.kfd * psi_fd + self.k1d * psi_1d) + (
                self.k1q / self.wn * dpsi_1q + self.k2q / self.wn * dpsi_2q)

        delta_w = self.wn * (w - 1)
        dx = np.zeros(len(xm))
        dx[self.x_idx['d']] = delta_w
        dx[self.x_idx['w']] = (1 / self.Tj) * (tm - te - tdkd - tdpe)
        dx[self.x_idx['psi_d']] = self.wn * (Vd + self.ra * Id + w * psi_q)
        dx[self.x_idx['psi_q']] = self.wn * (Vq + self.ra * Iq - w * psi_d)
        dx[self.x_idx['psi_fd']] = dpsi_fd
        dx[self.x_idx['psi_1d']] = dpsi_1d
        dx[self.x_idx['psi_1q']] = dpsi_1q
        dx[self.x_idx['psi_2q']] = dpsi_2q

        dx[self.x_idx['Id']] = (self.wn / self.xdpp * (Edpp - Vd - self.ra * Id + w * self.xqpp * Iq))
        dx[self.x_idx['Iq']] = (self.wn / self.xqpp * (Eqpp - Vq - self.ra * Iq - w * self.xdpp * Id))

        dEfd = 1 / self.Te * (-Efd + self.Kc * (Vref - Vf) + self.Kc / self.Tc * X_avr)

        dx[self.x_idx['Efd']] = dEfd
        dx[self.x_idx['Vf']] = 1 / self.Tm * (-Vf + Vac)
        dx[self.x_idx['Xavr']] = (Vref - Vf)

        return dx

    def abcd_linear(self, xo, uo):
        delta = xo[self.x_idx['d']]
        w = xo[self.x_idx['w']]
        i_d = xo[self.x_idx['Id']]
        i_q = xo[self.x_idx['Iq']]
        psi_d = xo[self.x_idx['psi_d']]
        psi_q = xo[self.x_idx['psi_q']]
        psi_fd = xo[self.x_idx['psi_fd']]
        psi_1d = xo[self.x_idx['psi_1d']]
        psi_2q = xo[self.x_idx['psi_2q']]
        psi_1q = xo[self.x_idx['psi_1q']]
        Vd = uo[0]
        Vq = uo[1]

        Ag = np.array([[0.0, self.wn, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, (-self.dkd - self.dpe / w + self.dpe * (w - 1) / w ** 2) / self.Tj,
                        -i_q / (self.Tj * self.cosn), i_d / (self.Tj * self.cosn), 0.0, 0.0, 0.0, 0.0,
                        psi_q / (self.Tj * self.cosn), -psi_d / (self.Tj * self.cosn), 0.0, 0.0, 0.0],
                       [0.0, self.wn * psi_q, 0.0, self.wn * w, 0.0, 0.0, 0.0, 0.0, self.ra * self.wn, 0.0, 0.0, 0.0,
                        0.0],
                       [0.0, -self.wn * psi_d, -self.wn * w, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.ra * self.wn, 0.0, 0.0,
                        0.0],
                       [0.0, 0.0, 0.0, 0.0, -self.rfd * self.wn * self.x1d_loop / self.xdet_d,
                        -self.rfd * self.wn * (-self.xad - self.xrld) / self.xdet_d, 0.0, 0.0,
                        -self.kfd * self.rfd * self.wn, 0.0, self.rfd * self.wn / self.xadu, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, -self.r1d * self.wn * (-self.xad - self.xrld) / self.xdet_d,
                        -self.r1d * self.wn * self.xfd_loop / self.xdet_d, 0.0, 0.0, -self.k1d * self.r1d * self.wn,
                        0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -self.r1q * self.wn * self.x2q_loop / self.xdet_q,
                        -self.r1q * self.wn * (-self.xaq - self.xrlq) / self.xdet_q, 0.0,
                        -self.k1q * self.r1q * self.wn, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -self.r2q * self.wn * (-self.xaq - self.xrlq) / self.xdet_q,
                        -self.r2q * self.wn * self.x1q_loop / self.xdet_q, 0.0, -self.k2q * self.r2q * self.wn, 0.0,
                        0.0, 0.0],
                       [0.0, self.wn * (-self.k1q * psi_1q - self.k2q * psi_2q + self.xqpp * i_q) / self.xdpp, 0.0, 0.0,
                        self.wn * (-self.k1d * self.r1d * (
                                -self.xad - self.xrld) / self.xdet_d - self.kfd * self.rfd * self.x1d_loop / self.xdet_d) / self.xdpp,
                        self.wn * (-self.k1d * self.r1d * self.xfd_loop / self.xdet_d - self.kfd * self.rfd * (
                                -self.xad - self.xrld) / self.xdet_d) / self.xdpp,
                        -self.k1q * self.wn * w / self.xdpp, -self.k2q * self.wn * w / self.xdpp,
                        self.wn * (-self.k1d ** 2 * self.r1d - self.kfd ** 2 * self.rfd - self.ra) / self.xdpp,
                        self.wn * self.xqpp * w / self.xdpp, self.kfd * self.rfd * self.wn / (self.xadu * self.xdpp),
                        0.0, 0.0],
                       [0.0, self.wn * (self.k1d * psi_1d + self.kfd * psi_fd - self.xdpp * i_d) / self.xqpp, 0.0, 0.0,
                        self.kfd * self.wn * w / self.xqpp, self.k1d * self.wn * w / self.xqpp, self.wn * (
                                -self.k1q * self.r1q * self.x2q_loop / self.xdet_q - self.k2q * self.r2q * (
                                -self.xaq - self.xrlq) / self.xdet_q) / self.xqpp, self.wn * (
                                -self.k1q * self.r1q * (
                                -self.xaq - self.xrlq) / self.xdet_q - self.k2q * self.r2q * self.x1q_loop / self.xdet_q) / self.xqpp,
                        -self.wn * self.xdpp * w / self.xqpp,
                        self.wn * (-self.k1q ** 2 * self.r1q - self.k2q ** 2 * self.r2q - self.ra) / self.xqpp, 0.0,
                        0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1 / self.Te, -self.Kc / self.Te,
                        self.Kc / (self.Tc * self.Te)],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1 / self.Tm, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1, 0.0]
                       ], dtype=float)

        Bvg = np.array([[0.0, 0.0],
                        [0.0, 0.0],
                        [self.wn, 0.0],
                        [0.0, self.wn],
                        [0.0, 0.0],
                        [0.0, 0.0],
                        [0.0, 0.0],
                        [0.0, 0.0],
                        [-self.wn / self.xdpp, 0.0],
                        [0.0, -self.wn / self.xqpp],
                        [0.0, 0.0],
                        [Vd / (self.Tm * np.sqrt(Vd ** 2 + Vq ** 2)), Vq / (self.Tm * np.sqrt(Vd ** 2 + Vq ** 2))],
                        [0.0, 0.0],
                        ], dtype=float)

        Tg = np.array([[np.cos(delta), np.sin(delta)],
                       [-np.sin(delta), np.cos(delta)]], dtype=float)

        Rvg = np.zeros((2, self.nx), dtype=float)
        Rvg[0, self.x_idx['d']] = Vq
        Rvg[1, self.x_idx['d']] = -Vd

        Pg = np.zeros((2, self.nx), dtype=float)
        Pg[0, self.x_idx['Id']] = 1
        Pg[1, self.x_idx['Iq']] = 1
        Pg[0, self.x_idx['d']] = -i_q
        Pg[1, self.x_idx['d']] = i_d

        Cg = Tg.T @ Pg

        Dg = np.zeros((2, 2), dtype=float)

        A = Ag + Bvg @ Rvg
        B = Bvg @ Tg
        C = Cg * self.Sn / self.Sb
        D = Dg

        return A, B, C, D


spec_savr = [
    ('Vref', float64),
    ('Kc', float64),
    ('Tc', float64),
    ('Te', float64),
    ('Tm', float64),
    ('x_idx', types.DictType(types.unicode_type, types.int64)),
    ('x_ind', int64),
    ('nx', int64),
    ('states', types.ListType(types.unicode_type)),
    ('type', int64),
    ('label', types.string),
    ('ID', int64),
]


# @jitclass(spec_savr)
class SAVR():
    """
    Simple AVR. # XXX NOT FINISHED

    """

    def __init__(self):
        self.Vref = 1
        self.Kc = 16
        self.Tc = 2.67
        self.Te = 0.3
        self.Tm = 0.005
        self.x_idx = typed.Dict.empty(*kv_ty)
        self.states = typed.List.empty_list(types.unicode_type)
        self.states.extend(['Efd', 'Vf', 'Xavr'])
        self.nx = len(self.states)
        for i, state in enumerate(self.states):
            self.x_idx[state] = i

        self.type = ModelType.SAVR
        self.label = 'SAVR'
