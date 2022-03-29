import numpy as np

# from Load_parameters import conv_idx, nx_conv, sc_idx, nx_sc
from dataset_creation.NSWPH_models import ModelType, CtrlMode
from dataset_creation.NSWPH_system_models import nswph_models


def PowerFlowNewton(Ybus, Sbus, V0, ref, pv_index, pq_index, pqv_index, max_iter, err_tol, kqv=None):
    success = 0  # Initialization of status flag and iteration counter
    n = 0
    V = V0
    div = 1
    # print(' iteration maximum P & Q mismatch (pu)')
    # print(' --------- ---------------------------')
    # Determine mismatch between initial guess and and specified value for P and Q
    if len(pqv_index) > 0 and (kqv is None or len(kqv) == 0):
        pv_index = np.append(pv_index, pqv_index)
        pq_pqv_inds = pq_index
    elif len(pqv_index) > 0 and kqv[:, 1] == 0:  # TODO check
        pq_index = np.append(pq_index, pqv_index)
        pqv_index = np.empty(0)
        pq_pqv_inds = pq_index
    else:
        pq_pqv_inds = np.append(pq_index, pqv_index)
    F = calculate_F(Ybus, Sbus, V, pv_index, pq_index, pqv_index, kqv, div)
    # Check if the desired tolerance is reached
    success, normF = CheckTolerance(F, n, err_tol)
    # Start the Newton iteration loop
    nL = [normF, normF]
    while (not success) and (n < max_iter):
        n += 1  # Update counter
        # Compute derivatives and generate the Jacobian matrix
        J_dS_dVm, J_dS_dTheta = generate_Derivatives(Ybus, V)
        J = generate_Jacobian(J_dS_dVm, J_dS_dTheta, pv_index, pq_pqv_inds)
        # Compute the update step
        dx = np.linalg.solve(J, F)
        # Update voltages and check if tolerance is now reached
        V = Update_Voltages(dx, V, pv_index, pq_pqv_inds)
        F = calculate_F(Ybus, Sbus, V, pv_index, pq_index, pqv_index, kqv, div)
        success, normF = CheckTolerance(F, n, err_tol)

        if round(normF, 8) == round(nL[0], 8):
            div = 2
        # else:
        #     div = 1
        # print(div)
        nL.pop(0)
        nL.append(normF)

    if not success:  # print out message concerning wether the power flow converged or not
        print('No Convergence !!!\n Stopped after %d iterations without solution...' % (n,))

    if any(np.isnan(V)):
        print('No Convergence !!!\n NAN in voltage vector')
        success = False
    # else :
    #     print('The Newton Rapson Power Flow Converged in %d iterations!' %(n, ))
    return V, success, n


def calculate_F(Ybus, Sbus, V, pv_index, pq_index, pqv_index, kqv, div):
    Delta_S = Sbus - V * (Ybus.dot(
        V)).conj()  # This function calculates the mismatch between the specified values of P and Q (In term of S)

    if kqv is not None and kqv[0, 0] > 0:
        for vsc_idx, (kq, kv, qref, vref) in zip(pqv_index, kqv):
            # Qref = np.imag(Sbus[vsc_idx])

            Qm = np.imag(V * (Ybus.dot(V)).conj())[vsc_idx]  # /(b*n)
            # print(V[vsc_idx])
            Vm = abs(V[vsc_idx])
            # Delta_S[vsc_idx] = np.real(Delta_S[vsc_idx])-1j*(lp.Kq*(Qm-lp.Qref*lp.Sn/noff/lp.Sb)+lp.Kv*(Vm-lp.Vref)*lp.Sn/noff/lp.Sb)#-1j*Qm
            # Delta_S[vsc_idx] = np.real(Delta_S[vsc_idx])-1j*(lp.Kq*(Qm-lp.Qref*lp.Sn*noff/lp.Sb)+lp.Kv*(Vm-lp.Vref)*lp.Sn*noff/lp.Sb)/10
            # Delta_S[vsc_idx] = np.real(Delta_S[vsc_idx])-1j*(kq*(Qm-0*b*n)+kv*(Vm-1.01)*b*n)/10

            # kq = kq*b*n
            # kv = kv*b*n
            Delta_S[vsc_idx] = np.real(Delta_S[vsc_idx]) - 1j * (kq * (Qm - qref) + kv * (Vm - vref)) / 10

    # Sbus[vsc_idx] = (Sbus[vsc_idx]+Delta_S[vsc_idx])/2
    # We only use the above function for PQ and PV buses.
    Delta_P = np.real(Delta_S) / div
    Delta_Q = np.imag(Delta_S) / div
    # print(Delta_Q)
    # pq_pqv_inds = np.append(pq_index, pqv_index)
    if len(pqv_index) > 0 and kqv is not None and kqv[0, 0] > 0:
        #     pv_index = np.append(pv_index, pqv_index)
        #     pq_pqv_inds = pq_index
        # else:
        pq_pqv_inds = np.append(pq_index, pqv_index)
    else:
        pq_pqv_inds = pq_index
    F = np.concatenate((Delta_P[pv_index], Delta_P[pq_pqv_inds], Delta_Q[pq_pqv_inds]), axis=0)
    # print('F',F)
    return F


def CheckTolerance(F, n, err_tol):
    normF = np.linalg.norm(F, np.inf)
    # print(normF)
    if normF > err_tol:
        success = 0
        # print('Not Success')
    else:
        success = 1
    #     print('Success')
    # print('Highest error %.3f' %normF)
    return success, normF


def generate_Derivatives(Ybus, V):
    J_ds_dVm = np.diag(V / np.absolute(V)).dot(np.diag((Ybus.dot(V)).conj())) + \
               np.diag(V).dot(Ybus.dot(np.diag(V / np.absolute(V))).conj())
    J_dS_dTheta = 1j * np.diag(V).dot((np.diag(Ybus.dot(V)) - Ybus.dot(np.diag(V))).conj())
    return J_ds_dVm, J_dS_dTheta


def generate_Jacobian(J_dS_dVm, J_dS_dTheta, pv_index, pq_index):
    pvpq_ind = np.append(pv_index, pq_index)

    J_11 = np.real(J_dS_dTheta[np.ix_(pvpq_ind, pvpq_ind)])
    J_12 = np.real(J_dS_dVm[np.ix_(pvpq_ind, pq_index)])
    J_21 = np.imag(J_dS_dTheta[np.ix_(pq_index, pvpq_ind)])
    J_22 = np.imag(J_dS_dVm[np.ix_(pq_index, pq_index)])

    J = np.block([[J_11, J_12], [J_21, J_22]])
    # print('J',J)
    return J


def Update_Voltages(dx, V, pv_index, pq_index):
    N1 = 0;
    N2 = len(pv_index)  # dx[N1:N2]-ang. on the pv buses
    N3 = N2;
    N4 = N3 + len(pq_index)  # dx[N3:N4]-ang. on the pq buses
    N5 = N4;
    N6 = N5 + len(pq_index)  # dx[N5:N6]-mag. on the pq buses
    Theta = np.angle(V);
    Vm = np.absolute(V)
    if len(pv_index) > 0:
        Theta[pv_index] += dx[N1:N2]
    if len(pq_index) > 0:
        Theta[pq_index] += dx[N3:N4]
        Vm[pq_index] += dx[N5:N6]
    V = Vm * np.exp(1j * Theta)
    # print('V',V)
    return V


def DisplayResults(V, Ybus, Y_from, Y_to, br_f, br_t, buscode):
    S_to = V[br_t] * (Y_to.dot(V)).conj()
    S_from = V[br_f] * (Y_from.dot(V)).conj()
    S_inj = V * (Ybus.dot(V)).conj()

    dash = '=' * 60
    print(dash)
    print('|{:^58s}|'.format('Bus results'))
    print(dash)
    print('{:^6s} {:^17s} {:^17s} {:^17s}'.format('Bus', 'Voltage', 'Generation', 'Load'))
    print(
        '{:^6s} {:^8s} {:^8s} {:^8s} {:^8s} {:^8s} {:^8s}'.format('#', 'Mag(pu)', 'Ang(deg)', 'P(pu)', 'Q(pu)', 'P(pu)',
                                                                  'Q(pu)'))
    print(
        '{:^6s} {:^8s} {:^8s} {:^8s} {:^8s} {:^8s} {:^8s}'.format('-' * 6, '-' * 8, '-' * 8, '-' * 8, '-' * 8, '-' * 8,
                                                                  '-' * 8))

    for i in range(0, len(V)):
        if np.real(S_inj[i]) > 0:
            print('{:^6d} {:^8.3f} {:^8.3f} {:^8.3f} {:^8.3f} {:^8s} {:^9s}'.format(i + 1, np.abs(V[i]),
                                                                                    np.rad2deg(np.angle(V[i])),
                                                                                    np.real(S_inj[i]),
                                                                                    np.imag(S_inj[i]), '-', '-'))
        else:
            print('{:^6d} {:^8.3f} {:^8.3f} {:^8s} {:^8s} {:^8.3f} {:^9.3f}'.format(i + 1, np.abs(V[i]),
                                                                                    np.rad2deg(np.angle(V[i])), '-',
                                                                                    '-', -np.real(S_inj[i]),
                                                                                    -np.imag(S_inj[i])))

    print(dash)
    print('|{:^58s}|'.format('Branch Flow'))
    print(dash)
    print(
        '{:^6s} {:<6s} {:<6s} {:^19s} {:^19s}'.format('Branch', 'From', 'To', 'From bus Injection', 'To bus Injection'))
    print('{:^6s} {:<6s} {:<6s} {:^9s} {:^9s} {:^9s} {:^9s}'.format('#', 'Bus', 'Bus', 'P(pu)', 'Q(pu)', 'P(pu)',
                                                                    'Q(pu)'))

    print(
        '{:^5s} {:^5s} {:^5s} {:^8s} {:^8s} {:^8s} {:^8s}'.format('-' * 6, '-' * 6, '-' * 6, '-' * 9, '-' * 9, '-' * 9,
                                                                  '-' * 9))

    for i in range(0, len(br_f)):
        print('{:^6d} {:^6d} {:^6d} {:^9.3f} {:^9.3f} {:^9.3f} {:^9.3f}'.format(i + 1, br_f[i] + 1, br_t[i] + 1,
                                                                                -np.real(S_from[i]),
                                                                                -np.imag(S_from[i]), -np.real(S_to[i]),
                                                                                -np.imag(S_to[i])))
    return


def initialize_state_vector(npr, models, max_iter=2000, err_tol=1e-7, pr=False):
    # TODO this function is not fully generic and only works for NSWPH and very similar systems

    # Carry out the power flow analysis ...
    # Sbus = npr.Sbus
    Sbus = np.zeros(npr.n_bus, dtype=complex)
    for m in models:
        if hasattr(m, 'Pref'):
            Sbus[m.bus_ind] += m.Pref * m.Sn / m.Sb
        if hasattr(m, 'Qref'):
            Sbus[m.bus_ind] += 1j * m.Qref * m.Sn / m.Sb

    if len(npr.pqv_index) > 0:
        kqv = np.zeros((len(npr.pqv_index), 4))
        for m in models:
            idx = np.flatnonzero(npr.pqv_index == m.bus_ind)
            if len(idx) > 0:
                kqv[idx, 0] = m.Kq
                kqv[idx, 1] += m.Kv * m.Sn / m.Sb
                kqv[idx, 2] += m.Qref * m.Sn / m.Sb
                kqv[idx, 3] = m.Vref
    else:
        kqv = None

    V0, success1, n = PowerFlowNewton(npr.Ybus, Sbus, npr.V0, npr.ref, npr.pv_index, npr.pq_index, npr.pqv_index,
                                      max_iter, err_tol, None)

    if success1:
        I = (npr.Ybus.dot(V0))
        Sbus = (V0 * I.conj())

        V, success, n = PowerFlowNewton(npr.Ybus, Sbus, V0, npr.ref, npr.pv_index, npr.pq_index, npr.pqv_index,
                                        max_iter, err_tol, kqv)

        if success:

            I = (npr.Ybus.dot(V))
            S = (V * I.conj())

            Pm_sc = S[npr.ref].real  # *sc1.Sb/sc1.Sn

            # Move active power from the slack bus to the offshore converters
            # TODO it might be possible to make the hub the slack bus but I have to
            # find a way to deal with the voltage droop

            vsc_secondary_ctrl = False
            ref_is_vs = False
            gens = []
            for k, m in enumerate(models):
                if hasattr(m, 'Kif') and m.Kif > 0:
                    vsc_secondary_ctrl = True
                    bus_ind = m.bus_ind
                if m.type in (ModelType.GEN_ORD_6, ModelType.GEN_2_2):
                    gens.append(k)
                if m.type == ModelType.VS and m.bus_ind == npr.ref[
                    0]:  # TODO if the system is split into subsystems, each with a slack bus
                    ref_is_vs = True
            if vsc_secondary_ctrl and not ref_is_vs:
                while abs(Pm_sc.sum()) > 1e-6:
                    Sbus[bus_ind] = Sbus[bus_ind] + (Pm_sc[0]) / 2  # TODO get the hub index from the model
                    V, success, n = PowerFlowNewton(npr.Ybus, Sbus, V, npr.ref, npr.pv_index, npr.pq_index,
                                                    npr.pqv_index, max_iter, err_tol, kqv)
                    I = (npr.Ybus.dot(V))
                    S = (V * I.conj())

                    Pm_sc = S[npr.ref].real  # *sc1.Sb/sc1.Sn
            else:
                bus_ind = npr.ref[0]

            if len(gens) > 0:
                Sbus_temp = (V0 * I.conj())
                for ind in gens:
                    gen = models[ind]
                    if not gen.bus_ind == bus_ind:
                        # gen_loss_temp = sum(abs(I[sc_bus_inds])**2)*gen.ra*gen.Sb/gen.Sn
                        gen_loss_temp = (abs(I[gen.bus_ind]) ** 2) * gen.ra * gen.Sb / gen.Sn
                        Sbus_temp[bus_ind] += gen_loss_temp
                        Sbus_temp[gen.bus_ind] -= gen_loss_temp

                V, success, n = PowerFlowNewton(npr.Ybus, Sbus_temp, V, npr.ref, npr.pv_index, npr.pq_index,
                                                npr.pqv_index, max_iter, err_tol, kqv)
                I = (npr.Ybus.dot(V))
                # Sbus = (V*I.conj())
                for ind in gens:
                    gen = models[ind]
                    if not gen.bus_ind == bus_ind:
                        # gen_loss = sum(abs(I[sc_bus_inds])**2)*gen.ra*gen.Sb/gen.Sn
                        gen_loss = (abs(I[gen.bus_ind]) ** 2) * gen.ra * gen.Sb / gen.Sn
                        Sbus[bus_ind] += gen_loss
                        Sbus[gen.bus_ind] -= gen_loss

                    # nsc = 1
                    # if nsc > 1:
                    #     for i in range(1,nsc):
                    #         Sbus[sc_bus_inds[i]] = -(sum(abs(I[sc_bus_inds])**2)*gen.ra*gen.Sb/gen.Sn)/nsc

            V, success, n = PowerFlowNewton(npr.Ybus, Sbus, V, npr.ref, npr.pv_index, npr.pq_index, npr.pqv_index,
                                            max_iter, err_tol, kqv)

        elif success1:
            V = V0
    else:
        success = success1
        V = V0

    if pr:
        # print(abs(V))
        t = 1

    # print(abs(V))
    I = (npr.Ybus.dot(V))
    Ifr = npr.Y_from.dot(V)
    Ito = npr.Y_to.dot(V)
    S = (V * I.conj())

    n_on = len(npr.ref) - 1
    for k in range(n_on):
        S[-1 - k] -= S[0].real / n_on

    I = (S / V).conj()

    nbus = npr.n_bus
    nbranch = npr.n_br

    i = 0

    N = np.zeros(npr.n_bus)
    nx = 2 * (nbus + nbranch)

    for model in models:  # TODO doesn't work for all configurations, need to find a better way
        if hasattr(model, 'bus_ind') and model.type == ModelType.VSC_1:
            N[model.bus_ind] += 1
        nx += model.nx

    x = np.zeros(nx)

    for model in models:
        if model.type == ModelType.VSC_1:
            model.x_ind = i

            bus_ind = model.bus_ind
            if model.ctrl == CtrlMode.VDC_Q:
                Pm = (S[bus_ind].real * model.Sb / model.Sn)
                Qm = model.Qref
                Vdc = 1
            else:
                Pm = (S[bus_ind].real * model.Sb / model.Sn) / N[bus_ind]
                Qm = (S[bus_ind].imag * model.Sb / model.Sn) / N[bus_ind]
                Vdc = 1 - 0.00419047619047619 * Pm / 1  # TODO this is not generic

            Theta = np.angle(V[bus_ind])

            vx = V[bus_ind].real
            vy = V[bus_ind].imag

            ix = I[bus_ind].real * model.Sb / model.Sn / N[bus_ind]
            iy = I[bus_ind].imag * model.Sb / model.Sn / N[bus_ind]

            vd = (vx * np.cos(Theta) + vy * np.sin(Theta))
            vq = (-vx * np.sin(Theta) + vy * np.cos(Theta))

            i_d = Pm / vd  # (ix*np.cos(Theta)+iy*np.sin(Theta))
            i_q = -Qm / vd  # (-ix*np.sin(Theta)+iy*np.cos(Theta))
            # Vdc = 1.0 # TODO calculate Vdc
            x[i + model.x_idx['Id']] = i_d
            x[i + model.x_idx['Iq']] = i_q
            # x[i+model.x_idx['Idc']] = Pm/Vdc
            x[i + model.x_idx['Md']] = model.Rt * i_d / model.Kic
            x[i + model.x_idx['Mq']] = model.Rt * i_q / model.Kic  ###
            x[i + model.x_idx['Madd']] = vd
            x[i + model.x_idx['Madq']] = vq
            x[i + model.x_idx['Theta']] = Theta
            x[i + model.x_idx['Xpll']] = 1 / model.Ki_pll
            x[i + model.x_idx['Xf']] = 0
            x[i + model.x_idx['Xp']] = i_d / model.Kip
            x[i + model.x_idx['Xq']] = i_q / model.Kiq
            x[i + model.x_idx['Pm']] = Pm
            x[i + model.x_idx['Qm']] = Qm
            x[i + model.x_idx['Vm']] = vd

            i += model.nx

        elif model.type == ModelType.GEN_ORD_6:
            model.x_ind = i

            bus_ind = model.bus_ind
            Pm = S[bus_ind].real * model.Sb / model.Sn
            Qm = S[bus_ind].imag * model.Sb / model.Sn

            Vg = V[bus_ind]
            Ig = I[bus_ind]

            Sb = 100
            Vb = 20
            Ib = Sb / Vb

            vx = Vg.real
            vy = Vg.imag

            ix = Ig.real
            iy = Ig.imag

            Z = (model.ra + 1j * model.xq)
            Eq = Vg + Z * Ig * model.Sb / model.Sn

            d = np.angle(
                Eq) - 1 * np.pi / 2  # Dq transformation in the book is different from the slides so -1*np.pi/2 is required

            Vd = vx * np.cos(d) + vy * np.sin(d)
            Vq = -vx * np.sin(d) + vy * np.cos(d)
            Id = (ix * np.cos(d) + iy * np.sin(d)) * (model.Sb / model.Sn)
            Iq = (-ix * np.sin(d) + iy * np.cos(d)) * (model.Sb / model.Sn)

            Efq = Vq + model.ra * Iq + model.xd * Id
            model.Efq0 = Efq  # for constant excitation voltage
            model.Pe0 = (Pm + (Id ** 2 + Iq ** 2) * model.ra) * model.Sn / model.Sb
            x[i + model.x_idx['d']] = d
            x[i + model.x_idx['w']] = 1
            x[i + model.x_idx['Eqp']] = Vq + model.ra * Iq + model.xdp * Id
            x[i + model.x_idx['Eqpp']] = Vq + model.ra * Iq + model.xdpp * Id
            x[i + model.x_idx['Edp']] = Vd + model.ra * Id - model.xqp * Iq
            x[i + model.x_idx['Edpp']] = Vd + model.ra * Id - model.xqpp * Iq
            x[i + model.x_idx['Efq']] = Efq
            x[i + model.x_idx['Vf']] = abs(Vg)
            x[i + model.x_idx['Xavr']] = Efq * model.Tc / model.Kc

            if 'Id' in model.x_idx:
                x[i + model.x_idx['Id']] = Id
                x[i + model.x_idx['Iq']] = Iq

            i += model.nx

        elif model.type == ModelType.GEN_2_2:
            model.x_ind = i

            bus_ind = model.bus_ind
            Pm = S[bus_ind].real * model.Sb / model.Sn
            Qm = S[bus_ind].imag * model.Sb / model.Sn

            Vg = V[bus_ind]
            Ig = I[bus_ind]

            vx = Vg.real
            vy = Vg.imag

            ix = Ig.real
            iy = Ig.imag

            Z = (model.ra + 1j * model.xq)
            Eq = Vg + Z * Ig * model.Sb / model.Sn

            d = np.angle(
                Eq) - 1 * np.pi / 2  # Dq transformation in the book is different from the slides so -1*np.pi/2 is required

            Vd = vx * np.cos(d) + vy * np.sin(d)
            Vq = -vx * np.sin(d) + vy * np.cos(d)
            Id = (ix * np.cos(d) + iy * np.sin(d)) * (model.Sb / model.Sn)
            Iq = (-ix * np.sin(d) + iy * np.cos(d)) * (model.Sb / model.Sn)

            psi_q = model.ra * Id - Vd
            psi_d = Vq + model.ra * Iq

            # psi_d = -(model.xl+model.xad)*Id+model.xad*ifd+model.xad*i1d

            ifd = ((model.xad + model.xl) * Id + psi_d) / model.xad

            Efd = Vq + model.ra * Iq + model.xd * Id
            # ifd = Efd/model.rfd

            i1d = 0
            i1q = 0
            i2q = 0

            psi_fd = -model.xad * Id + (model.xad + model.xrld + model.xfd) * ifd + (model.xad + model.xrld) * i1d
            psi_1d = -model.xad * Id + (model.xad + model.xrld) * ifd + (model.xad + model.xrld + model.x1d) * i1d

            ie = model.xadu * ifd
            ve = ie
            vfd = model.rfd / model.xadu * ve

            model.Efd0 = Efd  # for constant excitation voltage
            # model.Pe0 = (Pm+(Id**2+Iq**2)*model.ra)*model.Sn/model.Sb
            x[i + model.x_idx['d']] = d
            x[i + model.x_idx['w']] = 1
            x[i + model.x_idx['psi_d']] = psi_d
            x[i + model.x_idx['psi_q']] = psi_q
            x[i + model.x_idx['psi_fd']] = psi_fd
            x[i + model.x_idx['psi_1d']] = -model.xad * Id + (model.xad + model.xrld) * ifd + (
                        model.xad + model.xrld + model.x1d) * i1d
            x[i + model.x_idx[
                'psi_1q']] = 0  # -model.xaq*Id+(model.xaq+model.xrlq)*i2q+(model.xaq+model.xrlq+model.x1q)*i1q
            x[i + model.x_idx[
                'psi_2q']] = 0  # -model.xaq*Id+(model.xaq+model.xrlq+model.x2q)*i2q+(model.xaq+model.xrlq)*i1q
            x[i + model.x_idx['Id']] = Id
            x[i + model.x_idx['Iq']] = Iq

            x[i + model.x_idx['Efd']] = Efd
            x[i + model.x_idx['Vf']] = abs(Vg)
            x[i + model.x_idx['Xavr']] = Efd * model.Tc / model.Kc

            i += model.nx

        elif model.type == ModelType.DC_LINE:
            model.x_ind = i
            vsc_from = models[model.f]
            vsc_to = models[model.t]
            vsc_from.x_dc = i + model.x_idx['Vf']
            vsc_to.x_dc = i + model.x_idx['Vt']

            model.Sf = models[model.f].Sn
            model.St = models[model.t].Sn
            model.x_If = models[model.f].x_ind + models[model.f].x_idx['Idc']
            model.x_It = models[model.t].x_ind + models[model.t].x_idx['Idc']
            # Pf = -(S[vsc_from.bus_ind].real*vsc_from.Sb/vsc_from.Sn)/N[vsc_from.bus_ind]
            Pf = -(S[vsc_from.bus_ind].real) / N[vsc_from.bus_ind]
            if vsc_from.ctrl == CtrlMode.VDC_Q:
                Vf = vsc_from.Vref
                Il = Pf / Vf
                Vt = Vf + Il * model.R
            elif vsc_to.ctrl == CtrlMode.VDC_Q:
                Vt = vsc_to.Vref
                Il = Pf / Vt
                Vf = Vt + Il * model.R
            else:
                Vf = 1
                Vt = 1
                Il = Pf

            x[i + model.x_idx['Il']] = Il
            x[i + model.x_idx['Vf']] = Vf
            x[i + model.x_idx['Vt']] = Vt

            i += model.nx
        elif model.type == ModelType.VS:
            I2 = np.conj(S / V)  # XXX
            model.x_ind = i
            Ix = I2[model.bus_ind].real
            Iy = I2[model.bus_ind].imag
            Vx = V[model.bus_ind].real
            Vy = V[model.bus_ind].imag
            mVx = Vx + model.R * Ix - model.L * Iy
            mVy = Vy + model.R * Iy + model.L * Ix
            model.V0 = abs(mVx + 1j * mVy)

            x[i + model.x_idx['phi']] = np.angle(mVx + 1j * mVy)
            if model.nx > 1:  # XXX temp fix
                x[i + model.x_idx['Ix']] = Ix
                x[i + model.x_idx['Iy']] = Iy

            i += model.nx

    npr.x_ind = i
    for n in range(nbus):
        x[i] = V[n].real
        x[i + 1] = V[n].imag

        i = i + 2

    for n in range(nbranch):
        # TODO choose appropriate current direction
        x[i] = Ifr[n].real
        x[i + 1] = Ifr[n].imag

        i = i + 2

    for model in models:
        if model.type == ModelType.VSC_1 and model.Kif > 0:
            if not np.isnan(Sbus[model.bus_ind].real):
                model.Pref = (Sbus[model.bus_ind].real * model.Sb / model.Sn) / N[model.bus_ind]

    return x, success


if __name__ == "__main__":
    pr = True

    n_sc = 1
    n_off = 2
    n_wf = 5

    npr, models, M = nswph_models(n_sc, n_off, n_wf, Pwf=0.0)

    x0, success = initialize_state_vector(npr, models, pr=pr)

    model = models[0]
