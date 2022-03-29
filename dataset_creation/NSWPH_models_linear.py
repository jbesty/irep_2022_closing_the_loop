import numpy as np

from dataset_creation.NSWPH_models import ModelType


def compute_damping_ratio(eig_vals):
    sigma = np.real(eig_vals)
    omega = np.abs(np.imag(eig_vals))

    # Compute the damping ratio
    damping = np.divide(-sigma, np.sqrt(sigma ** 2 + omega ** 2),
                        out=np.zeros_like(sigma),
                        where=abs(sigma) != 0)

    # Index of the smallest
    smallest_damping_index = np.argmin(damping)

    return damping, smallest_damping_index


def converter_linear(xo, uo, cpr):
    Id = xo[cpr.x_idx['Id']]
    Iq = xo[cpr.x_idx['Iq']]
    theta = xo[cpr.x_idx['Theta']]
    Xpll = xo[cpr.x_idx['Xpll']]
    vd = uo[0]
    vq = uo[1]
    if len(uo) > 2:
        vdc = uo[2]
    else:
        vdc = 1

    # Ac = np.array([[cpr.wn*(-cpr.Kpc - cpr.Rt)/cpr.Lt, 0, cpr.Kic*cpr.wn/cpr.Lt, 0, cpr.wn/cpr.Lt, 0, 0, -cpr.Ki_pll*cpr.Kpc*cpr.Kpf*cpr.Kpp*cpr.wn/cpr.Lt, cpr.Kif*cpr.Kpc*cpr.Kpp*cpr.wn/cpr.Lt, cpr.Kip*cpr.Kpc*cpr.wn/cpr.Lt, 0, -cpr.Kpc*cpr.Kpp*cpr.wn/cpr.Lt, 0, 0],
    #     [0, cpr.wn*(-cpr.Kpc - cpr.Rt)/cpr.Lt, 0, cpr.Kic*cpr.wn/cpr.Lt, 0, cpr.wn/cpr.Lt, 0, 0, 0, 0, cpr.Kiq*cpr.Kpc*cpr.wn/cpr.Lt, 0, cpr.Kpc*cpr.Kpq*cpr.Kq*cpr.wn/cpr.Lt, cpr.Kpc*cpr.Kpq*cpr.Kv*cpr.wn/cpr.Lt],
    #     [-1, 0, 0, 0, 0, 0, 0, -cpr.Ki_pll*cpr.Kpf*cpr.Kpp, cpr.Kif*cpr.Kpp, cpr.Kip, 0, -cpr.Kpp, 0, 0],
    #     [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, cpr.Kiq, 0, cpr.Kpq*cpr.Kq, cpr.Kpq*cpr.Kv],
    #     [0, 0, 0, 0, -1/cpr.Tad, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, -1/cpr.Tad, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, cpr.Ki_pll*cpr.wn, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, -cpr.Ki_pll, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, -cpr.Ki_pll*cpr.Kpf, cpr.Kif, 0, 0, -1, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, cpr.Kq, cpr.Kv],
    #     [vd/cpr.Tpm, vq/cpr.Tpm, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1/cpr.Tpm, 0, 0],
    #     [vq/cpr.Tpm, -vd/cpr.Tpm, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1/cpr.Tpm, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1/cpr.Tvm],
    #     ])
    # Ac =  np.array([[cpr.wn*(-cpr.Kpc/vdc - cpr.Rt)/cpr.Lt, cpr.wn*(cpr.Lt*(Xpll*cpr.Ki_pll + cpr.Kp_pll*vq) - cpr.Lt*(Xpll*cpr.Ki_pll + cpr.Kp_pll*vq)/vdc)/cpr.Lt, 0, cpr.Kic*cpr.wn/(cpr.Lt*vdc), 0, cpr.wn/(cpr.Lt*vdc), 0, 0, cpr.wn*(Iq*cpr.Ki_pll*cpr.Lt + (-Iq*cpr.Ki_pll*cpr.Lt - cpr.Ki_pll*cpr.Kpc*cpr.Kpf*cpr.Kpp)/vdc)/cpr.Lt, cpr.Kif*cpr.Kpc*cpr.Kpp*cpr.wn/(cpr.Lt*vdc), cpr.Kip*cpr.Kpc*cpr.wn/(cpr.Lt*vdc), 0, -cpr.Kpc*cpr.Kpp*cpr.wn/(cpr.Lt*vdc), 0, 0],
    #             [cpr.wn*(-cpr.Lt*(Xpll*cpr.Ki_pll + cpr.Kp_pll*vq) + cpr.Lt*(Xpll*cpr.Ki_pll + cpr.Kp_pll*vq)/vdc)/cpr.Lt, cpr.wn*(-cpr.Kpc/vdc - cpr.Rt)/cpr.Lt, 0, 0, cpr.Kic*cpr.wn/(cpr.Lt*vdc), 0, cpr.wn/(cpr.Lt*vdc), 0, cpr.wn*(-Id*cpr.Ki_pll*cpr.Lt + Id*cpr.Ki_pll*cpr.Lt/vdc)/cpr.Lt, 0, 0, cpr.Kiq*cpr.Kpc*cpr.wn/(cpr.Lt*vdc), 0, cpr.Kpc*cpr.Kpq*cpr.Kq*cpr.wn/(cpr.Lt*vdc), cpr.Kpc*cpr.Kpq*cpr.Kv*cpr.wn/(cpr.Lt*vdc)],
    #             [cpr.wn*vd/vdc, cpr.wn*vq/vdc, -cpr.wn, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [-1, 0, 0, 0, 0, 0, 0, 0, -cpr.Ki_pll*cpr.Kpf*cpr.Kpp, cpr.Kif*cpr.Kpp, cpr.Kip, 0, -cpr.Kpp, 0, 0],
    #             [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, cpr.Kiq, 0, cpr.Kpq*cpr.Kq, cpr.Kpq*cpr.Kv],
    #             [0, 0, 0, 0, 0, -1/cpr.Tad, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, -1/cpr.Tad, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, cpr.Ki_pll*cpr.wn, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, -cpr.Ki_pll, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, -cpr.Ki_pll*cpr.Kpf, cpr.Kif, 0, 0, -1, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, cpr.Kq, cpr.Kv],
    #             [vd/cpr.Tpm, vq/cpr.Tpm, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1/cpr.Tpm, 0, 0],
    #             [vq/cpr.Tpm, -vd/cpr.Tpm, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1/cpr.Tpm, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1/cpr.Tvm],
    #             ])

    Ac = np.array([[cpr.wn * (-cpr.Kpc / vdc - cpr.Rt) / cpr.Lt, cpr.wn * (
            cpr.Lt * (Xpll * cpr.Ki_pll + cpr.Kp_pll * vq) - cpr.Lt * (
            Xpll * cpr.Ki_pll + cpr.Kp_pll * vq) / vdc) / cpr.Lt, 0, cpr.Kic * cpr.wn / (cpr.Lt * vdc), 0,
                    cpr.wn / (cpr.Lt * vdc), 0, 0, cpr.wn * (Iq * cpr.Ki_pll * cpr.Lt + (
                -Iq * cpr.Ki_pll * cpr.Lt - cpr.Ki_pll * cpr.Kpc * cpr.Kpf * cpr.Kpp) / vdc) / cpr.Lt,
                    cpr.Kif * cpr.Kpc * cpr.Kpp * cpr.wn / (cpr.Lt * vdc), cpr.Kip * cpr.Kpc * cpr.wn / (cpr.Lt * vdc),
                    0, -cpr.Kpc * cpr.Kpp * cpr.wn / (cpr.Lt * vdc), 0, 0],
                   [cpr.wn * (-cpr.Lt * (Xpll * cpr.Ki_pll + cpr.Kp_pll * vq) + cpr.Lt * (
                           Xpll * cpr.Ki_pll + cpr.Kp_pll * vq) / vdc) / cpr.Lt,
                    cpr.wn * (-cpr.Kpc / vdc - cpr.Rt) / cpr.Lt, 0, 0, cpr.Kic * cpr.wn / (cpr.Lt * vdc), 0,
                    cpr.wn / (cpr.Lt * vdc), 0,
                    cpr.wn * (-Id * cpr.Ki_pll * cpr.Lt + Id * cpr.Ki_pll * cpr.Lt / vdc) / cpr.Lt, 0, 0,
                    cpr.Kiq * cpr.Kpc * cpr.wn / (cpr.Lt * vdc), 0,
                    cpr.Kpc * cpr.Kpq * cpr.Kq * cpr.wn / (cpr.Lt * vdc),
                    cpr.Kpc * cpr.Kpq * cpr.Kv * cpr.wn / (cpr.Lt * vdc)],
                   [cpr.wn * vd / (cpr.Ldc * vdc), cpr.wn * vq / (cpr.Ldc * vdc), -cpr.wn / cpr.Ldc, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0],
                   [-1, 0, 0, 0, 0, 0, 0, 0, -cpr.Ki_pll * cpr.Kpf * cpr.Kpp, cpr.Kif * cpr.Kpp, cpr.Kip, 0, -cpr.Kpp,
                    0, 0],
                   [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, cpr.Kiq, 0, cpr.Kpq * cpr.Kq, cpr.Kpq * cpr.Kv],
                   [0, 0, 0, 0, 0, -1 / cpr.Tad, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, -1 / cpr.Tad, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, cpr.Ki_pll * cpr.wn, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, -cpr.Ki_pll, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, -cpr.Ki_pll * cpr.Kpf, cpr.Kif, 0, 0, -1, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, cpr.Kq, cpr.Kv],
                   [vd / cpr.Tpm, vq / cpr.Tpm, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1 / cpr.Tpm, 0, 0],
                   [vq / cpr.Tpm, -vd / cpr.Tpm, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1 / cpr.Tpm, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1 / cpr.Tvm],
                   ])
    # Bvc = np.array([[-cpr.wn/cpr.Lt, -cpr.Kp_pll*cpr.Kpc*cpr.Kpf*cpr.Kpp*cpr.wn/cpr.Lt],
    #     [0, -cpr.wn/cpr.Lt],
    #     [0, -cpr.Kp_pll*cpr.Kpf*cpr.Kpp],
    #     [0, 0],
    #     [1/cpr.Tad, 0],
    #     [0, 1/cpr.Tad],
    #     [0, cpr.Kp_pll*cpr.wn],
    #     [0, 1],
    #     [0, -cpr.Kp_pll],
    #     [0, -cpr.Kp_pll*cpr.Kpf],
    #     [0, 0],
    #     [Id/cpr.Tpm, Iq/cpr.Tpm],
    #     [-Iq/cpr.Tpm, Id/cpr.Tpm],
    #     [vd/(cpr.Tvm*np.sqrt(vd**2 + vq**2)), vq/(cpr.Tvm*np.sqrt(vd**2 + vq**2))],
    #     ])
    # Bvc = np.array([[-cpr.wn/cpr.Lt, cpr.wn*(Iq*cpr.Kp_pll*cpr.Lt + (-Iq*cpr.Kp_pll*cpr.Lt - cpr.Kp_pll*cpr.Kpc*cpr.Kpf*cpr.Kpp)/vdc)/cpr.Lt],
    #                 [0, cpr.wn*(-Id*cpr.Kp_pll*cpr.Lt + Id*cpr.Kp_pll*cpr.Lt/vdc - 1)/cpr.Lt],
    #                 [Id*cpr.wn/vdc, Iq*cpr.wn/vdc],
    #                 [0, -cpr.Kp_pll*cpr.Kpf*cpr.Kpp],
    #                 [0, 0],
    #                 [1/cpr.Tad, 0],
    #                 [0, 1/cpr.Tad],
    #                 [0, cpr.Kp_pll*cpr.wn],
    #                 [0, 1],
    #                 [0, -cpr.Kp_pll],
    #                 [0, -cpr.Kp_pll*cpr.Kpf],
    #                 [0, 0],
    #                 [Id/cpr.Tpm, Iq/cpr.Tpm],
    #                 [-Iq/cpr.Tpm, Id/cpr.Tpm],
    #                 [vd/(cpr.Tvm*np.sqrt(vd**2 + vq**2)), vq/(cpr.Tvm*np.sqrt(vd**2 + vq**2))],
    #                 ])

    Bvc = np.array([[-cpr.wn / cpr.Lt, cpr.wn * (Iq * cpr.Kp_pll * cpr.Lt + (
            -Iq * cpr.Kp_pll * cpr.Lt - cpr.Kp_pll * cpr.Kpc * cpr.Kpf * cpr.Kpp) / vdc) / cpr.Lt],
                    [0, cpr.wn * (-Id * cpr.Kp_pll * cpr.Lt + Id * cpr.Kp_pll * cpr.Lt / vdc - 1) / cpr.Lt],
                    [Id * cpr.wn / (cpr.Ldc * vdc), Iq * cpr.wn / (cpr.Ldc * vdc)],
                    [0, -cpr.Kp_pll * cpr.Kpf * cpr.Kpp],
                    [0, 0],
                    [1 / cpr.Tad, 0],
                    [0, 1 / cpr.Tad],
                    [0, cpr.Kp_pll * cpr.wn],
                    [0, 1],
                    [0, -cpr.Kp_pll],
                    [0, -cpr.Kp_pll * cpr.Kpf],
                    [0, 0],
                    [Id / cpr.Tpm, Iq / cpr.Tpm],
                    [-Iq / cpr.Tpm, Id / cpr.Tpm],
                    [vd / (cpr.Tvm * np.sqrt(vd ** 2 + vq ** 2)), vq / (cpr.Tvm * np.sqrt(vd ** 2 + vq ** 2))],
                    ])

    Tc = np.array([[np.cos(theta), np.sin(theta)],
                   [-np.sin(theta), np.cos(theta)]])

    Rvc = np.zeros((2, cpr.nx))
    Rvc[0, cpr.x_idx['Theta']] = vq
    Rvc[1, cpr.x_idx['Theta']] = -vd

    Pc = np.zeros((2, cpr.nx))
    Pc[0, cpr.x_idx['Id']] = 1
    Pc[1, cpr.x_idx['Iq']] = 1
    Pc[0, cpr.x_idx['Theta']] = -Iq
    Pc[1, cpr.x_idx['Theta']] = Id

    Cc = Tc.T @ Pc

    Dc = np.zeros((2, 2))

    A = Ac + Bvc @ Rvc
    B = Bvc @ Tc
    C = Cc * cpr.Sn / cpr.Sb
    D = Dc

    return A, B, C, D


def c_linear0(x0, u0, cpr):
    Id = x0[cpr.x_idx['Id']]
    Iq = x0[cpr.x_idx['Iq']]
    theta = x0[cpr.x_idx['Theta']]
    vd = u0[0]
    vq = u0[1]

    Ac0 = np.array([[cpr.wn * (-cpr.Kpc - cpr.Rt) / cpr.Lt, 0, cpr.Kic * cpr.wn / cpr.Lt, 0, cpr.wn / cpr.Lt, 0, 0,
                     -cpr.Ki_pll * cpr.Kpc * cpr.Kpf * cpr.Kpp * cpr.wn / cpr.Lt,
                     cpr.Kif * cpr.Kpc * cpr.Kpp * cpr.wn / cpr.Lt, cpr.Kip * cpr.Kpc * cpr.wn / cpr.Lt, 0,
                     -cpr.Kpc * cpr.Kpp * cpr.wn / cpr.Lt, 0, 0],
                    [0, cpr.wn * (-cpr.Kpc - cpr.Rt) / cpr.Lt, 0, cpr.Kic * cpr.wn / cpr.Lt, 0, cpr.wn / cpr.Lt, 0, 0,
                     0, 0, cpr.Kiq * cpr.Kpc * cpr.wn / cpr.Lt, 0, cpr.Kpc * cpr.Kpq * cpr.Kq * cpr.wn / cpr.Lt,
                     cpr.Kpc * cpr.Kpq * cpr.Kv * cpr.wn / cpr.Lt],
                    [-1, 0, 0, 0, 0, 0, 0, -cpr.Ki_pll * cpr.Kpf * cpr.Kpp, cpr.Kif * cpr.Kpp, cpr.Kip, 0, -cpr.Kpp, 0,
                     0],
                    [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, cpr.Kiq, 0, cpr.Kpq * cpr.Kq, cpr.Kpq * cpr.Kv],
                    [0, 0, 0, 0, -1 / cpr.Tad, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, -1 / cpr.Tad, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, cpr.Ki_pll * cpr.wn, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, -cpr.Ki_pll, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, -cpr.Ki_pll * cpr.Kpf, cpr.Kif, 0, 0, -1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, cpr.Kq, cpr.Kv],
                    [vd / cpr.Tpm, vq / cpr.Tpm, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1 / cpr.Tpm, 0, 0],
                    [vq / cpr.Tpm, -vd / cpr.Tpm, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1 / cpr.Tpm, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1 / cpr.Tvm],
                    ])

    Bvc0 = np.array([[-cpr.wn / cpr.Lt, -cpr.Kp_pll * cpr.Kpc * cpr.Kpf * cpr.Kpp * cpr.wn / cpr.Lt],
                     [0, -cpr.wn / cpr.Lt],
                     [0, -cpr.Kp_pll * cpr.Kpf * cpr.Kpp],
                     [0, 0],
                     [1 / cpr.Tad, 0],
                     [0, 1 / cpr.Tad],
                     [0, cpr.Kp_pll * cpr.wn],
                     [0, 1],
                     [0, -cpr.Kp_pll],
                     [0, -cpr.Kp_pll * cpr.Kpf],
                     [0, 0],
                     [Id / cpr.Tpm, Iq / cpr.Tpm],
                     [-Iq / cpr.Tpm, Id / cpr.Tpm],
                     [vd / (cpr.Tvm * np.sqrt(vd ** 2 + vq ** 2)), vq / (cpr.Tvm * np.sqrt(vd ** 2 + vq ** 2))],
                     ])

    Tc0 = np.array([[np.cos(theta), np.sin(theta)],
                    [-np.sin(theta), np.cos(theta)]])

    Rvc0 = np.zeros((2, cpr.nx))
    Rvc0[0, cpr.x_idx['Theta']] = vq
    Rvc0[1, cpr.x_idx['Theta']] = -vd

    Pc0 = np.zeros((2, cpr.nx))
    Pc0[0, cpr.x_idx['Id']] = 1
    Pc0[1, cpr.x_idx['Iq']] = 1
    Pc0[0, cpr.x_idx['Theta']] = -Iq
    Pc0[1, cpr.x_idx['Theta']] = Id

    Cc0 = Tc0.T @ Pc0

    Dc0 = np.zeros((2, 2))

    A0 = Ac0 + Bvc0 @ Rvc0
    B0 = Bvc0 @ Tc0
    C0 = Cc0 * cpr.Sn / cpr.Sb
    D0 = Dc0

    return A0, B0, C0, D0


def vsc_linear_1(xo, uo, cpr):
    Id = xo[cpr.x_idx['Id']]
    Iq = xo[cpr.x_idx['Iq']]
    theta = xo[cpr.x_idx['Theta']]
    Xpll = xo[cpr.x_idx['Xpll']]
    Madd = xo[cpr.x_idx['Madd']]
    Madq = xo[cpr.x_idx['Madq']]
    Md = xo[cpr.x_idx['Md']]
    Mq = xo[cpr.x_idx['Mq']]
    Xp = xo[cpr.x_idx['Xp']]
    Xq = xo[cpr.x_idx['Xq']]
    Xf = xo[cpr.x_idx['Xf']]
    Pm = xo[cpr.x_idx['Pm']]
    Qm = xo[cpr.x_idx['Qm']]
    Vm = xo[cpr.x_idx['Vm']]
    vd = uo[0]
    vq = uo[1]
    vdc = uo[2]

    # Ac = np.array([[cpr.wn*(-cpr.Kpc/vdc - cpr.Rt)/cpr.Lt, cpr.wn*(cpr.Lt*(Xpll*cpr.Ki_pll + cpr.Kp_pll*vq) - cpr.Lt*(Xpll*cpr.Ki_pll + cpr.Kp_pll*vq)/vdc)/cpr.Lt, cpr.Kic*cpr.wn/(cpr.Lt*vdc), 0, cpr.wn/(cpr.Lt*vdc), 0, 0, cpr.wn*(Iq*cpr.Ki_pll*cpr.Lt + (-Iq*cpr.Ki_pll*cpr.Lt - cpr.Ki_pll*cpr.Kpc*cpr.Kpf*cpr.Kpp)/vdc)/cpr.Lt, cpr.Kif*cpr.Kpc*cpr.Kpp*cpr.wn/(cpr.Lt*vdc), cpr.Kip*cpr.Kpc*cpr.wn/(cpr.Lt*vdc), 0, -cpr.Kpc*cpr.Kpp*cpr.wn/(cpr.Lt*vdc), 0, 0],
    #             [cpr.wn*(-cpr.Lt*(Xpll*cpr.Ki_pll + cpr.Kp_pll*vq) + cpr.Lt*(Xpll*cpr.Ki_pll + cpr.Kp_pll*vq)/vdc)/cpr.Lt, cpr.wn*(-cpr.Kpc/vdc - cpr.Rt)/cpr.Lt, 0, cpr.Kic*cpr.wn/(cpr.Lt*vdc), 0, cpr.wn/(cpr.Lt*vdc), 0, cpr.wn*(-Id*cpr.Ki_pll*cpr.Lt + Id*cpr.Ki_pll*cpr.Lt/vdc)/cpr.Lt, 0, 0, cpr.Kiq*cpr.Kpc*cpr.wn/(cpr.Lt*vdc), 0, cpr.Kpc*cpr.Kpq*cpr.Kq*cpr.wn/(cpr.Lt*vdc), cpr.Kpc*cpr.Kpq*cpr.Kv*cpr.wn/(cpr.Lt*vdc)],
    #             [-1, 0, 0, 0, 0, 0, 0, -cpr.Ki_pll*cpr.Kpf*cpr.Kpp, cpr.Kif*cpr.Kpp, cpr.Kip, 0, -cpr.Kpp, 0, 0],
    #             [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, cpr.Kiq, 0, cpr.Kpq*cpr.Kq, cpr.Kpq*cpr.Kv],
    #             [0, 0, 0, 0, -1/cpr.Tad, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, -1/cpr.Tad, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, cpr.Ki_pll*cpr.wn, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, -cpr.Ki_pll, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, -cpr.Ki_pll*cpr.Kpf, cpr.Kif, 0, 0, -1, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, cpr.Kq, cpr.Kv],
    #             [vd/cpr.Tpm, vq/cpr.Tpm, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1/cpr.Tpm, 0, 0],
    #             [vq/cpr.Tpm, -vd/cpr.Tpm, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1/cpr.Tpm, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1/cpr.Tvm],
    #             ])
    Ac = np.array([[cpr.wn * (-cpr.Kpc / vdc - cpr.Rt) / cpr.Lt, cpr.wn * (
            cpr.Lt * (Xpll * cpr.Ki_pll + cpr.Kp_pll * vq) - cpr.Lt * (
            Xpll * cpr.Ki_pll + cpr.Kp_pll * vq) / vdc) / cpr.Lt, 0, cpr.Kic * cpr.wn / (cpr.Lt * vdc), 0,
                    cpr.wn / (cpr.Lt * vdc), 0, 0, cpr.wn * (Iq * cpr.Ki_pll * cpr.Lt + (
                -Iq * cpr.Ki_pll * cpr.Lt - cpr.Ki_pll * cpr.Kpc * cpr.Kpf * cpr.Kpp) / vdc) / cpr.Lt,
                    cpr.Kif * cpr.Kpc * cpr.Kpp * cpr.wn / (cpr.Lt * vdc), cpr.Kip * cpr.Kpc * cpr.wn / (cpr.Lt * vdc),
                    0, -cpr.Kpc * cpr.Kpp * cpr.wn / (cpr.Lt * vdc), 0, 0],
                   [cpr.wn * (-cpr.Lt * (Xpll * cpr.Ki_pll + cpr.Kp_pll * vq) + cpr.Lt * (
                           Xpll * cpr.Ki_pll + cpr.Kp_pll * vq) / vdc) / cpr.Lt,
                    cpr.wn * (-cpr.Kpc / vdc - cpr.Rt) / cpr.Lt, 0, 0, cpr.Kic * cpr.wn / (cpr.Lt * vdc), 0,
                    cpr.wn / (cpr.Lt * vdc), 0,
                    cpr.wn * (-Id * cpr.Ki_pll * cpr.Lt + Id * cpr.Ki_pll * cpr.Lt / vdc) / cpr.Lt, 0, 0,
                    cpr.Kiq * cpr.Kpc * cpr.wn / (cpr.Lt * vdc), 0,
                    cpr.Kpc * cpr.Kpq * cpr.Kq * cpr.wn / (cpr.Lt * vdc),
                    cpr.Kpc * cpr.Kpq * cpr.Kv * cpr.wn / (cpr.Lt * vdc)],
                   [cpr.wn * vd / (cpr.Ldc * vdc), cpr.wn * vq / (cpr.Ldc * vdc), -cpr.wn / cpr.Ldc, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0],
                   [-1, 0, 0, 0, 0, 0, 0, 0, -cpr.Ki_pll * cpr.Kpf * cpr.Kpp, cpr.Kif * cpr.Kpp, cpr.Kip, 0, -cpr.Kpp,
                    0, 0],
                   [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, cpr.Kiq, 0, cpr.Kpq * cpr.Kq, cpr.Kpq * cpr.Kv],
                   [0, 0, 0, 0, 0, -1 / cpr.Tad, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, -1 / cpr.Tad, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, cpr.Ki_pll * cpr.wn, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, -cpr.Ki_pll, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, -cpr.Ki_pll * cpr.Kpf, cpr.Kif, 0, 0, -1, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, cpr.Kq, cpr.Kv],
                   [vd / cpr.Tpm, vq / cpr.Tpm, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1 / cpr.Tpm, 0, 0],
                   [vq / cpr.Tpm, -vd / cpr.Tpm, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1 / cpr.Tpm, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1 / cpr.Tvm],
                   ])
    # Bvc = np.array([[-cpr.wn/cpr.Lt, cpr.wn*(Iq*cpr.Kp_pll*cpr.Lt + (-Iq*cpr.Kp_pll*cpr.Lt - cpr.Kp_pll*cpr.Kpc*cpr.Kpf*cpr.Kpp)/vdc)/cpr.Lt, -cpr.wn*(-Iq*cpr.Lt*(Xpll*cpr.Ki_pll + cpr.Kp_pll*vq) + Madd + Md*cpr.Kic + cpr.Kpc*(-Id + Xp*cpr.Kip + cpr.Kpp*(-Pm + Xf*cpr.Kif + cpr.Kpf*(-Xpll*cpr.Ki_pll - cpr.Kp_pll*vq + 1) + cpr.Pref)))/(cpr.Lt*vdc**2)],
    #                 [0, cpr.wn*(-Id*cpr.Kp_pll*cpr.Lt + Id*cpr.Kp_pll*cpr.Lt/vdc - 1)/cpr.Lt, -cpr.wn*(Id*cpr.Lt*(Xpll*cpr.Ki_pll + cpr.Kp_pll*vq) + Madq + Mq*cpr.Kic + cpr.Kpc*(-Iq + Xq*cpr.Kiq + cpr.Kpq*(cpr.Kq*(Qm - cpr.Qref) + cpr.Kv*(Vm - cpr.Vref))))/(cpr.Lt*vdc**2)],
    #                 [0, -cpr.Kp_pll*cpr.Kpf*cpr.Kpp, 0],
    #                 [0, 0, 0],
    #                 [1/cpr.Tad, 0, 0],
    #                 [0, 1/cpr.Tad, 0],
    #                 [0, cpr.Kp_pll*cpr.wn, 0],
    #                 [0, 1, 0],
    #                 [0, -cpr.Kp_pll, 0],
    #                 [0, -cpr.Kp_pll*cpr.Kpf, 0],
    #                 [0, 0, 0],
    #                 [Id/cpr.Tpm, Iq/cpr.Tpm, 0],
    #                 [-Iq/cpr.Tpm, Id/cpr.Tpm, 0],
    #                 [vd/(cpr.Tvm*np.sqrt(vd**2 + vq**2)), vq/(cpr.Tvm*np.sqrt(vd**2 + vq**2)), 0],
    #                 ])
    Bvac = np.array([[-cpr.wn / cpr.Lt, cpr.wn * (Iq * cpr.Kp_pll * cpr.Lt + (
            -Iq * cpr.Kp_pll * cpr.Lt - cpr.Kp_pll * cpr.Kpc * cpr.Kpf * cpr.Kpp) / vdc) / cpr.Lt],
                     [0, cpr.wn * (-Id * cpr.Kp_pll * cpr.Lt + Id * cpr.Kp_pll * cpr.Lt / vdc - 1) / cpr.Lt],
                     [Id * cpr.wn / (cpr.Ldc * vdc), Iq * cpr.wn / (cpr.Ldc * vdc)],
                     [0, -cpr.Kp_pll * cpr.Kpf * cpr.Kpp],
                     [0, 0],
                     [1 / cpr.Tad, 0],
                     [0, 1 / cpr.Tad],
                     [0, cpr.Kp_pll * cpr.wn],
                     [0, 1],
                     [0, -cpr.Kp_pll],
                     [0, -cpr.Kp_pll * cpr.Kpf],
                     [0, 0],
                     [Id / cpr.Tpm, Iq / cpr.Tpm],
                     [-Iq / cpr.Tpm, Id / cpr.Tpm],
                     [vd / (cpr.Tvm * np.sqrt(vd ** 2 + vq ** 2)), vq / (cpr.Tvm * np.sqrt(vd ** 2 + vq ** 2))],
                     ])

    Bvdc = np.array([[-cpr.wn * (
            -Iq * cpr.Lt * (Xpll * cpr.Ki_pll + cpr.Kp_pll * vq) + Madd + Md * cpr.Kic + cpr.Kpc * (
            -Id + Xp * cpr.Kip + cpr.Kpp * (
            -Pm + Xf * cpr.Kif + cpr.Kpf * (-Xpll * cpr.Ki_pll - cpr.Kp_pll * vq + 1) + cpr.Pref))) / (
                              cpr.Lt * vdc ** 2)],
                     [-cpr.wn * (Id * cpr.Lt * (Xpll * cpr.Ki_pll + cpr.Kp_pll * vq) + Madq + Mq * cpr.Kic + cpr.Kpc * (
                             -Iq + Xq * cpr.Kiq + cpr.Kpq * (
                             cpr.Kq * (Qm - cpr.Qref) + cpr.Kv * (Vm - cpr.Vref)))) / (cpr.Lt * vdc ** 2)],
                     [-cpr.wn * (Id * vd + Iq * vq) / (cpr.Ldc * vdc ** 2)],
                     [0],
                     [0],
                     [0],
                     [0],
                     [0],
                     [0],
                     [0],
                     [0],
                     [0],
                     [0],
                     [0],
                     [0],
                     ])

    Tc = np.array([[np.cos(theta), np.sin(theta)],
                   [-np.sin(theta), np.cos(theta)]])

    Rvc = np.zeros((2, cpr.nx))
    Rvc[0, cpr.x_idx['Theta']] = vq
    Rvc[1, cpr.x_idx['Theta']] = -vd

    Pc = np.zeros((2, cpr.nx))
    Pc[0, cpr.x_idx['Id']] = 1
    Pc[1, cpr.x_idx['Iq']] = 1
    Pc[0, cpr.x_idx['Theta']] = -Iq
    Pc[1, cpr.x_idx['Theta']] = Id

    Cc = Tc.T @ Pc
    Cdc = np.zeros((1, cpr.nx))
    Cdc[0, cpr.x_idx['Idc']] = 1
    Dc = np.zeros((2, 2))

    A = Ac + Bvac @ Rvc
    B = Bvac @ Tc
    C = Cc * cpr.Sn / cpr.Sb
    D = Dc

    return A, B, Bvdc, C, Cdc, D


def vsc_linear_2(x0, u0, cpr):
    Id = x0[cpr.x_idx['Id']]
    Iq = x0[cpr.x_idx['Iq']]
    theta = x0[cpr.x_idx['Theta']]
    Xpll = x0[cpr.x_idx['Xpll']]
    Madd = x0[cpr.x_idx['Madd']]
    Madq = x0[cpr.x_idx['Madq']]
    Md = x0[cpr.x_idx['Md']]
    Mq = x0[cpr.x_idx['Mq']]
    Xp = x0[cpr.x_idx['Xp']]
    Xq = x0[cpr.x_idx['Xq']]
    Qm = x0[cpr.x_idx['Qm']]
    Vm = x0[cpr.x_idx['Vm']]
    vd = u0[0]
    vq = u0[1]
    vdc = u0[2]

    # Ac = np.array([[cpr.wn*(-cpr.Kpc/vdc - cpr.Rt)/cpr.Lt, cpr.wn*(cpr.Lt*(Xpll*cpr.Ki_pll + cpr.Kp_pll*vq) - cpr.Lt*(Xpll*cpr.Ki_pll + cpr.Kp_pll*vq)/vdc)/cpr.Lt, cpr.Kic*cpr.wn/(cpr.Lt*vdc), 0, cpr.wn/(cpr.Lt*vdc), 0, 0, cpr.wn*(Iq*cpr.Ki_pll*cpr.Lt - Iq*cpr.Ki_pll*cpr.Lt/vdc)/cpr.Lt, 0, cpr.Kip*cpr.Kpc*cpr.wn/(cpr.Lt*vdc), 0, 0, 0, cpr.Kpc*cpr.Kpp*cpr.wn/(cpr.Lt*cpr.Vref*vdc)],
    #                 [cpr.wn*(-cpr.Lt*(Xpll*cpr.Ki_pll + cpr.Kp_pll*vq) + cpr.Lt*(Xpll*cpr.Ki_pll + cpr.Kp_pll*vq)/vdc)/cpr.Lt, cpr.wn*(-cpr.Kpc/vdc - cpr.Rt)/cpr.Lt, 0, cpr.Kic*cpr.wn/(cpr.Lt*vdc), 0, cpr.wn/(cpr.Lt*vdc), 0, cpr.wn*(-Id*cpr.Ki_pll*cpr.Lt + Id*cpr.Ki_pll*cpr.Lt/vdc)/cpr.Lt, 0, 0, cpr.Kiq*cpr.Kpc*cpr.wn/(cpr.Lt*vdc), 0, cpr.Kpc*cpr.Kpq*cpr.wn/(cpr.Lt*vdc), 0],
    #                 [-1, 0, 0, 0, 0, 0, 0, 0, 0, cpr.Kip, 0, 0, 0, cpr.Kpp/cpr.Vref],
    #                 [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, cpr.Kiq, 0, cpr.Kpq, 0],
    #                 [0, 0, 0, 0, -1/cpr.Tad, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                 [0, 0, 0, 0, 0, -1/cpr.Tad, 0, 0, 0, 0, 0, 0, 0, 0],
    #                 [0, 0, 0, 0, 0, 0, 0, cpr.Ki_pll*cpr.wn, 0, 0, 0, 0, 0, 0],
    #                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                 [0, 0, 0, 0, 0, 0, 0, -cpr.Ki_pll, 0, 0, 0, 0, 0, 0],
    #                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1/cpr.Vref],
    #                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    #                 [vd/cpr.Tpm, vq/cpr.Tpm, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1/cpr.Tpm, 0, 0],
    #                 [vq/cpr.Tpm, -vd/cpr.Tpm, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1/cpr.Tpm, 0],
    #                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1/cpr.Tvm],
    #                 ])
    Ac = np.array([[cpr.wn * (-cpr.Kpc / vdc - cpr.Rt) / cpr.Lt, cpr.wn * (
            cpr.Lt * (Xpll * cpr.Ki_pll + cpr.Kp_pll * vq) - cpr.Lt * (
            Xpll * cpr.Ki_pll + cpr.Kp_pll * vq) / vdc) / cpr.Lt, 0, cpr.Kic * cpr.wn / (cpr.Lt * vdc), 0,
                    cpr.wn / (cpr.Lt * vdc), 0, 0,
                    cpr.wn * (Iq * cpr.Ki_pll * cpr.Lt - Iq * cpr.Ki_pll * cpr.Lt / vdc) / cpr.Lt, 0,
                    cpr.Kip * cpr.Kpc * cpr.wn / (cpr.Lt * vdc), 0, 0, 0,
                    cpr.Kpc * cpr.Kpp * cpr.wn / (cpr.Lt * cpr.Vref * vdc)],
                   [cpr.wn * (-cpr.Lt * (Xpll * cpr.Ki_pll + cpr.Kp_pll * vq) + cpr.Lt * (
                           Xpll * cpr.Ki_pll + cpr.Kp_pll * vq) / vdc) / cpr.Lt,
                    cpr.wn * (-cpr.Kpc / vdc - cpr.Rt) / cpr.Lt, 0, 0, cpr.Kic * cpr.wn / (cpr.Lt * vdc), 0,
                    cpr.wn / (cpr.Lt * vdc), 0,
                    cpr.wn * (-Id * cpr.Ki_pll * cpr.Lt + Id * cpr.Ki_pll * cpr.Lt / vdc) / cpr.Lt, 0, 0,
                    cpr.Kiq * cpr.Kpc * cpr.wn / (cpr.Lt * vdc), 0, cpr.Kpc * cpr.Kpq * cpr.wn / (cpr.Lt * vdc), 0],
                   [cpr.wn * vd / (cpr.Ldc * vdc), cpr.wn * vq / (cpr.Ldc * vdc), -cpr.wn / cpr.Ldc, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0],
                   [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, cpr.Kip, 0, 0, 0, cpr.Kpp / cpr.Vref],
                   [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, cpr.Kiq, 0, cpr.Kpq, 0],
                   [0, 0, 0, 0, 0, -1 / cpr.Tad, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, -1 / cpr.Tad, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, cpr.Ki_pll * cpr.wn, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, -cpr.Ki_pll, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 / cpr.Vref],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                   [vd / cpr.Tpm, vq / cpr.Tpm, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1 / cpr.Tpm, 0, 0],
                   [vq / cpr.Tpm, -vd / cpr.Tpm, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1 / cpr.Tpm, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1 / cpr.Tvm],
                   ])

    # Bvc = np.array([[-cpr.wn/cpr.Lt, cpr.wn*(Iq*cpr.Kp_pll*cpr.Lt - Iq*cpr.Kp_pll*cpr.Lt/vdc)/cpr.Lt, -cpr.wn*(-Iq*cpr.Lt*(Xpll*cpr.Ki_pll + cpr.Kp_pll*vq) + Madd + Md*cpr.Kic + cpr.Kpc*(-Id + Xp*cpr.Kip + cpr.Kpp*(Vm/cpr.Vref - 1)))/(cpr.Lt*vdc**2)],
    #                 [0, cpr.wn*(-Id*cpr.Kp_pll*cpr.Lt + Id*cpr.Kp_pll*cpr.Lt/vdc - 1)/cpr.Lt, -cpr.wn*(Id*cpr.Lt*(Xpll*cpr.Ki_pll + cpr.Kp_pll*vq) + Madq + Mq*cpr.Kic + cpr.Kpc*(-Iq + Xq*cpr.Kiq + cpr.Kpq*(Qm - cpr.Qref)))/(cpr.Lt*vdc**2)],
    #                 [0, 0, 0],
    #                 [0, 0, 0],
    #                 [1/cpr.Tad, 0, 0],
    #                 [0, 1/cpr.Tad, 0],
    #                 [0, cpr.Kp_pll*cpr.wn, 0],
    #                 [0, 1, 0],
    #                 [0, -cpr.Kp_pll, 0],
    #                 [0, 0, 0],
    #                 [0, 0, 0],
    #                 [Id/cpr.Tpm, Iq/cpr.Tpm, 0],
    #                 [-Iq/cpr.Tpm, Id/cpr.Tpm, 0],
    #                 [0, 0, 1/cpr.Tvm],
    #                 ])
    Bvac = np.array([[-cpr.wn / cpr.Lt, cpr.wn * (Iq * cpr.Kp_pll * cpr.Lt - Iq * cpr.Kp_pll * cpr.Lt / vdc) / cpr.Lt],
                     [0, cpr.wn * (-Id * cpr.Kp_pll * cpr.Lt + Id * cpr.Kp_pll * cpr.Lt / vdc - 1) / cpr.Lt],
                     [Id * cpr.wn / (cpr.Ldc * vdc), Iq * cpr.wn / (cpr.Ldc * vdc)],
                     [0, 0],
                     [0, 0],
                     [1 / cpr.Tad, 0],
                     [0, 1 / cpr.Tad],
                     [0, cpr.Kp_pll * cpr.wn],
                     [0, 1],
                     [0, -cpr.Kp_pll],
                     [0, 0],
                     [0, 0],
                     [Id / cpr.Tpm, Iq / cpr.Tpm],
                     [-Iq / cpr.Tpm, Id / cpr.Tpm],
                     [0, 0],
                     ])
    Bvdc = np.array([[-cpr.wn * (
            -Iq * cpr.Lt * (Xpll * cpr.Ki_pll + cpr.Kp_pll * vq) + Madd + Md * cpr.Kic + cpr.Kpc * (
            -Id + Xp * cpr.Kip + cpr.Kpp * (Vm / cpr.Vref - 1))) / (cpr.Lt * vdc ** 2)],
                     [-cpr.wn * (Id * cpr.Lt * (Xpll * cpr.Ki_pll + cpr.Kp_pll * vq) + Madq + Mq * cpr.Kic + cpr.Kpc * (
                             -Iq + Xq * cpr.Kiq + cpr.Kpq * (Qm - cpr.Qref))) / (cpr.Lt * vdc ** 2)],
                     [-cpr.wn * (Id * vd + Iq * vq) / (cpr.Ldc * vdc ** 2)],
                     [0],
                     [0],
                     [0],
                     [0],
                     [0],
                     [0],
                     [0],
                     [0],
                     [0],
                     [0],
                     [0],
                     [1 / cpr.Tvm],
                     ])

    Tc = np.array([[np.cos(theta), np.sin(theta)],
                   [-np.sin(theta), np.cos(theta)]])

    Rvc = np.zeros((2, cpr.nx))
    Rvc[0, cpr.x_idx['Theta']] = vq
    Rvc[1, cpr.x_idx['Theta']] = -vd

    Pc = np.zeros((2, cpr.nx))
    Pc[0, cpr.x_idx['Id']] = 1
    Pc[1, cpr.x_idx['Iq']] = 1
    Pc[0, cpr.x_idx['Theta']] = -Iq
    Pc[1, cpr.x_idx['Theta']] = Id

    Cc = Tc.T @ Pc
    Cdc = np.zeros((1, cpr.nx))
    Cdc[0, cpr.x_idx['Idc']] = 1
    Dc = np.zeros((2, 2))

    A = Ac + Bvac @ Rvc
    B = Bvac @ Tc
    C = Cc * cpr.Sn / cpr.Sb
    D = Dc

    return A, B, Bvdc, C, Cdc, D


def dc_cable_linear(x0, u0, lpr):
    A = np.array([[-lpr.R * lpr.wn / lpr.L, lpr.wn / lpr.L, -lpr.wn / lpr.L],
                  [-2 * lpr.wn / lpr.C, -lpr.G * lpr.wn / lpr.C, 0],
                  [2 * lpr.wn / lpr.C, 0, -lpr.G * lpr.wn / lpr.C],
                  ])
    B = np.array([[0, 0],
                  [2 * lpr.wn / lpr.C, 0],
                  [0, -2 * lpr.wn / lpr.C],
                  ])
    C = np.zeros((1, lpr.nx))
    D = np.zeros((1, 2))

    return A, B, C, D


def vs_linear(x0, u0, vspr):
    phi = x0[vspr.x_idx['phi']]
    Ix = x0[vspr.x_idx['Ix']]
    Iy = x0[vspr.x_idx['Iy']]
    Vx = u0[0]
    Vy = u0[1]
    Ex = vspr.V0 * np.cos(phi)
    Ey = vspr.V0 * np.sin(phi)
    # A = np.array([[-1 + (Ex/(-Ix*vspr.L - Ix*vspr.R - Iy*vspr.R - Vy + Ex + vspr.Vy) + vspr.V0*(Ix*vspr.R - Iy*vspr.L - Iy*vspr.R + Vx + Ey - vspr.Vx)*np.sin(phi)/(-Ix*vspr.L - Ix*vspr.R - Iy*vspr.R - Vy + Ex + vspr.Vy)**2)/(1 + (Ix*vspr.R - Iy*vspr.L - Iy*vspr.R + Vx + Ey - vspr.Vx)**2/(-Ix*vspr.L - Ix*vspr.R - Iy*vspr.R - Vy + Ex + vspr.Vy)**2), (vspr.R/(-Ix*vspr.L - Ix*vspr.R - Iy*vspr.R - Vy + vspr.V0*np.cos(phi) + vspr.Vy) + (vspr.L + vspr.R)*(Ix*vspr.R - Iy*vspr.L - Iy*vspr.R + Vx + Ey - vspr.Vx)/(-Ix*vspr.L - Ix*vspr.R - Iy*vspr.R - Vy + vspr.V0*np.cos(phi) + vspr.Vy)**2)/(1 + (Ix*vspr.R - Iy*vspr.L - Iy*vspr.R + Vx + Ey - vspr.Vx)**2/(-Ix*vspr.L - Ix*vspr.R - Iy*vspr.R - Vy + vspr.V0*np.cos(phi) + vspr.Vy)**2), (vspr.R*(Ix*vspr.R - Iy*vspr.L - Iy*vspr.R + Vx + Ey - vspr.Vx)/(-Ix*vspr.L - Ix*vspr.R - Iy*vspr.R - Vy + vspr.V0*np.cos(phi) + vspr.Vy)**2 + (-vspr.L - vspr.R)/(-Ix*vspr.L - Ix*vspr.R - Iy*vspr.R - Vy + vspr.V0*np.cos(phi) + vspr.Vy))/(1 + (Ix*vspr.R - Iy*vspr.L - Iy*vspr.R + Vx + Ey - vspr.Vx)**2/(-Ix*vspr.L - Ix*vspr.R - Iy*vspr.R - Vy + vspr.V0*np.cos(phi) + vspr.Vy)**2)],
    #             [0, -vspr.R*vspr.wn/vspr.L, vspr.wn],
    #             [0, -vspr.wn, -vspr.R*vspr.wn/vspr.L],
    #             ])
    A = np.array([[-1 + ((Ey - Ex) * (Ix * vspr.R - Iy * vspr.R - Iy * vspr.X + Vx + Ey - Ex) / (
            -Ix * vspr.R - Ix * vspr.X - Iy * vspr.R - Vy + Ey + Ex) ** 2 + (Ey + Ex) / (
                                 -Ix * vspr.R - Ix * vspr.X - Iy * vspr.R - Vy + Ey + Ex)) / (
                           1 + (Ix * vspr.R - Iy * vspr.R - Iy * vspr.X + Vx + Ey - Ex) ** 2 / (
                           -Ix * vspr.R - Ix * vspr.X - Iy * vspr.R - Vy + Ey + Ex) ** 2), (
                           vspr.R / (-Ix * vspr.R - Ix * vspr.X - Iy * vspr.R - Vy + Ey + Ex) + (
                           vspr.R + vspr.X) * (Ix * vspr.R - Iy * vspr.R - Iy * vspr.X + Vx + Ey - Ex) / (
                                   -Ix * vspr.R - Ix * vspr.X - Iy * vspr.R - Vy + Ey + Ex) ** 2) / (
                           1 + (Ix * vspr.R - Iy * vspr.R - Iy * vspr.X + Vx + Ey - Ex) ** 2 / (
                           -Ix * vspr.R - Ix * vspr.X - Iy * vspr.R - Vy + Ey + Ex) ** 2), (
                           vspr.R * (Ix * vspr.R - Iy * vspr.R - Iy * vspr.X + Vx + Ey - Ex) / (
                           -Ix * vspr.R - Ix * vspr.X - Iy * vspr.R - Vy + Ey + Ex) ** 2 + (
                                   -vspr.R - vspr.X) / (
                                   -Ix * vspr.R - Ix * vspr.X - Iy * vspr.R - Vy + Ey + Ex)) / (
                           1 + (Ix * vspr.R - Iy * vspr.R - Iy * vspr.X + Vx + Ey - Ex) ** 2 / (
                           -Ix * vspr.R - Ix * vspr.X - Iy * vspr.R - Vy + Ey + Ex) ** 2)],
                  [-Ey * vspr.wn / vspr.X, -vspr.R * vspr.wn / vspr.X, vspr.wn],
                  [Ex * vspr.wn / vspr.X, -vspr.wn, -vspr.R * vspr.wn / vspr.X],
                  ])

    # B = np.array([[1/((1 + (Ix*vspr.R - Iy*vspr.L - Iy*vspr.R + Vx + Ey - vspr.Vx)**2/(-Ix*vspr.L - Ix*vspr.R - Iy*vspr.R - Vy + vspr.V0*np.cos(phi) + vspr.Vy)**2)*(-Ix*vspr.L - Ix*vspr.R - Iy*vspr.R - Vy + vspr.V0*np.cos(phi) + vspr.Vy)), (Ix*vspr.R - Iy*vspr.L - Iy*vspr.R + Vx + Ey - vspr.Vx)/((1 + (Ix*vspr.R - Iy*vspr.L - Iy*vspr.R + Vx + Ey - vspr.Vx)**2/(-Ix*vspr.L - Ix*vspr.R - Iy*vspr.R - Vy + vspr.V0*np.cos(phi) + vspr.Vy)**2)*(-Ix*vspr.L - Ix*vspr.R - Iy*vspr.R - Vy + vspr.V0*np.cos(phi) + vspr.Vy)**2)],
    #     [-vspr.wn/vspr.L, 0],
    #     [0, -vspr.wn/vspr.L],
    #     ])
    B = np.array([[1 / ((1 + (Ix * vspr.R - Iy * vspr.R - Iy * vspr.X + Vx + Ey - Ex) ** 2 / (
            -Ix * vspr.R - Ix * vspr.X - Iy * vspr.R - Vy + Ey + Ex) ** 2) * (
                                -Ix * vspr.R - Ix * vspr.X - Iy * vspr.R - Vy + Ey + Ex)),
                   (Ix * vspr.R - Iy * vspr.R - Iy * vspr.X + Vx + Ey - Ex) / ((1 + (
                           Ix * vspr.R - Iy * vspr.R - Iy * vspr.X + Vx + Ey - Ex) ** 2 / (
                                                                                        -Ix * vspr.R - Ix * vspr.X - Iy * vspr.R - Vy + Ey + Ex) ** 2) * (
                                                                                       -Ix * vspr.R - Ix * vspr.X - Iy * vspr.R - Vy + Ey + Ex) ** 2)],
                  [-vspr.wn / vspr.X, 0],
                  [0, -vspr.wn / vspr.X],
                  ])

    C = np.zeros((2, vspr.nx))
    C[0, vspr.x_idx['Ix']] = 1
    C[1, vspr.x_idx['Iy']] = 1
    D = np.zeros((2, 2))

    return A, B, C, D


def vs_linear2(x0, u0, vspr):
    phi = x0[vspr.x_idx['phi']]
    # Ix = x0[vspr.x_idx['Ix']]
    # Iy = x0[vspr.x_idx['Iy']]
    Vx = u0[0]
    Vy = u0[1]

    Ex = vspr.V0 * np.cos(phi)
    Ey = vspr.V0 * np.sin(phi)
    Ix = (-vspr.R * (Vx - Ex) - vspr.X * (Vy - Ey)) / (vspr.R ** 2 + vspr.X ** 2)
    Iy = (vspr.X * (Vx - Ex) - vspr.R * (Vy - Ey)) / (vspr.R ** 2 + vspr.X ** 2)

    # ####
    # phi = np.deg2rad(10)
    # d = np.deg2rad(3)
    # theta = np.deg2rad(85)
    # E = 1.1*np.exp(1j*phi)
    # V = 1.05*np.exp(1j*d)
    # Z = 0.15*np.exp(1j*theta)

    # I = (E-V)/Z

    # (abs(E)*np.cos(phi-theta)-abs(V)*np.cos(d-theta))/abs(Z)
    # (abs(E)*np.sin(phi-theta)-abs(V)*np.sin(d-theta))/abs(Z)

    # (E.real*(1/Z).real-E.imag*(1/Z).imag
    # -V.real*(1/Z).real+V.imag*(1/Z).imag)

    # np.cos(-theta)/abs(Z)

    # (1/Z).real

    # r=Z.real
    # x = Z.imag

    # Ix = I.real
    # Iy = I.imag
    # Ex = E.real
    # Ey = E.imag

    # Ex-r*Ix+x*Iy
    # Ey-r*Iy-x*Ix

    # Zmat = np.array([[-Z.real,Z.imag],
    #                   [-Z.imag,-Z.real]])

    # Zmat_inv = np.linalg.inv(Zmat)

    # I2 = Zmat_inv@np.array([Vx-Ex,Vy-Ey])
    # I2 = Zmat_inv@np.array([Vx,Vy])-Zmat_inv@np.array([Ex,Ey])

    # # Iq = (gpr.xdpp*(Vd-Edpp)-gpr.ra*(Vq-Eqpp))/(gpr.ra**2+gpr.xqpp*gpr.xdpp)

    # Ix,Iy = vspr.Zvs_inv@np.array([vspr.V0*np.cos(phi)-Vx,vspr.V0*np.sin(phi)-Vy])

    # TODO this might be nicer with dq transformation
    # A = np.array([-1 + (Ex/(-Ix*vspr.X - Ix*vspr.R - Iy*vspr.R - Vy + Ex + vspr.Vy) + vspr.V0*(Ix*vspr.R - Iy*vspr.X - Iy*vspr.R + Vx + Ey - vspr.Vx)*np.sin(phi)/(-Ix*vspr.X - Ix*vspr.R - Iy*vspr.R - Vy + Ex + vspr.Vy)**2)/(1 + (Ix*vspr.R - Iy*vspr.X - Iy*vspr.R + Vx + Ey - vspr.Vx)**2/(-Ix*vspr.X - Ix*vspr.R - Iy*vspr.R - Vy + Ex + vspr.Vy)**2)])
    A = np.array([[-1 + (vspr.V0 * np.cos(phi) / (
            -Ix * vspr.R - Ix * vspr.X - Iy * vspr.R - Vy + vspr.V0 * np.cos(phi) + vspr.Vy) + vspr.V0 * (
                                 Ix * vspr.R - Iy * vspr.R - Iy * vspr.X + Vx + vspr.V0 * np.sin(
                             phi) - vspr.Vx) * np.sin(phi) / (
                                 -Ix * vspr.R - Ix * vspr.X - Iy * vspr.R - Vy + vspr.V0 * np.cos(
                             phi) + vspr.Vy) ** 2) / (1 + (
            Ix * vspr.R - Iy * vspr.R - Iy * vspr.X + Vx + vspr.V0 * np.sin(phi) - vspr.Vx) ** 2 / (
                                                              -Ix * vspr.R - Ix * vspr.X - Iy * vspr.R - Vy + vspr.V0 * np.cos(
                                                          phi) + vspr.Vy) ** 2)],
                  ])
    # B = np.array([1/((1 + (Ix*vspr.R - Iy*vspr.X - Iy*vspr.R + Vx + Ey - vspr.Vx)**2/(-Ix*vspr.X - Ix*vspr.R - Iy*vspr.R - Vy + Ex + vspr.Vy)**2)*(-Ix*vspr.X - Ix*vspr.R - Iy*vspr.R - Vy + Ex + vspr.Vy))])
    B = np.array([[1 / ((1 + (Ix * vspr.R - Iy * vspr.R - Iy * vspr.X + Vx + Ey - vspr.Vx) ** 2 / (
            -Ix * vspr.R - Ix * vspr.X - Iy * vspr.R - Vy + Ex + vspr.Vy) ** 2) * (
                                -Ix * vspr.R - Ix * vspr.X - Iy * vspr.R - Vy + Ex + vspr.Vy)),
                   (Ix * vspr.R - Iy * vspr.R - Iy * vspr.X + Vx + Ey - vspr.Vx) / ((1 + (
                           Ix * vspr.R - Iy * vspr.R - Iy * vspr.X + Vx + Ey - vspr.Vx) ** 2 / (
                                                                                             -Ix * vspr.R - Ix * vspr.X - Iy * vspr.R - Vy + Ex + vspr.Vy) ** 2) * (
                                                                                            -Ix * vspr.R - Ix * vspr.X - Iy * vspr.R - Vy + Ex + vspr.Vy) ** 2)]])

    C = -vspr.Zvs_inv @ np.array([[Ex], [Ey]])

    D = vspr.Zvs_inv

    return A, B, C, D


def sixth_order_model(x0, u0, gpr):
    delta = x0[gpr.x_idx['d']] + np.pi / 2  # XXX delta is shifted 90 degrees due to different dq transformation
    e_qpp = x0[gpr.x_idx['Eqpp']]
    e_dpp = x0[gpr.x_idx['Edpp']]
    v_d = u0[0][0]
    v_q = u0[1][0]

    i_d, i_q = gpr.Zg_inv @ np.array([v_d - e_dpp, v_q - e_qpp])

    Ag_bar = np.array([[0, gpr.wn, 0, 0, 0, 0],
                       [0, -gpr.D / gpr.Tj, 0, -i_q / gpr.Tj, 0, -i_d / gpr.Tj],
                       [0, 0, -gpr.kd / gpr.Tdp, (gpr.kd - 1) / gpr.Tdp, 0, 0],
                       [0, 0, 1 / gpr.Tdpp, -1 / gpr.Tdpp, 0, 0],
                       [0, 0, 0, 0, -gpr.kq / gpr.Tqp, (gpr.kq - 1) / gpr.Tqp],
                       [0, 0, 0, 0, 1 / gpr.Tqpp, -1 / gpr.Tqpp]])

    Big_bar = np.array([[0, 0],
                        [((gpr.xdpp - gpr.xqpp) * i_q - e_dpp) / gpr.Tj,
                         ((gpr.xdpp - gpr.xqpp) * i_d - e_qpp) / gpr.Tj],
                        [1.1 / gpr.Tdp, 0],
                        [(gpr.xdpp - gpr.xdp) / gpr.Tdpp, 0],
                        [0, 0],
                        [0, (gpr.xqp - gpr.xqpp) / gpr.Tqpp]])

    Bvg_bar = np.array([[0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0]])

    Pg_bar = np.array([[0, 0, 0, 0, 0, 1],
                       [0, 0, 0, 1, 0, 0]])
    Zg_bar = np.array([[-gpr.ra, gpr.xqpp],
                       [-gpr.xdpp, -gpr.ra]])

    Tg0 = np.array([[np.sin(delta), -np.cos(delta)],
                    [np.cos(delta), np.sin(delta)]])

    Rvg = np.array([[v_q, 0, 0, 0, 0, 0],
                    [-v_d, 0, 0, 0, 0, 0]])
    Rig = np.array([[i_q, 0, 0, 0, 0, 0],
                    [-i_d, 0, 0, 0, 0, 0]])

    Cg = Tg0.T @ (np.linalg.inv(Zg_bar) @ (Rvg - Pg_bar) - Rig)
    Dg = Tg0.T @ np.linalg.inv(Zg_bar) @ Tg0

    Ag = Ag_bar + Big_bar @ np.linalg.inv(Zg_bar) @ (Rvg - Pg_bar) + Bvg_bar @ Rvg
    Bg = (Big_bar @ np.linalg.inv(Zg_bar) + Bvg_bar) @ Tg0

    return Ag, Bg, Cg, Dg


def sixth_order_model_avr_(x0, u0, gpr):
    delta = x0[gpr.x_idx['d']] + np.pi / 2  # XXX delta is shifted 90 degrees due to different dq transformation
    e_qpp = x0[gpr.x_idx['Eqpp']]
    e_dpp = x0[gpr.x_idx['Edpp']]
    v_d = u0[0]  # [0]
    v_q = u0[1]  # [0]

    i_d, i_q = gpr.Zg_inv @ np.array([v_d - e_dpp, v_q - e_qpp])

    Ag_bar = np.array([[0, gpr.wn, 0, 0, 0, 0, 0, 0, 0],
                       [0, -gpr.D / gpr.Tj, 0, -i_q / gpr.Tj, 0, -i_d / gpr.Tj, 0, 0, 0],
                       [0, 0, -gpr.kd / gpr.Tdp, (gpr.kd - 1) / gpr.Tdp, 0, 0, 1 / gpr.Tdp, 0, 0],
                       [0, 0, 1 / gpr.Tdpp, -1 / gpr.Tdpp, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, -gpr.kq / gpr.Tqp, (gpr.kq - 1) / gpr.Tqp, 0, 0, 0],
                       [0, 0, 0, 0, 1 / gpr.Tqpp, -1 / gpr.Tqpp, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, -1 / gpr.Te, -gpr.Kc / gpr.Te, gpr.Kc / (gpr.Tc * gpr.Te)],
                       [0, 0, 0, 0, 0, 0, 0, -1 / gpr.Tm, 0],
                       [0, 0, 0, 0, 0, 0, 0, -1, 0],
                       ])

    Big_bar = np.array([[0, 0],
                        [(-e_dpp + gpr.xdpp * i_q - gpr.xqpp * i_q) / gpr.Tj,
                         (-e_qpp + gpr.xdpp * i_d - gpr.xqpp * i_d) / gpr.Tj],
                        [0, 0],
                        [(-gpr.xdp + gpr.xdpp) / gpr.Tdpp, 0],
                        [0, 0],
                        [0, (gpr.xqp - gpr.xqpp) / gpr.Tqpp],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        ])

    Bvg_bar = np.array([[0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [v_d / (gpr.Tm * np.sqrt(v_d ** 2 + v_q ** 2)), v_q / (gpr.Tm * np.sqrt(v_d ** 2 + v_q ** 2))],
                        [0, 0],
                        ])

    Pg_bar = np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 1, 0, 0, 0, 0, 0]])

    Zg_bar = np.array([[-gpr.ra, gpr.xqpp],
                       [-gpr.xdpp, -gpr.ra]])

    Tg0 = np.array([[np.sin(delta), -np.cos(delta)],
                    [np.cos(delta), np.sin(delta)]])

    Rvg = np.array([[v_q, 0, 0, 0, 0, 0, 0, 0, 0],
                    [-v_d, 0, 0, 0, 0, 0, 0, 0, 0]])
    Rig = np.array([[i_q, 0, 0, 0, 0, 0, 0, 0, 0],
                    [-i_d, 0, 0, 0, 0, 0, 0, 0, 0]])

    Cg = (Tg0.T @ (np.linalg.inv(Zg_bar) @ (
            Rvg - Pg_bar) - Rig)) * gpr.Sn / gpr.Sb  # TODO move *gpr.Sn/gpr.Sb outside of this function
    Dg = (Tg0.T @ np.linalg.inv(Zg_bar) @ Tg0) * gpr.Sn / gpr.Sb

    Ag = Ag_bar + Big_bar @ np.linalg.inv(Zg_bar) @ (Rvg - Pg_bar) + Bvg_bar @ Rvg
    Bg = (Big_bar @ np.linalg.inv(Zg_bar) + Bvg_bar) @ Tg0

    return Ag, Bg, Cg, Dg


def sixth_order_model_avr(x0, u0, gpr):
    delta = x0[gpr.x_idx['d']] + np.pi / 2  # XXX delta is shifted 90 degrees due to different dq transformation
    e_qpp = x0[gpr.x_idx['Eqpp']]
    e_dpp = x0[gpr.x_idx['Edpp']]
    v_d = u0[0]  # [0]
    v_q = u0[1]  # [0]

    i_d, i_q = gpr.Zg_inv @ np.array([v_d - e_dpp, v_q - e_qpp])

    Ag_bar = np.array([[0, gpr.wn, 0, 0, 0, 0, 0, 0, 0],
                       [0, -gpr.D / gpr.Tj, 0, -i_q / gpr.Tj, 0, -i_d / gpr.Tj, 0, 0, 0],
                       [0, 0, -1 / gpr.Tdp, 0, 0, 0, 1 / gpr.Tdp, 0, 0],
                       [0, 0, 1 / gpr.Tdpp, -1 / gpr.Tdpp, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, -1 / gpr.Tqp, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1 / gpr.Tqpp, -1 / gpr.Tqpp, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, -1 / gpr.Te, -gpr.Kc / gpr.Te, gpr.Kc / (gpr.Tc * gpr.Te)],
                       [0, 0, 0, 0, 0, 0, 0, -1 / gpr.Tm, 0],
                       [0, 0, 0, 0, 0, 0, 0, -1, 0],
                       ])

    Big_bar = np.array([[0, 0],
                        [(-e_dpp + gpr.xdpp * i_q - gpr.xqpp * i_q) / gpr.Tj,
                         (-e_qpp + gpr.xdpp * i_d - gpr.xqpp * i_d) / gpr.Tj],
                        [(-gpr.xd + gpr.xdp) / gpr.Tdp, 0],
                        [(-gpr.xdp + gpr.xdpp) / gpr.Tdpp, 0],
                        [0, (gpr.xq - gpr.xqp) / gpr.Tqp],
                        [0, (gpr.xqp - gpr.xqpp) / gpr.Tqpp],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        ])

    Bvg_bar = np.array([[0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [v_d / (gpr.Tm * np.sqrt(v_d ** 2 + v_q ** 2)), v_q / (gpr.Tm * np.sqrt(v_d ** 2 + v_q ** 2))],
                        [0, 0],
                        ])

    Pg_bar = np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 1, 0, 0, 0, 0, 0]])

    Zg_bar = np.array([[-gpr.ra, gpr.xqpp],
                       [-gpr.xdpp, -gpr.ra]])

    Tg0 = np.array([[np.sin(delta), -np.cos(delta)],
                    [np.cos(delta), np.sin(delta)]])

    Rvg = np.array([[v_q, 0, 0, 0, 0, 0, 0, 0, 0],
                    [-v_d, 0, 0, 0, 0, 0, 0, 0, 0]])
    Rig = np.array([[i_q, 0, 0, 0, 0, 0, 0, 0, 0],
                    [-i_d, 0, 0, 0, 0, 0, 0, 0, 0]])

    Cg = (Tg0.T @ (np.linalg.inv(Zg_bar) @ (
            Rvg - Pg_bar) - Rig)) * gpr.Sn / gpr.Sb  # TODO move *gpr.Sn/gpr.Sb outside of this function
    Dg = (Tg0.T @ np.linalg.inv(Zg_bar) @ Tg0) * gpr.Sn / gpr.Sb

    Ag = Ag_bar + Big_bar @ np.linalg.inv(Zg_bar) @ (Rvg - Pg_bar) + Bvg_bar @ Rvg
    Bg = (Big_bar @ np.linalg.inv(Zg_bar) + Bvg_bar) @ Tg0

    return Ag, Bg, Cg, Dg


def standard_model_linear(xo, uo, gpr):
    delta = xo[gpr.x_idx['d']]
    w = xo[gpr.x_idx['w']]
    i_d = xo[gpr.x_idx['Id']]
    i_q = xo[gpr.x_idx['Iq']]
    psi_d = xo[gpr.x_idx['psi_d']]
    psi_q = xo[gpr.x_idx['psi_q']]
    psi_fd = xo[gpr.x_idx['psi_fd']]
    psi_1d = xo[gpr.x_idx['psi_1d']]
    psi_2q = xo[gpr.x_idx['psi_2q']]
    psi_1q = xo[gpr.x_idx['psi_1q']]
    Vd = uo[0]
    Vq = uo[1]

    Ag = np.array([[0, gpr.wn, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, (-gpr.dkd - gpr.dpe / w + gpr.dpe * (w - 1) / w ** 2) / gpr.Tj, -i_q / (gpr.Tj * gpr.cosn),
                    i_d / (gpr.Tj * gpr.cosn), 0, 0, 0, 0, psi_q / (gpr.Tj * gpr.cosn), -psi_d / (gpr.Tj * gpr.cosn), 0,
                    0, 0],
                   [0, gpr.wn * psi_q, 0, gpr.wn * w, 0, 0, 0, 0, gpr.ra * gpr.wn, 0, 0, 0, 0],
                   [0, -gpr.wn * psi_d, -gpr.wn * w, 0, 0, 0, 0, 0, 0, gpr.ra * gpr.wn, 0, 0, 0],
                   [0, 0, 0, 0, -gpr.rfd * gpr.wn * gpr.x1d_loop / gpr.xdet_d,
                    -gpr.rfd * gpr.wn * (-gpr.xad - gpr.xrld) / gpr.xdet_d, 0, 0, -gpr.kfd * gpr.rfd * gpr.wn, 0,
                    gpr.rfd * gpr.wn / gpr.xadu, 0, 0],
                   [0, 0, 0, 0, -gpr.r1d * gpr.wn * (-gpr.xad - gpr.xrld) / gpr.xdet_d,
                    -gpr.r1d * gpr.wn * gpr.xfd_loop / gpr.xdet_d, 0, 0, -gpr.k1d * gpr.r1d * gpr.wn, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, -gpr.r1q * gpr.wn * gpr.x2q_loop / gpr.xdet_q,
                    -gpr.r1q * gpr.wn * (-gpr.xaq - gpr.xrlq) / gpr.xdet_q, 0, -gpr.k1q * gpr.r1q * gpr.wn, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, -gpr.r2q * gpr.wn * (-gpr.xaq - gpr.xrlq) / gpr.xdet_q,
                    -gpr.r2q * gpr.wn * gpr.x1q_loop / gpr.xdet_q, 0, -gpr.k2q * gpr.r2q * gpr.wn, 0, 0, 0],
                   [0, gpr.wn * (-gpr.k1q * psi_1q - gpr.k2q * psi_2q + gpr.xqpp * i_q) / gpr.xdpp, 0, 0, gpr.wn * (
                           -gpr.k1d * gpr.r1d * (
                           -gpr.xad - gpr.xrld) / gpr.xdet_d - gpr.kfd * gpr.rfd * gpr.x1d_loop / gpr.xdet_d) / gpr.xdpp,
                    gpr.wn * (-gpr.k1d * gpr.r1d * gpr.xfd_loop / gpr.xdet_d - gpr.kfd * gpr.rfd * (
                            -gpr.xad - gpr.xrld) / gpr.xdet_d) / gpr.xdpp, -gpr.k1q * gpr.wn * w / gpr.xdpp,
                    -gpr.k2q * gpr.wn * w / gpr.xdpp,
                    gpr.wn * (-gpr.k1d ** 2 * gpr.r1d - gpr.kfd ** 2 * gpr.rfd - gpr.ra) / gpr.xdpp,
                    gpr.wn * gpr.xqpp * w / gpr.xdpp, gpr.kfd * gpr.rfd * gpr.wn / (gpr.xadu * gpr.xdpp), 0, 0],
                   [0, gpr.wn * (gpr.k1d * psi_1d + gpr.kfd * psi_fd - gpr.xdpp * i_d) / gpr.xqpp, 0, 0,
                    gpr.kfd * gpr.wn * w / gpr.xqpp, gpr.k1d * gpr.wn * w / gpr.xqpp, gpr.wn * (
                            -gpr.k1q * gpr.r1q * gpr.x2q_loop / gpr.xdet_q - gpr.k2q * gpr.r2q * (
                            -gpr.xaq - gpr.xrlq) / gpr.xdet_q) / gpr.xqpp, gpr.wn * (-gpr.k1q * gpr.r1q * (
                           -gpr.xaq - gpr.xrlq) / gpr.xdet_q - gpr.k2q * gpr.r2q * gpr.x1q_loop / gpr.xdet_q) / gpr.xqpp,
                    -gpr.wn * gpr.xdpp * w / gpr.xqpp,
                    gpr.wn * (-gpr.k1q ** 2 * gpr.r1q - gpr.k2q ** 2 * gpr.r2q - gpr.ra) / gpr.xqpp, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1 / gpr.Te, -gpr.Kc / gpr.Te, gpr.Kc / (gpr.Tc * gpr.Te)],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1 / gpr.Tm, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
                   ])

    Big = np.array([[0, 0],
                    [psi_q / (gpr.Tj * gpr.cosn), -psi_d / (gpr.Tj * gpr.cosn)],
                    [gpr.ra * gpr.wn, 0],
                    [0, gpr.ra * gpr.wn],
                    [-gpr.kfd * gpr.rfd * gpr.wn, 0],
                    [-gpr.k1d * gpr.r1d * gpr.wn, 0],
                    [0, -gpr.k1q * gpr.r1q * gpr.wn],
                    [0, -gpr.k2q * gpr.r2q * gpr.wn],
                    [gpr.wn * (-gpr.k1d ** 2 * gpr.r1d - gpr.kfd ** 2 * gpr.rfd - gpr.ra) / gpr.xdpp,
                     gpr.wn * gpr.xqpp * w / gpr.xdpp],
                    [-gpr.wn * gpr.xdpp * w / gpr.xqpp,
                     gpr.wn * (-gpr.k1q ** 2 * gpr.r1q - gpr.k2q ** 2 * gpr.r2q - gpr.ra) / gpr.xqpp],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    ])
    Bvg = np.array([[0, 0],
                    [0, 0],
                    [gpr.wn, 0],
                    [0, gpr.wn],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [-gpr.wn / gpr.xdpp, 0],
                    [0, -gpr.wn / gpr.xqpp],
                    [0, 0],
                    [Vd / (gpr.Tm * np.sqrt(Vd ** 2 + Vq ** 2)), Vq / (gpr.Tm * np.sqrt(Vd ** 2 + Vq ** 2))],
                    [0, 0],
                    ])

    Tg = np.array([[np.cos(delta), np.sin(delta)],
                   [-np.sin(delta), np.cos(delta)]])

    # Zg = np.array([[-gpr.ra,gpr.xqpp],[-gpr.xdpp,-gpr.ra]])

    Rvg = np.zeros((2, gpr.nx))
    Rvg[0, gpr.x_idx['d']] = Vq
    Rvg[1, gpr.x_idx['d']] = -Vd

    Rig = np.zeros((2, gpr.nx))
    Rig[0, gpr.x_idx['d']] = i_q
    Rig[1, gpr.x_idx['d']] = -i_d

    Pg = np.zeros((2, gpr.nx))
    Pg[0, gpr.x_idx['Id']] = 1
    Pg[1, gpr.x_idx['Iq']] = 1
    Pg[0, gpr.x_idx['d']] = -i_q
    Pg[1, gpr.x_idx['d']] = i_d

    # Cg = Tg.T@(np.linalg.inv(Zg)@(Rvg-Pg)-Rig)
    Cg = Tg.T @ Pg

    # Cdc = np.zeros((1,gpr.nx))
    # Cdc[0,gpr.x_idx['Idc']] = 1
    Dg = np.zeros((2, 2))

    A = Ag + Bvg @ Rvg
    B = Bvg @ Tg
    C = Cg * gpr.Sn / gpr.Sb
    D = Dg

    C @ xo

    return A, B, C, D


def network_linear(npr):
    for i in range(npr.n_br):
        f = npr.f[i]
        t = npr.t[i]

        R = np.real(-1 / npr.Ybus[f, t])
        L = np.imag(-1 / npr.Ybus[f, t])

    s = 2 * (npr.n_bus + npr.n_br)
    Nb = npr.n_bus * 2
    A = np.zeros((s, s))
    B = np.zeros((s, Nb))

    for n in range(0, Nb, 2):
        A[n, n + 1] = npr.wn
        A[n + 1, n] = -npr.wn

    ib = Nb
    ii = 0

    for n in range(npr.n_bus):
        Csh = np.imag(npr.Ybus[n].sum())
        if Csh <= 1e-6:
            Csh = 0.001
        for i, (f, t) in enumerate(zip(npr.f, npr.t)):
            R = np.real(-1 / npr.Ybus[f, t])
            L = np.imag(-1 / npr.Ybus[f, t])
            if f == n:
                A[2 * n, 2 * i + ib] = -npr.wn / Csh
                A[2 * n + 1, 2 * i + ib + 1] = -npr.wn / Csh
            if t == n:
                A[2 * n, 2 * i + ib] = npr.wn / Csh
                A[2 * n + 1, 2 * i + ib + 1] = npr.wn / Csh

                A[2 * i + ib, 2 * n] = -npr.wn / L
                A[2 * i + ib, 2 * f] = npr.wn / L

                A[2 * i + ib, 2 * i + ib] = -R * npr.wn / L
                A[2 * i + ib, 2 * i + ib + 1] = npr.wn

                A[2 * i + ib + 1, 2 * n + 1] = -npr.wn / L
                A[2 * i + ib + 1, 2 * f + 1] = npr.wn / L

                A[2 * i + ib + 1, 2 * i + ib] = -npr.wn
                A[2 * i + ib + 1, 2 * i + ib + 1] = -R * npr.wn / L

        B[2 * n, 2 * n] = npr.wn / Csh
        B[2 * n + 1, 2 * n + 1] = npr.wn / Csh
        ii += 2

    # ###
    # Ybus_xy = np.zeros((Nb,Nb))
    # for i in range(npr.n_bus):
    #     k = 2*i
    #     Ybus_xy[k,k] = npr.Ybus[i,i].real
    #     Ybus_xy[k,k+1] = -npr.Ybus[i,i].imag
    #     Ybus_xy[k+1,k] = npr.Ybus[i,i].imag
    #     Ybus_xy[k+1,k+1] = npr.Ybus[i,i].real

    # for f,t in zip(npr.f,npr.t):
    #     f2 = 2*f
    #     t2 = 2*t
    #     Ybus_xy[f2,t2] = npr.Ybus[f,t].real
    #     Ybus_xy[f2,t2+1] = -npr.Ybus[f,t].imag
    #     Ybus_xy[f2+1,t2] = npr.Ybus[f,t].imag
    #     Ybus_xy[f2+1,t2+1] = npr.Ybus[f,t].real

    #     Ybus_xy[t2,f2] = npr.Ybus[t,f].real
    #     Ybus_xy[t2,f2+1] = -npr.Ybus[t,f].imag
    #     Ybus_xy[t2+1,f2] = npr.Ybus[t,f].imag
    #     Ybus_xy[t2+1,f2+1] = npr.Ybus[t,f].real

    # ###
    C = np.zeros((Nb, s))
    v_inds = range(Nb)
    C[v_inds, v_inds] = 1

    # D = np.linalg.inv(Ybus_xy)
    D = np.zeros((Nb, Nb))

    return A, B, C, D


def calc_eigenvalues_old(x0, npr, models):
    """


    Parameters
    ----------
    x0 : ndarray
        Initial state vector.
    npr : network_parameters
        Object containing network parameters.
    models : list
        List of models.

    Returns
    -------
    lambda_1 : ndarray
        Eigenvalue vector.
    P_1 : ndarray
        Participation factor array.
    Amat : ndarray
        System A matrix.

    """

    inw = npr.x_ind
    n_vsc = 0
    n_gen = 0

    size = (len(x0), len(x0))
    A_tilde = np.zeros(size)
    B_tilde = np.zeros(size)
    C_tilde = np.zeros(size)

    an, bn, cn, dn = network_linear(npr)
    A_tilde[inw:, inw:] = an

    for model in models:

        if model.type == ModelType.VSC_1:
            # break
            bus_idx = model.bus_ind * 2
            idx = inw + bus_idx
            vx = x0[idx]
            vy = x0[idx + 1]

            Theta_pll = x0[model.x_ind + model.x_idx['Theta']]
            vd = vx * np.cos(Theta_pll) + vy * np.sin(Theta_pll)
            vq = -vx * np.sin(Theta_pll) + vy * np.cos(Theta_pll)

            uo = np.array([vd, vq])
            xo = x0[np.arange(model.x_ind, model.x_ind + model.nx)]

            avsc, bvsc, cvsc, dvsc = converter_linear(xo, uo, model)

            A_tilde[model.x_ind:model.x_ind + model.nx, model.x_ind:model.x_ind + model.nx] = avsc
            B_tilde[model.x_ind:model.x_ind + model.nx, idx:idx + 2] = bvsc
            B_tilde[idx:idx + 2, idx:idx + 2] = bn[model.bus_ind:model.bus_ind + 2,
                                                model.bus_ind:model.bus_ind + 2] @ dvsc
            C_tilde[idx:idx + 2, model.x_ind:model.x_ind + model.nx] = bn[bus_idx:bus_idx + 2,
                                                                       bus_idx:bus_idx + 2] @ cvsc

            n_vsc += 1

        elif model.type == ModelType.gen:
            bus_idx = model.bus_ind * 2
            idx = inw + bus_idx
            vx = x0[idx]
            vy = x0[idx + 1]
            d = x0[model.x_ind + model.x_idx['d']]

            vd = vx * np.cos(d) + vy * np.sin(d)
            vq = -vx * np.sin(d) + vy * np.cos(d)

            uo = np.array([vd, vq])
            xo = x0[np.arange(model.x_ind, model.x_ind + model.nx)]

            ag, bg, cg, dg = sixth_order_model_avr(xo, uo, model)

            A_tilde[model.x_ind:model.x_ind + model.nx, model.x_ind:model.x_ind + model.nx] = ag
            B_tilde[model.x_ind:model.x_ind + model.nx, idx:idx + 2] = bg
            B_tilde[idx:idx + 2, idx:idx + 2] = bn[bus_idx:bus_idx + 2, bus_idx:bus_idx + 2] @ dg
            C_tilde[idx:idx + 2, model.x_ind:model.x_ind + model.nx] = bn[bus_idx:bus_idx + 2, bus_idx:bus_idx + 2] @ cg

            n_gen += 1

    Amat = A_tilde + B_tilde + C_tilde
    lambda_1, Phi_1 = np.linalg.eig(Amat)

    Psi_1 = np.linalg.inv(Phi_1)

    P_1 = Phi_1 * Psi_1.T

    # # Set Zero mode eigenvalues to zero # TODO need to confirm that this is ok!
    # zm = np.where(abs(P_1) > 1)[1]
    # lambda_1[zm] = 0

    return lambda_1, P_1, Amat


def calc_eigenvalues(x0, npr, models, tol=1e-6):
    """

    Parameters
    ----------
    x0 : ndarray
        Initial state vector.
    npr : network_parameters
        Object containing network parameters.
    models : list
        List of models.

    Returns
    -------
    lambda_1 : ndarray
        Eigenvalue vector.
    P_1 : ndarray
        Participation factor array.
    Amat : ndarray
        System A matrix.

    """

    inw = npr.x_ind
    n_vsc = 0
    n_gen = 0

    size = (len(x0), len(x0))
    A_tilde = np.zeros(size)
    B_tilde = np.zeros(size)
    C_tilde = np.zeros(size)

    an, bn, cn, dn = network_linear(npr)
    A_tilde[inw:, inw:] = an

    for model in models:

        if model.type == ModelType.VSC_1:
            # break
            bus_idx = model.bus_ind * 2
            idx = inw + bus_idx
            vx = x0[idx]
            vy = x0[idx + 1]

            if not model.x_dc == -1:
                vdc = x0[model.x_dc]
            else:
                vdc = 1

            Theta_pll = x0[model.x_ind + model.x_idx['Theta']]
            vd = vx * np.cos(Theta_pll) + vy * np.sin(Theta_pll)
            vq = -vx * np.sin(Theta_pll) + vy * np.cos(Theta_pll)

            uo = np.array([vd, vq, vdc])
            xo = x0[np.arange(model.x_ind, model.x_ind + model.nx)]

            # if model.ctrl == CtrlMode.P_Vac:
            #     avsc,bacvsc,bdcvsc,cacvsc,cdcvsc,dvsc = vsc_linear_1(xo,uo,model)
            #     # A,B,Bvdc,C,Cdc,D
            #     # break
            #     # B_tilde[model.x_ind:model.x_ind+model.nx,model.x_dc:model.x_dc+1]=bdcvsc # TODO add DC side
            #     # C_tilde[idx:idx+1,model.x_ind:model.x_ind+model.nx]=bdcvsc
            # elif model.ctrl == CtrlMode.Vdc_Q:
            #     avsc,bacvsc,bdcvsc,cacvsc,cdcvsc,dvsc = vsc_linear_2(xo,uo,model)
            # elif 'Idc' in model.x_idx:
            #     avsc,bacvsc,cacvsc,dvsc = converter_linear(xo,uo,model)
            # else:
            #     # avsc,bvsc,cvsc,dvsc = converter_linear(xo,uo,model)
            # avsc,bacvsc,cacvsc,dvsc = c_linear0(xo,uo,model)

            # avsc,bacvsc,cacvsc,dvsc = converter_linear(xo,uo,model)

            avsc, bacvsc, cacvsc, dvsc = model.abcd_linear(xo, uo)

            A_tilde[model.x_ind:model.x_ind + model.nx, model.x_ind:model.x_ind + model.nx] = avsc
            B_tilde[model.x_ind:model.x_ind + model.nx, idx:idx + 2] = bacvsc
            B_tilde[idx:idx + 2, idx:idx + 2] = bn[model.bus_ind:model.bus_ind + 2,
                                                model.bus_ind:model.bus_ind + 2] @ dvsc
            C_tilde[idx:idx + 2, model.x_ind:model.x_ind + model.nx] = bn[bus_idx:bus_idx + 2,
                                                                       bus_idx:bus_idx + 2] @ cacvsc

            # B_tilde[model.x_ind:model.x_ind+model.nx,inw:] = bacvsc@cn[bus_idx:bus_idx+2] # XXX alternative way

            # if not model.x_dc == -1:
            #     B_tilde[model.x_ind:model.x_ind+model.nx,[model.x_dc]]=bdcvsc

            n_vsc += 1

        elif model.type == ModelType.GEN_ORD_6:
            bus_idx = model.bus_ind * 2
            idx = inw + bus_idx
            vx = x0[idx]
            vy = x0[idx + 1]
            d = x0[model.x_ind + model.x_idx['d']]

            vd = vx * np.cos(d) + vy * np.sin(d)
            vq = -vx * np.sin(d) + vy * np.cos(d)

            uo = np.array([vd, vq])
            xo = x0[np.arange(model.x_ind, model.x_ind + model.nx)]

            ag, bg, cg, dg = sixth_order_model_avr(xo, uo, model)

            A_tilde[model.x_ind:model.x_ind + model.nx, model.x_ind:model.x_ind + model.nx] = ag
            B_tilde[model.x_ind:model.x_ind + model.nx, idx:idx + 2] = bg
            B_tilde[idx:idx + 2, idx:idx + 2] = bn[bus_idx:bus_idx + 2, bus_idx:bus_idx + 2] @ dg
            C_tilde[idx:idx + 2, model.x_ind:model.x_ind + model.nx] = bn[bus_idx:bus_idx + 2, bus_idx:bus_idx + 2] @ cg

            n_gen += 1

        elif model.type == ModelType.GEN_2_2:

            bus_idx = model.bus_ind * 2
            idx = inw + bus_idx
            vx = x0[idx]
            vy = x0[idx + 1]
            d = x0[model.x_ind + model.x_idx['d']]

            vd = vx * np.cos(d) + vy * np.sin(d)
            vq = -vx * np.sin(d) + vy * np.cos(d)

            uo = np.array([vd, vq])
            xo = x0[np.arange(model.x_ind, model.x_ind + model.nx)]

            # ag,bg,cg,dg = standard_model_linear(xo,uo,model)
            ag, bg, cg, dg = model.abcd_linear(xo, uo)

            A_tilde[model.x_ind:model.x_ind + model.nx, model.x_ind:model.x_ind + model.nx] = ag
            B_tilde[model.x_ind:model.x_ind + model.nx, idx:idx + 2] = bg
            B_tilde[idx:idx + 2, idx:idx + 2] = bn[bus_idx:bus_idx + 2, bus_idx:bus_idx + 2] @ dg
            C_tilde[idx:idx + 2, model.x_ind:model.x_ind + model.nx] = bn[bus_idx:bus_idx + 2, bus_idx:bus_idx + 2] @ cg

            n_gen += 1

        elif model.type == ModelType.DC_LINE:
            # break
            mf = models[model.f]
            mt = models[model.t]
            If = -x0[mf.x_ind + mf.x_idx['Idc']] * mf.Sn / npr.Sb

            It = x0[mt.x_ind + mt.x_idx['Idc']] * mt.Sn / npr.Sb
            It = -If  # TODO check the initialization. In steady state If=-It (if G=0)
            uo = np.array([If, It])
            xo = x0[np.arange(model.x_ind, model.x_ind + model.nx)]

            acb, bcb, ccb, dcb = dc_cable_linear(xo, uo, model)

            A_tilde[model.x_ind:model.x_ind + model.nx, model.x_ind:model.x_ind + model.nx] = acb

            # cvsc[np.array([0,1]),np.array([mf.x_idx['Idc'],mt.x_idx['Idc']])]

            # B_tilde[model.x_ind:model.x_ind+model.nx,idx:idx+2]=bcb
            # B_tilde[idx:idx+2,idx:idx+2]=bn[bus_idx:bus_idx+2,bus_idx:bus_idx+2]@dcb

            cvsc = np.zeros((1, mf.nx))
            cvsc[0, mf.x_idx['Idc']] = 1

            C_tilde[model.x_ind:model.x_ind + model.nx, mf.x_ind:mf.x_ind + mf.nx] = bcb[:, [0]] @ cvsc
            C_tilde[model.x_ind:model.x_ind + model.nx, mt.x_ind:mt.x_ind + mt.nx] = bcb[:, [1]] @ cvsc


        elif model.type == ModelType.VS:
            # break
            bus_idx = model.bus_ind * 2
            idx = inw + bus_idx
            vx = x0[idx]
            vy = x0[idx + 1]
            uo = np.array([vx, vy])
            xo = x0[np.arange(model.x_ind, model.x_ind + model.nx)]
            avs, bvs, cvs, dvs = vs_linear(xo, uo, model)

            A_tilde[model.x_ind:model.x_ind + model.nx, model.x_ind:model.x_ind + model.nx] = avs
            B_tilde[model.x_ind:model.x_ind + model.nx, idx:idx + 2] = bvs
            B_tilde[idx:idx + 2, idx:idx + 2] = bn[bus_idx:bus_idx + 2, bus_idx:bus_idx + 2] @ dvs
            C_tilde[idx:idx + 2, model.x_ind:model.x_ind + model.nx] = bn[bus_idx:bus_idx + 2,
                                                                       bus_idx:bus_idx + 2] @ cvs

    Amat = A_tilde + B_tilde + C_tilde
    lambda_1, Phi_1 = np.linalg.eig(Amat)

    # Set eigenvalues smaller than the tolerance to zero
    lambda_1[abs(np.real(lambda_1)) < tol] = 0

    Psi_1 = np.linalg.inv(Phi_1)
    P_1 = Phi_1 * Psi_1.T

    # Set Zero mode eigenvalues to zero # TODO need to confirm that this is ok!
    zm = np.where(abs(P_1) > 1.1)[0]
    lambda_1[zm] = 0

    return lambda_1, P_1, Amat


def calc_state_matrix(x0, npr, models, tol=1e-6):
    """

    Parameters
    ----------
    x0 : ndarray
        Initial state vector.
    npr : network_parameters
        Object containing network parameters.
    models : list
        List of models.

    Returns
    -------
        System A matrix.

    """

    inw = npr.x_ind
    n_vsc = 0
    n_gen = 0

    size = (len(x0), len(x0))
    A_tilde = np.zeros(size)
    B_tilde = np.zeros(size)
    C_tilde = np.zeros(size)

    an, bn, cn, dn = network_linear(npr)
    A_tilde[inw:, inw:] = an

    for model in models:

        if model.type == ModelType.VSC_1:
            # break
            bus_idx = model.bus_ind * 2
            idx = inw + bus_idx
            vx = x0[idx]
            vy = x0[idx + 1]

            if not model.x_dc == -1:
                vdc = x0[model.x_dc]
            else:
                vdc = 1

            Theta_pll = x0[model.x_ind + model.x_idx['Theta']]
            vd = vx * np.cos(Theta_pll) + vy * np.sin(Theta_pll)
            vq = -vx * np.sin(Theta_pll) + vy * np.cos(Theta_pll)

            uo = np.array([vd, vq, vdc])
            xo = x0[np.arange(model.x_ind, model.x_ind + model.nx)]

            # if model.ctrl == CtrlMode.P_Vac:
            #     avsc,bacvsc,bdcvsc,cacvsc,cdcvsc,dvsc = vsc_linear_1(xo,uo,model)
            #     # A,B,Bvdc,C,Cdc,D
            #     # break
            #     # B_tilde[model.x_ind:model.x_ind+model.nx,model.x_dc:model.x_dc+1]=bdcvsc # TODO add DC side
            #     # C_tilde[idx:idx+1,model.x_ind:model.x_ind+model.nx]=bdcvsc
            # elif model.ctrl == CtrlMode.Vdc_Q:
            #     avsc,bacvsc,bdcvsc,cacvsc,cdcvsc,dvsc = vsc_linear_2(xo,uo,model)
            # elif 'Idc' in model.x_idx:
            #     avsc,bacvsc,cacvsc,dvsc = converter_linear(xo,uo,model)
            # else:
            #     # avsc,bvsc,cvsc,dvsc = converter_linear(xo,uo,model)
            # avsc,bacvsc,cacvsc,dvsc = c_linear0(xo,uo,model)

            # avsc,bacvsc,cacvsc,dvsc = converter_linear(xo,uo,model)

            avsc, bacvsc, cacvsc, dvsc = model.abcd_linear(xo, uo)

            A_tilde[model.x_ind:model.x_ind + model.nx, model.x_ind:model.x_ind + model.nx] = avsc
            B_tilde[model.x_ind:model.x_ind + model.nx, idx:idx + 2] = bacvsc
            B_tilde[idx:idx + 2, idx:idx + 2] = bn[model.bus_ind:model.bus_ind + 2,
                                                model.bus_ind:model.bus_ind + 2] @ dvsc
            C_tilde[idx:idx + 2, model.x_ind:model.x_ind + model.nx] = bn[bus_idx:bus_idx + 2,
                                                                       bus_idx:bus_idx + 2] @ cacvsc

            # B_tilde[model.x_ind:model.x_ind+model.nx,inw:] = bacvsc@cn[bus_idx:bus_idx+2] # XXX alternative way

            # if not model.x_dc == -1:
            #     B_tilde[model.x_ind:model.x_ind+model.nx,[model.x_dc]]=bdcvsc

            n_vsc += 1

        elif model.type == ModelType.GEN_ORD_6:
            bus_idx = model.bus_ind * 2
            idx = inw + bus_idx
            vx = x0[idx]
            vy = x0[idx + 1]
            d = x0[model.x_ind + model.x_idx['d']]

            vd = vx * np.cos(d) + vy * np.sin(d)
            vq = -vx * np.sin(d) + vy * np.cos(d)

            uo = np.array([vd, vq])
            xo = x0[np.arange(model.x_ind, model.x_ind + model.nx)]

            ag, bg, cg, dg = sixth_order_model_avr(xo, uo, model)

            A_tilde[model.x_ind:model.x_ind + model.nx, model.x_ind:model.x_ind + model.nx] = ag
            B_tilde[model.x_ind:model.x_ind + model.nx, idx:idx + 2] = bg
            B_tilde[idx:idx + 2, idx:idx + 2] = bn[bus_idx:bus_idx + 2, bus_idx:bus_idx + 2] @ dg
            C_tilde[idx:idx + 2, model.x_ind:model.x_ind + model.nx] = bn[bus_idx:bus_idx + 2, bus_idx:bus_idx + 2] @ cg

            n_gen += 1

        elif model.type == ModelType.GEN_2_2:

            bus_idx = model.bus_ind * 2
            idx = inw + bus_idx
            vx = x0[idx]
            vy = x0[idx + 1]
            d = x0[model.x_ind + model.x_idx['d']]

            vd = vx * np.cos(d) + vy * np.sin(d)
            vq = -vx * np.sin(d) + vy * np.cos(d)

            uo = np.array([vd, vq])
            xo = x0[np.arange(model.x_ind, model.x_ind + model.nx)]

            # ag,bg,cg,dg = standard_model_linear(xo,uo,model)
            ag, bg, cg, dg = model.abcd_linear(xo, uo)

            A_tilde[model.x_ind:model.x_ind + model.nx, model.x_ind:model.x_ind + model.nx] = ag
            B_tilde[model.x_ind:model.x_ind + model.nx, idx:idx + 2] = bg
            B_tilde[idx:idx + 2, idx:idx + 2] = bn[bus_idx:bus_idx + 2, bus_idx:bus_idx + 2] @ dg
            C_tilde[idx:idx + 2, model.x_ind:model.x_ind + model.nx] = bn[bus_idx:bus_idx + 2, bus_idx:bus_idx + 2] @ cg

            n_gen += 1

        elif model.type == ModelType.DC_LINE:
            # break
            mf = models[model.f]
            mt = models[model.t]
            If = -x0[mf.x_ind + mf.x_idx['Idc']] * mf.Sn / npr.Sb

            It = x0[mt.x_ind + mt.x_idx['Idc']] * mt.Sn / npr.Sb
            It = -If  # TODO check the initialization. In steady state If=-It (if G=0)
            uo = np.array([If, It])
            xo = x0[np.arange(model.x_ind, model.x_ind + model.nx)]

            acb, bcb, ccb, dcb = dc_cable_linear(xo, uo, model)

            A_tilde[model.x_ind:model.x_ind + model.nx, model.x_ind:model.x_ind + model.nx] = acb

            # cvsc[np.array([0,1]),np.array([mf.x_idx['Idc'],mt.x_idx['Idc']])]

            # B_tilde[model.x_ind:model.x_ind+model.nx,idx:idx+2]=bcb
            # B_tilde[idx:idx+2,idx:idx+2]=bn[bus_idx:bus_idx+2,bus_idx:bus_idx+2]@dcb

            cvsc = np.zeros((1, mf.nx))
            cvsc[0, mf.x_idx['Idc']] = 1

            C_tilde[model.x_ind:model.x_ind + model.nx, mf.x_ind:mf.x_ind + mf.nx] = bcb[:, [0]] @ cvsc
            C_tilde[model.x_ind:model.x_ind + model.nx, mt.x_ind:mt.x_ind + mt.nx] = bcb[:, [1]] @ cvsc


        elif model.type == ModelType.VS:
            # break
            bus_idx = model.bus_ind * 2
            idx = inw + bus_idx
            vx = x0[idx]
            vy = x0[idx + 1]
            uo = np.array([vx, vy])
            xo = x0[np.arange(model.x_ind, model.x_ind + model.nx)]
            avs, bvs, cvs, dvs = vs_linear(xo, uo, model)

            A_tilde[model.x_ind:model.x_ind + model.nx, model.x_ind:model.x_ind + model.nx] = avs
            B_tilde[model.x_ind:model.x_ind + model.nx, idx:idx + 2] = bvs
            B_tilde[idx:idx + 2, idx:idx + 2] = bn[bus_idx:bus_idx + 2, bus_idx:bus_idx + 2] @ dvs
            C_tilde[idx:idx + 2, model.x_ind:model.x_ind + model.nx] = bn[bus_idx:bus_idx + 2,
                                                                       bus_idx:bus_idx + 2] @ cvs

    Amat = A_tilde + B_tilde + C_tilde

    return Amat


def calc_eigenvalues_test(x0, npr, models, tol=1e-6):
    inw = npr.x_ind
    # n_vsc = 0
    # n_gen = 0

    size = (len(x0), len(x0))
    Nm = sum([m.nx for m in models])
    Nb = npr.n_bus
    Nbr = npr.n_br
    Nn = 2 * Nb + 2 * Nbr
    A_tilde = np.zeros((Nm, Nm))
    B_tilde = np.zeros((Nm, 2 * Nb))
    C_tilde = np.zeros((2 * Nb, Nm))
    D_tilde = np.zeros((2 * Nb, Nn))

    for model in models:

        if model.type == ModelType.VSC_1:
            # break
            bus_idx = model.bus_ind * 2
            idx = inw + bus_idx
            vx = x0[idx]
            vy = x0[idx + 1]

            if not model.x_dc == -1:
                vdc = x0[model.x_dc]
            else:
                vdc = 1

            Theta_pll = x0[model.x_ind + model.x_idx['Theta']]
            vd = vx * np.cos(Theta_pll) + vy * np.sin(Theta_pll)
            vq = -vx * np.sin(Theta_pll) + vy * np.cos(Theta_pll)

            uo = np.array([vd, vq, vdc])
            xo = x0[np.arange(model.x_ind, model.x_ind + model.nx)]

            # if model.ctrl == CtrlMode.P_Vac:
            #     avsc,bacvsc,bdcvsc,cacvsc,cdcvsc,dvsc = vsc_linear_1(xo,uo,model)
            #     # A,B,Bvdc,C,Cdc,D
            #     # break
            #     # B_tilde[model.x_ind:model.x_ind+model.nx,model.x_dc:model.x_dc+1]=bdcvsc # TODO add DC side
            #     # C_tilde[idx:idx+1,model.x_ind:model.x_ind+model.nx]=bdcvsc
            # elif model.ctrl == CtrlMode.Vdc_Q:
            #     avsc,bacvsc,bdcvsc,cacvsc,cdcvsc,dvsc = vsc_linear_2(xo,uo,model)
            # elif 'Idc' in model.x_idx:
            #     avsc,bacvsc,cacvsc,dvsc = converter_linear(xo,uo,model)
            # else:
            #     # avsc,bvsc,cvsc,dvsc = converter_linear(xo,uo,model)
            am, bm, cm, dm = c_linear0(xo, uo, model)

            A_tilde[model.x_ind:model.x_ind + model.nx, model.x_ind:model.x_ind + model.nx] = am
            B_tilde[model.x_ind:model.x_ind + model.nx, bus_idx:bus_idx + 2] = bm
            C_tilde[bus_idx:bus_idx + 2, model.x_ind:model.x_ind + model.nx] = cm
            D_tilde[bus_idx:bus_idx + 2, bus_idx:bus_idx + 2] = dm

            # print(model,cacvsc@xo)
            # B_tilde[model.x_ind:model.x_ind+model.nx,inw:] = bacvsc@cn[bus_idx:bus_idx+2] # XXX alternative way

            # if not model.x_dc == -1:
            #     B_tilde[model.x_ind:model.x_ind+model.nx,[model.x_dc]]=bdcvsc

            # n_vsc += 1

        elif model.type == ModelType.GEN_ORD_6:
            bus_idx = model.bus_ind * 2
            idx = inw + bus_idx
            vx = x0[idx]
            vy = x0[idx + 1]
            d = x0[model.x_ind + model.x_idx['d']]

            vd = vx * np.cos(d) + vy * np.sin(d)
            vq = -vx * np.sin(d) + vy * np.cos(d)

            uo = np.array([vd, vq])
            xo = x0[np.arange(model.x_ind, model.x_ind + model.nx)]

            am, bm, cm, dm = sixth_order_model_avr(xo, uo, model)

            # A_tilde[model.x_ind:model.x_ind+model.nx,model.x_ind:model.x_ind+model.nx]=ag
            # B_tilde[model.x_ind:model.x_ind+model.nx,idx:idx+2]=bg
            # B_tilde[idx:idx+2,idx:idx+2]=bn[bus_idx:bus_idx+2,bus_idx:bus_idx+2]@dg
            # C_tilde[idx:idx+2,model.x_ind:model.x_ind+model.nx] = bn[bus_idx:bus_idx+2,bus_idx:bus_idx+2]@cg

            A_tilde[model.x_ind:model.x_ind + model.nx, model.x_ind:model.x_ind + model.nx] = am
            B_tilde[model.x_ind:model.x_ind + model.nx, bus_idx:bus_idx + 2] = bm
            C_tilde[bus_idx:bus_idx + 2, model.x_ind:model.x_ind + model.nx] = cm
            D_tilde[bus_idx:bus_idx + 2, bus_idx:bus_idx + 2] = dm

            # n_gen += 1

        elif model.type == ModelType.GEN_2_2:

            bus_idx = model.bus_ind * 2
            idx = inw + bus_idx
            vx = x0[idx]
            vy = x0[idx + 1]
            d = x0[model.x_ind + model.x_idx['d']]

            vd = vx * np.cos(d) + vy * np.sin(d)
            vq = -vx * np.sin(d) + vy * np.cos(d)

            uo = np.array([vd, vq])
            xo = x0[np.arange(model.x_ind, model.x_ind + model.nx)]

            am, bm, cm, dm = standard_model_linear(xo, uo, model)

            # A_tilde[model.x_ind:model.x_ind+model.nx,model.x_ind:model.x_ind+model.nx]=ag
            # B_tilde[model.x_ind:model.x_ind+model.nx,idx:idx+2]=bg
            # B_tilde[idx:idx+2,idx:idx+2]=bn[bus_idx:bus_idx+2,bus_idx:bus_idx+2]@dg
            # C_tilde[idx:idx+2,model.x_ind:model.x_ind+model.nx] = bn[bus_idx:bus_idx+2,bus_idx:bus_idx+2]@cg
            A_tilde[model.x_ind:model.x_ind + model.nx, model.x_ind:model.x_ind + model.nx] = am
            B_tilde[model.x_ind:model.x_ind + model.nx, bus_idx:bus_idx + 2] = bm
            C_tilde[bus_idx:bus_idx + 2, model.x_ind:model.x_ind + model.nx] = cm
            D_tilde[bus_idx:bus_idx + 2, bus_idx:bus_idx + 2] = dm
            # n_gen += 1
            # print(model,cg@xo)

        elif model.type == ModelType.DC_LINE:
            # break
            mf = models[model.f]
            mt = models[model.t]
            If = -x0[mf.x_ind + mf.x_idx['Idc']] * mf.Sn / npr.Sb

            It = x0[mt.x_ind + mt.x_idx['Idc']] * mt.Sn / npr.Sb
            It = -If  # TODO check the initialization. In steady state If=-It (if G=0)
            uo = np.array([If, It])
            xo = x0[np.arange(model.x_ind, model.x_ind + model.nx)]

            am, bm, cm, dm = dc_cable_linear(xo, uo, model)

            A_tilde[model.x_ind:model.x_ind + model.nx, model.x_ind:model.x_ind + model.nx] = am

            # cvsc[np.array([0,1]),np.array([mf.x_idx['Idc'],mt.x_idx['Idc']])]

            # B_tilde[model.x_ind:model.x_ind+model.nx,idx:idx+2]=bcb
            # B_tilde[idx:idx+2,idx:idx+2]=bn[bus_idx:bus_idx+2,bus_idx:bus_idx+2]@dcb

            cm = np.zeros((1, mf.nx))
            cm[0, mf.x_idx['Idc']] = 1

            C_tilde[model.x_ind:model.x_ind + model.nx, mf.x_ind:mf.x_ind + mf.nx] = bm[:, [0]] @ cm
            C_tilde[model.x_ind:model.x_ind + model.nx, mt.x_ind:mt.x_ind + mt.nx] = bm[:, [1]] @ cm


        elif model.type == ModelType.VS:
            # break
            bus_idx = model.bus_ind * 2
            idx = inw + bus_idx
            vx = x0[idx]
            vy = x0[idx + 1]
            uo = np.array([vx, vy])
            xo = x0[np.arange(model.x_ind, model.x_ind + model.nx)]
            am, bm, cm, dm = vs_linear(xo, uo, model)

            # A_tilde[model.x_ind:model.x_ind+model.nx,model.x_ind:model.x_ind+model.nx]=avs
            # B_tilde[model.x_ind:model.x_ind+model.nx,idx:idx+2]=bvs
            # B_tilde[idx:idx+2,idx:idx+2]=bn[bus_idx:bus_idx+2,bus_idx:bus_idx+2]@dvs
            # C_tilde[idx:idx+2,model.x_ind:model.x_ind+model.nx] = bn[bus_idx:bus_idx+2,bus_idx:bus_idx+2]@cvs

            A_tilde[model.x_ind:model.x_ind + model.nx, model.x_ind:model.x_ind + model.nx] = am
            B_tilde[model.x_ind:model.x_ind + model.nx, bus_idx:bus_idx + 2] = bm
            C_tilde[bus_idx:bus_idx + 2, model.x_ind:model.x_ind + model.nx] = cm
            D_tilde[bus_idx:bus_idx + 2, bus_idx:bus_idx + 2] = dm

    an, bn, cn, dn = network_linear(npr)
    Im = np.zeros(2 * npr.n_bus)
    Ib = np.zeros(npr.n_bus, dtype=complex)
    for model in models:
        if model.type == ModelType.gen_standard:
            th = x0[model.x_ind + model.x_idx['d']]
        else:
            th = x0[model.x_ind + model.x_idx['Theta']]

        Id = x0[model.x_ind + model.x_idx['Id']]
        Iq = x0[model.x_ind + model.x_idx['Iq']]
        Ix = (Id * np.cos(th) - Iq * np.sin(th)) * model.Sn / npr.Sb
        Iy = (Id * np.sin(th) + Iq * np.cos(th)) * model.Sn / npr.Sb
        Im[2 * model.bus_ind] += Ix
        Im[2 * model.bus_ind + 1] += Iy

        Ib[model.bus_ind] += Ix + 1j * Iy

    # xm = x0[:inw]
    # xn = x0[inw:]

    # cn@xn+dn@Im
    # cn@xn+dn@(C_tilde@xm+D_tilde@xn)

    # C_tilde@xm-Im

    Amat = np.vstack([np.hstack([A_tilde, B_tilde @ cn]),
                      np.hstack([bn @ C_tilde, an + bn @ D_tilde])])

    return Amat


def nswph_linear(x0, u, npr, off, wf, gpr):
    delta = x0[gpr.x_idx['d']]
    e_dpp = x0[gpr.x_idx['Edpp']]
    e_qpp = x0[gpr.x_idx['Eqpp']]
    Idw = x0[gpr.nx + wf.x_idx['Id']]
    Iqw = x0[gpr.nx + wf.x_idx['Iq']]
    thetaw = x0[gpr.nx + wf.x_idx['Theta']]
    # Mpll = x0[7]
    # ilx = x0[12]
    # ily = x0[13]
    # vd = x0[14]
    # vq = x0[15]
    Ido = x0[gpr.nx + wf.nx + off.x_idx['Id']]
    Iqo = x0[gpr.nx + wf.nx + off.x_idx['Iq']]
    thetao = x0[gpr.nx + wf.nx + off.x_idx['Theta']]

    inw = npr.x_ind
    voffx = x0[inw + 0]
    voffy = x0[inw + 1]
    vscx = x0[inw + 2]
    vscy = x0[inw + 3]
    vtfwx = x0[inw + 4]
    vtfwy = x0[inw + 5]
    vwfx = x0[inw + 6]
    vwfy = x0[inw + 7]

    Csc = 0.01
    Chub = 0.01
    Lcb = 0.002103529614325069
    Rcb = 0.0014730639731404956
    Ccb = 0.4269650611647936 / 2
    Ltsc = 0.15 / 3
    Rtsc = 0.01 / 3

    Ltwf = 0.15 / 8
    Rtwf = 0.01 / 8

    A = np.array([[0, npr.wn, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [(-gpr.ra * ((-2 * gpr.ra * (-vscx * np.sin(delta) + vscy * np.cos(delta)) - 2 * gpr.xqpp * (
                          -vscx * np.cos(delta) - vscy * np.sin(delta))) * (-gpr.ra * (
                          -e_dpp + vscx * np.cos(delta) + vscy * np.sin(delta)) - gpr.xqpp * (
                                                                                    -e_qpp - vscx * np.sin(
                                                                                delta) + vscy * np.cos(
                                                                                delta))) / (
                                       gpr.ra ** 2 + gpr.xdpp * gpr.xqpp) ** 2 + (-2 * gpr.ra * (
                          -vscx * np.cos(delta) - vscy * np.sin(delta)) + 2 * gpr.xdpp * (-vscx * np.sin(
                      delta) + vscy * np.cos(delta))) * (-gpr.ra * (
                          -e_qpp - vscx * np.sin(delta) + vscy * np.cos(delta)) + gpr.xdpp * (
                                                                 -e_dpp + vscx * np.cos(delta) + vscy * np.sin(
                                                             delta))) / (
                                       gpr.ra ** 2 + gpr.xdpp * gpr.xqpp) ** 2) - (
                            -gpr.ra * (-vscx * np.sin(delta) + vscy * np.cos(delta)) - gpr.xqpp * (
                            -vscx * np.cos(delta) - vscy * np.sin(delta))) * (
                            vscx * np.cos(delta) + vscy * np.sin(delta)) / (gpr.ra ** 2 + gpr.xdpp * gpr.xqpp) - (
                            -gpr.ra * (-vscx * np.cos(delta) - vscy * np.sin(delta)) + gpr.xdpp * (
                            -vscx * np.sin(delta) + vscy * np.cos(delta))) * (
                            -vscx * np.sin(delta) + vscy * np.cos(delta)) / (gpr.ra ** 2 + gpr.xdpp * gpr.xqpp) - (
                            -gpr.ra * (-e_dpp + vscx * np.cos(delta) + vscy * np.sin(delta)) - gpr.xqpp * (
                            -e_qpp - vscx * np.sin(delta) + vscy * np.cos(delta))) * (
                            -vscx * np.sin(delta) + vscy * np.cos(delta)) / (gpr.ra ** 2 + gpr.xdpp * gpr.xqpp) - (
                            -gpr.ra * (-e_qpp - vscx * np.sin(delta) + vscy * np.cos(delta)) + gpr.xdpp * (
                            -e_dpp + vscx * np.cos(delta) + vscy * np.sin(delta))) * (
                            -vscx * np.cos(delta) - vscy * np.sin(delta)) / (
                            gpr.ra ** 2 + gpr.xdpp * gpr.xqpp)) / gpr.Tj, -gpr.D / gpr.Tj, 0, (-gpr.ra * (
                          2 * gpr.ra * (
                          -gpr.ra * (-e_qpp - vscx * np.sin(delta) + vscy * np.cos(delta)) + gpr.xdpp * (
                          -e_dpp + vscx * np.cos(delta) + vscy * np.sin(delta))) / (
                                  gpr.ra ** 2 + gpr.xdpp * gpr.xqpp) ** 2 + 2 * gpr.xqpp * (-gpr.ra * (
                          -e_dpp + vscx * np.cos(delta) + vscy * np.sin(delta)) - gpr.xqpp * (
                                                                                                    -e_qpp - vscx * np.sin(
                                                                                                delta) + vscy * np.cos(
                                                                                                delta))) / (
                                  gpr.ra ** 2 + gpr.xdpp * gpr.xqpp) ** 2) - gpr.ra * (-vscx * np.sin(
                      delta) + vscy * np.cos(delta)) / (gpr.ra ** 2 + gpr.xdpp * gpr.xqpp) - gpr.xqpp * (vscx * np.cos(
                      delta) + vscy * np.sin(delta)) / (gpr.ra ** 2 + gpr.xdpp * gpr.xqpp)) / gpr.Tj, 0, (-gpr.ra * (
                          2 * gpr.ra * (
                          -gpr.ra * (-e_dpp + vscx * np.cos(delta) + vscy * np.sin(delta)) - gpr.xqpp * (
                          -e_qpp - vscx * np.sin(delta) + vscy * np.cos(delta))) / (
                                  gpr.ra ** 2 + gpr.xdpp * gpr.xqpp) ** 2 - 2 * gpr.xdpp * (-gpr.ra * (
                          -e_qpp - vscx * np.sin(delta) + vscy * np.cos(delta)) + gpr.xdpp * (
                                                                                                    -e_dpp + vscx * np.cos(
                                                                                                delta) + vscy * np.sin(
                                                                                                delta))) / (
                                  gpr.ra ** 2 + gpr.xdpp * gpr.xqpp) ** 2) - gpr.ra * (vscx * np.cos(
                      delta) + vscy * np.sin(delta)) / (gpr.ra ** 2 + gpr.xdpp * gpr.xqpp) + gpr.xdpp * (-vscx * np.sin(
                      delta) + vscy * np.cos(delta)) / (gpr.ra ** 2 + gpr.xdpp * gpr.xqpp)) / gpr.Tj, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-gpr.ra * ((
                                                                                                                        -gpr.ra * (
                                                                                                                        -e_dpp + vscx * np.cos(
                                                                                                                    delta) + vscy * np.sin(
                                                                                                                    delta)) - gpr.xqpp * (
                                                                                                                                -e_qpp - vscx * np.sin(
                                                                                                                            delta) + vscy * np.cos(
                                                                                                                            delta))) * (
                                                                                                                        -2 * gpr.ra * np.cos(
                                                                                                                    delta) + 2 * gpr.xqpp * np.sin(
                                                                                                                    delta)) / (
                                                                                                                        gpr.ra ** 2 + gpr.xdpp * gpr.xqpp) ** 2 + (
                                                                                                                        -gpr.ra * (
                                                                                                                        -e_qpp - vscx * np.sin(
                                                                                                                    delta) + vscy * np.cos(
                                                                                                                    delta)) + gpr.xdpp * (
                                                                                                                                -e_dpp + vscx * np.cos(
                                                                                                                            delta) + vscy * np.sin(
                                                                                                                            delta))) * (
                                                                                                                        2 * gpr.ra * np.sin(
                                                                                                                    delta) + 2 * gpr.xdpp * np.cos(
                                                                                                                    delta)) / (
                                                                                                                        gpr.ra ** 2 + gpr.xdpp * gpr.xqpp) ** 2) - (
                                                                                                             -gpr.ra * (
                                                                                                             -e_dpp + vscx * np.cos(
                                                                                                         delta) + vscy * np.sin(
                                                                                                         delta)) - gpr.xqpp * (
                                                                                                                     -e_qpp - vscx * np.sin(
                                                                                                                 delta) + vscy * np.cos(
                                                                                                                 delta))) * np.cos(
                      delta) / (gpr.ra ** 2 + gpr.xdpp * gpr.xqpp) + (-gpr.ra * (
                          -e_qpp - vscx * np.sin(delta) + vscy * np.cos(delta)) + gpr.xdpp * (
                                                                              -e_dpp + vscx * np.cos(
                                                                          delta) + vscy * np.sin(delta))) * np.sin(
                      delta) / (gpr.ra ** 2 + gpr.xdpp * gpr.xqpp) - (gpr.ra * np.sin(delta) + gpr.xdpp * np.cos(
                      delta)) * (-vscx * np.sin(delta) + vscy * np.cos(delta)) / (gpr.ra ** 2 + gpr.xdpp * gpr.xqpp) - (
                                                                                                             -gpr.ra * np.cos(
                                                                                                         delta) + gpr.xqpp * np.sin(
                                                                                                         delta)) * (
                                                                                                             vscx * np.cos(
                                                                                                         delta) + vscy * np.sin(
                                                                                                         delta)) / (
                                                                                                             gpr.ra ** 2 + gpr.xdpp * gpr.xqpp)) / gpr.Tj,
                   (-gpr.ra * ((-gpr.ra * (-e_dpp + vscx * np.cos(delta) + vscy * np.sin(delta)) - gpr.xqpp * (
                           -e_qpp - vscx * np.sin(delta) + vscy * np.cos(delta))) * (
                                       -2 * gpr.ra * np.sin(delta) - 2 * gpr.xqpp * np.cos(delta)) / (
                                       gpr.ra ** 2 + gpr.xdpp * gpr.xqpp) ** 2 + (-gpr.ra * (
                           -e_qpp - vscx * np.sin(delta) + vscy * np.cos(delta)) + gpr.xdpp * (
                                                                                          -e_dpp + vscx * np.cos(
                                                                                      delta) + vscy * np.sin(
                                                                                      delta))) * (
                                       -2 * gpr.ra * np.cos(delta) + 2 * gpr.xdpp * np.sin(delta)) / (
                                       gpr.ra ** 2 + gpr.xdpp * gpr.xqpp) ** 2) - (
                            -gpr.ra * (-e_dpp + vscx * np.cos(delta) + vscy * np.sin(delta)) - gpr.xqpp * (
                            -e_qpp - vscx * np.sin(delta) + vscy * np.cos(delta))) * np.sin(delta) / (
                            gpr.ra ** 2 + gpr.xdpp * gpr.xqpp) - (
                            -gpr.ra * (-e_qpp - vscx * np.sin(delta) + vscy * np.cos(delta)) + gpr.xdpp * (
                            -e_dpp + vscx * np.cos(delta) + vscy * np.sin(delta))) * np.cos(delta) / (
                            gpr.ra ** 2 + gpr.xdpp * gpr.xqpp) - (
                            -gpr.ra * np.sin(delta) - gpr.xqpp * np.cos(delta)) * (
                            vscx * np.cos(delta) + vscy * np.sin(delta)) / (gpr.ra ** 2 + gpr.xdpp * gpr.xqpp) - (
                            -gpr.ra * np.cos(delta) + gpr.xdpp * np.sin(delta)) * (
                            -vscx * np.sin(delta) + vscy * np.cos(delta)) / (
                            gpr.ra ** 2 + gpr.xdpp * gpr.xqpp)) / gpr.Tj, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, -gpr.kd / gpr.Tdp, (gpr.kd - 1) / gpr.Tdp, 0, 0, 1 / gpr.Tdp, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [-(gpr.xdp - gpr.xdpp) * (-gpr.ra * (-vscx * np.sin(delta) + vscy * np.cos(delta)) - gpr.xqpp * (
                          -vscx * np.cos(delta) - vscy * np.sin(delta))) / (
                           gpr.Tdpp * (gpr.ra ** 2 + gpr.xdpp * gpr.xqpp)), 0, 1 / gpr.Tdpp,
                   (-gpr.xqpp * (gpr.xdp - gpr.xdpp) / (gpr.ra ** 2 + gpr.xdpp * gpr.xqpp) - 1) / gpr.Tdpp, 0,
                   -gpr.ra * (gpr.xdp - gpr.xdpp) / (gpr.Tdpp * (gpr.ra ** 2 + gpr.xdpp * gpr.xqpp)), 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   -(gpr.xdp - gpr.xdpp) * (-gpr.ra * np.cos(delta) + gpr.xqpp * np.sin(delta)) / (
                           gpr.Tdpp * (gpr.ra ** 2 + gpr.xdpp * gpr.xqpp)),
                   -(gpr.xdp - gpr.xdpp) * (-gpr.ra * np.sin(delta) - gpr.xqpp * np.cos(delta)) / (
                           gpr.Tdpp * (gpr.ra ** 2 + gpr.xdpp * gpr.xqpp)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, -gpr.kq / gpr.Tqp, (gpr.kq - 1) / gpr.Tqp, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [(gpr.xqp - gpr.xqpp) * (-gpr.ra * (-vscx * np.cos(delta) - vscy * np.sin(delta)) + gpr.xdpp * (
                          -vscx * np.sin(delta) + vscy * np.cos(delta))) / (
                           gpr.Tqpp * (gpr.ra ** 2 + gpr.xdpp * gpr.xqpp)), 0, 0,
                   gpr.ra * (gpr.xqp - gpr.xqpp) / (gpr.Tqpp * (gpr.ra ** 2 + gpr.xdpp * gpr.xqpp)), 1 / gpr.Tqpp,
                   (-gpr.xdpp * (gpr.xqp - gpr.xqpp) / (gpr.ra ** 2 + gpr.xdpp * gpr.xqpp) - 1) / gpr.Tqpp, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   (gpr.xqp - gpr.xqpp) * (gpr.ra * np.sin(delta) + gpr.xdpp * np.cos(delta)) / (
                           gpr.Tqpp * (gpr.ra ** 2 + gpr.xdpp * gpr.xqpp)),
                   (gpr.xqp - gpr.xqpp) * (-gpr.ra * np.cos(delta) + gpr.xdpp * np.sin(delta)) / (
                           gpr.Tqpp * (gpr.ra ** 2 + gpr.xdpp * gpr.xqpp)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, -1 / gpr.Te, -gpr.Kc / gpr.Te, gpr.Kc / (gpr.Tc * gpr.Te), 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0],
                  [0, 0, 0, 0, 0, 0, 0, -1 / gpr.Tm, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, vscx / (gpr.Tm * np.sqrt(vscx ** 2 + vscy ** 2)),
                   vscy / (gpr.Tm * np.sqrt(vscx ** 2 + vscy ** 2)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, npr.wn * (-wf.Kpc - wf.Rt) / wf.Lt, 0, npr.wn * wf.Kic / wf.Lt, 0,
                   npr.wn / wf.Lt, 0, npr.wn * (
                           vwfx * np.sin(thetaw) - vwfy * np.cos(thetaw) - wf.Kp_pll * wf.Kpc * wf.Kpf * wf.Kpp * (
                           -vwfx * np.cos(thetaw) - vwfy * np.sin(thetaw))) / wf.Lt,
                   -npr.wn * wf.Ki_pll * wf.Kpc * wf.Kpf * wf.Kpp / wf.Lt, npr.wn * wf.Kif * wf.Kpc * wf.Kpp / wf.Lt,
                   npr.wn * wf.Kip * wf.Kpc / wf.Lt, 0, -npr.wn * wf.Kpc * wf.Kpp / wf.Lt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   npr.wn * (wf.Kp_pll * wf.Kpc * wf.Kpf * wf.Kpp * np.sin(thetaw) - np.cos(thetaw)) / wf.Lt,
                   npr.wn * (-wf.Kp_pll * wf.Kpc * wf.Kpf * wf.Kpp * np.cos(thetaw) - np.sin(thetaw)) / wf.Lt, 0, 0, 0,
                   0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, npr.wn * (-wf.Kpc - wf.Rt) / wf.Lt, 0, npr.wn * wf.Kic / wf.Lt, 0,
                   npr.wn / wf.Lt, npr.wn * (vwfx * np.cos(thetaw) + vwfy * np.sin(thetaw)) / wf.Lt, 0, 0, 0,
                   npr.wn * wf.Kiq * wf.Kpc / wf.Lt, 0, npr.wn * wf.Kpc * wf.Kpq * wf.Kq / wf.Lt,
                   npr.wn * wf.Kpc * wf.Kpq * wf.Kv / wf.Lt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   npr.wn * np.sin(thetaw) / wf.Lt, -npr.wn * np.cos(thetaw) / wf.Lt, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
                   -wf.Kp_pll * wf.Kpf * wf.Kpp * (-vwfx * np.cos(thetaw) - vwfy * np.sin(thetaw)),
                   -wf.Ki_pll * wf.Kpf * wf.Kpp, wf.Kif * wf.Kpp, wf.Kip, 0, -wf.Kpp, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, wf.Kp_pll * wf.Kpf * wf.Kpp * np.sin(thetaw),
                   -wf.Kp_pll * wf.Kpf * wf.Kpp * np.cos(thetaw), 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, wf.Kiq, 0, wf.Kpq * wf.Kq, wf.Kpq * wf.Kv,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1 / wf.Tad, 0,
                   (-vwfx * np.sin(thetaw) + vwfy * np.cos(thetaw)) / wf.Tad, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.cos(thetaw) / wf.Tad, np.sin(thetaw) / wf.Tad, 0, 0, 0, 0,
                   0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1 / wf.Tad,
                   (-vwfx * np.cos(thetaw) - vwfy * np.sin(thetaw)) / wf.Tad, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -np.sin(thetaw) / wf.Tad, np.cos(thetaw) / wf.Tad, 0, 0, 0, 0,
                   0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   npr.wn * wf.Kp_pll * (-vwfx * np.cos(thetaw) - vwfy * np.sin(thetaw)), npr.wn * wf.Ki_pll, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   -npr.wn * wf.Kp_pll * np.sin(thetaw), npr.wn * wf.Kp_pll * np.cos(thetaw), 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -vwfx * np.cos(thetaw) - vwfy * np.sin(thetaw), 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -np.sin(thetaw),
                   np.cos(thetaw), 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   -wf.Kp_pll * (-vwfx * np.cos(thetaw) - vwfy * np.sin(thetaw)), -wf.Ki_pll, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, wf.Kp_pll * np.sin(thetaw),
                   -wf.Kp_pll * np.cos(thetaw), 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   -wf.Kp_pll * wf.Kpf * (-vwfx * np.cos(thetaw) - vwfy * np.sin(thetaw)), -wf.Ki_pll * wf.Kpf, wf.Kif,
                   0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   wf.Kp_pll * wf.Kpf * np.sin(thetaw), -wf.Kp_pll * wf.Kpf * np.cos(thetaw), 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, wf.Kq, wf.Kv, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, (vwfx * np.cos(thetaw) + vwfy * np.sin(thetaw)) / wf.Tpm,
                   (-vwfx * np.sin(thetaw) + vwfy * np.cos(thetaw)) / wf.Tpm, 0, 0, 0, 0, (
                           Idw * (-vwfx * np.sin(thetaw) + vwfy * np.cos(thetaw)) + Iqw * (
                           -vwfx * np.cos(thetaw) - vwfy * np.sin(thetaw))) / wf.Tpm, 0, 0, 0, 0, -1 / wf.Tpm,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   (Idw * np.cos(thetaw) - Iqw * np.sin(thetaw)) / wf.Tpm,
                   (Idw * np.sin(thetaw) + Iqw * np.cos(thetaw)) / wf.Tpm, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, (-vwfx * np.sin(thetaw) + vwfy * np.cos(thetaw)) / wf.Tpm,
                   (-vwfx * np.cos(thetaw) - vwfy * np.sin(thetaw)) / wf.Tpm, 0, 0, 0, 0, (
                           Idw * (-vwfx * np.cos(thetaw) - vwfy * np.sin(thetaw)) - Iqw * (
                           -vwfx * np.sin(thetaw) + vwfy * np.cos(thetaw))) / wf.Tpm, 0, 0, 0, 0, 0,
                   -1 / wf.Tpm, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   (-Idw * np.sin(thetaw) - Iqw * np.cos(thetaw)) / wf.Tpm,
                   (Idw * np.cos(thetaw) - Iqw * np.sin(thetaw)) / wf.Tpm, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (
                          (-2 * vwfx * np.sin(thetaw) + 2 * vwfy * np.cos(thetaw)) * (
                          vwfx * np.cos(thetaw) + vwfy * np.sin(thetaw)) / 2 + (
                                  -vwfx * np.sin(thetaw) + vwfy * np.cos(thetaw)) * (
                                  -2 * vwfx * np.cos(thetaw) - 2 * vwfy * np.sin(thetaw)) / 2) / (
                           wf.Tvm * np.sqrt((-vwfx * np.sin(thetaw) + vwfy * np.cos(thetaw)) ** 2 + (
                           vwfx * np.cos(thetaw) + vwfy * np.sin(thetaw)) ** 2)), 0, 0, 0, 0, 0, 0, -1 / wf.Tvm,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (
                           -(-vwfx * np.sin(thetaw) + vwfy * np.cos(thetaw)) * np.sin(thetaw) + (
                           vwfx * np.cos(thetaw) + vwfy * np.sin(thetaw)) * np.cos(thetaw)) / (wf.Tvm * np.sqrt(
                      (-vwfx * np.sin(thetaw) + vwfy * np.cos(thetaw)) ** 2 + (
                              vwfx * np.cos(thetaw) + vwfy * np.sin(thetaw)) ** 2)), (
                           (-vwfx * np.sin(thetaw) + vwfy * np.cos(thetaw)) * np.cos(thetaw) + (
                           vwfx * np.cos(thetaw) + vwfy * np.sin(thetaw)) * np.sin(thetaw)) / (wf.Tvm * np.sqrt(
                      (-vwfx * np.sin(thetaw) + vwfy * np.cos(thetaw)) ** 2 + (
                              vwfx * np.cos(thetaw) + vwfy * np.sin(thetaw)) ** 2)), 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   npr.wn * (-off.Kpc - off.Rt) / off.Lt, 0, npr.wn * off.Kic / off.Lt, 0, npr.wn / off.Lt, 0,
                   npr.wn * (-off.Kp_pll * off.Kpc * off.Kpf * off.Kpp * (
                           -voffx * np.cos(thetao) - voffy * np.sin(thetao)) + voffx * np.sin(
                       thetao) - voffy * np.cos(thetao)) / off.Lt,
                   -npr.wn * off.Ki_pll * off.Kpc * off.Kpf * off.Kpp / off.Lt,
                   npr.wn * off.Kif * off.Kpc * off.Kpp / off.Lt, npr.wn * off.Kip * off.Kpc / off.Lt, 0,
                   -npr.wn * off.Kpc * off.Kpp / off.Lt, 0, 0,
                   npr.wn * (off.Kp_pll * off.Kpc * off.Kpf * off.Kpp * np.sin(thetao) - np.cos(thetao)) / off.Lt,
                   npr.wn * (-off.Kp_pll * off.Kpc * off.Kpf * off.Kpp * np.cos(thetao) - np.sin(thetao)) / off.Lt, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   npr.wn * (-off.Kpc - off.Rt) / off.Lt, 0, npr.wn * off.Kic / off.Lt, 0, npr.wn / off.Lt,
                   npr.wn * (voffx * np.cos(thetao) + voffy * np.sin(thetao)) / off.Lt, 0, 0, 0,
                   npr.wn * off.Kiq * off.Kpc / off.Lt, 0, npr.wn * off.Kpc * off.Kpq * off.Kq / off.Lt,
                   npr.wn * off.Kpc * off.Kpq * off.Kv / off.Lt, npr.wn * np.sin(thetao) / off.Lt,
                   -npr.wn * np.cos(thetao) / off.Lt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
                   -off.Kp_pll * off.Kpf * off.Kpp * (-voffx * np.cos(thetao) - voffy * np.sin(thetao)),
                   -off.Ki_pll * off.Kpf * off.Kpp, off.Kif * off.Kpp, off.Kip, 0, -off.Kpp, 0, 0,
                   off.Kp_pll * off.Kpf * off.Kpp * np.sin(thetao), -off.Kp_pll * off.Kpf * off.Kpp * np.cos(thetao), 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0,
                   off.Kiq, 0, off.Kpq * off.Kq, off.Kpq * off.Kv, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1 / off.Tad, 0,
                   (-voffx * np.sin(thetao) + voffy * np.cos(thetao)) / off.Tad, 0, 0, 0, 0, 0, 0, 0,
                   np.cos(thetao) / off.Tad, np.sin(thetao) / off.Tad, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1 / off.Tad,
                   (-voffx * np.cos(thetao) - voffy * np.sin(thetao)) / off.Tad, 0, 0, 0, 0, 0, 0, 0,
                   -np.sin(thetao) / off.Tad, np.cos(thetao) / off.Tad, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   npr.wn * off.Kp_pll * (-voffx * np.cos(thetao) - voffy * np.sin(thetao)), npr.wn * off.Ki_pll, 0, 0,
                   0, 0, 0, 0, -npr.wn * off.Kp_pll * np.sin(thetao), npr.wn * off.Kp_pll * np.cos(thetao), 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   -voffx * np.cos(thetao) - voffy * np.sin(thetao), 0, 0, 0, 0, 0, 0, 0, -np.sin(thetao),
                   np.cos(thetao), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   -off.Kp_pll * (-voffx * np.cos(thetao) - voffy * np.sin(thetao)), -off.Ki_pll, 0, 0, 0, 0, 0, 0,
                   off.Kp_pll * np.sin(thetao), -off.Kp_pll * np.cos(thetao), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   -off.Kp_pll * off.Kpf * (-voffx * np.cos(thetao) - voffy * np.sin(thetao)), -off.Ki_pll * off.Kpf,
                   off.Kif, 0, 0, -1, 0, 0, off.Kp_pll * off.Kpf * np.sin(thetao),
                   -off.Kp_pll * off.Kpf * np.cos(thetao), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, off.Kq, off.Kv, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   (voffx * np.cos(thetao) + voffy * np.sin(thetao)) / off.Tpm,
                   (-voffx * np.sin(thetao) + voffy * np.cos(thetao)) / off.Tpm, 0, 0, 0, 0, (
                           Ido * (-voffx * np.sin(thetao) + voffy * np.cos(thetao)) + Iqo * (
                           -voffx * np.cos(thetao) - voffy * np.sin(thetao))) / off.Tpm, 0, 0, 0, 0,
                   -1 / off.Tpm, 0, 0, (Ido * np.cos(thetao) - Iqo * np.sin(thetao)) / off.Tpm,
                   (Ido * np.sin(thetao) + Iqo * np.cos(thetao)) / off.Tpm, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   (-voffx * np.sin(thetao) + voffy * np.cos(thetao)) / off.Tpm,
                   (-voffx * np.cos(thetao) - voffy * np.sin(thetao)) / off.Tpm, 0, 0, 0, 0, (
                           Ido * (-voffx * np.cos(thetao) - voffy * np.sin(thetao)) - Iqo * (
                           -voffx * np.sin(thetao) + voffy * np.cos(thetao))) / off.Tpm, 0, 0, 0, 0, 0,
                   -1 / off.Tpm, 0, (-Ido * np.sin(thetao) - Iqo * np.cos(thetao)) / off.Tpm,
                   (Ido * np.cos(thetao) - Iqo * np.sin(thetao)) / off.Tpm, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (
                          (-2 * voffx * np.sin(thetao) + 2 * voffy * np.cos(thetao)) * (
                          voffx * np.cos(thetao) + voffy * np.sin(thetao)) / 2 + (
                                  -voffx * np.sin(thetao) + voffy * np.cos(thetao)) * (
                                  -2 * voffx * np.cos(thetao) - 2 * voffy * np.sin(thetao)) / 2) / (
                           off.Tvm * np.sqrt((-voffx * np.sin(thetao) + voffy * np.cos(thetao)) ** 2 + (
                           voffx * np.cos(thetao) + voffy * np.sin(thetao)) ** 2)), 0, 0, 0, 0, 0, 0,
                   -1 / off.Tvm, (-(-voffx * np.sin(thetao) + voffy * np.cos(thetao)) * np.sin(thetao) + (
                          voffx * np.cos(thetao) + voffy * np.sin(thetao)) * np.cos(thetao)) / (off.Tvm * np.sqrt(
                      (-voffx * np.sin(thetao) + voffy * np.cos(thetao)) ** 2 + (
                              voffx * np.cos(thetao) + voffy * np.sin(thetao)) ** 2)), (
                           (-voffx * np.sin(thetao) + voffy * np.cos(thetao)) * np.cos(thetao) + (
                           voffx * np.cos(thetao) + voffy * np.sin(thetao)) * np.sin(thetao)) / (
                           off.Tvm * np.sqrt((-voffx * np.sin(thetao) + voffy * np.cos(thetao)) ** 2 + (
                           voffx * np.cos(thetao) + voffy * np.sin(thetao)) ** 2)), 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   npr.wn * off.Sn * np.cos(thetao) / (Chub * npr.Sb),
                   -npr.wn * off.Sn * np.sin(thetao) / (Chub * npr.Sb), 0, 0, 0, 0,
                   npr.wn * off.Sn * (-Ido * np.sin(thetao) - Iqo * np.cos(thetao)) / (Chub * npr.Sb), 0, 0, 0, 0, 0, 0,
                   0, 0, npr.wn, 0, 0, 0, 0, 0, 0, npr.wn / Chub, 0, npr.wn / Chub, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   npr.wn * off.Sn * np.sin(thetao) / (Chub * npr.Sb),
                   npr.wn * off.Sn * np.cos(thetao) / (Chub * npr.Sb), 0, 0, 0, 0,
                   npr.wn * off.Sn * (Ido * np.cos(thetao) - Iqo * np.sin(thetao)) / (Chub * npr.Sb), 0, 0, 0, 0, 0, 0,
                   0, -npr.wn, 0, 0, 0, 0, 0, 0, 0, 0, npr.wn / Chub, 0, npr.wn / Chub, 0, 0],
                  [gpr.Sn * npr.wn * ((-gpr.ra * (-vscx * np.sin(delta) + vscy * np.cos(delta)) - gpr.xqpp * (
                          -vscx * np.cos(delta) - vscy * np.sin(delta))) * np.cos(delta) / (
                                              gpr.ra ** 2 + gpr.xdpp * gpr.xqpp) - (-gpr.ra * (
                          -vscx * np.cos(delta) - vscy * np.sin(delta)) + gpr.xdpp * (-vscx * np.sin(
                      delta) + vscy * np.cos(delta))) * np.sin(delta) / (gpr.ra ** 2 + gpr.xdpp * gpr.xqpp) - (
                                              -gpr.ra * (-e_dpp + vscx * np.cos(delta) + vscy * np.sin(
                                          delta)) - gpr.xqpp * (-e_qpp - vscx * np.sin(delta) + vscy * np.cos(
                                          delta))) * np.sin(delta) / (gpr.ra ** 2 + gpr.xdpp * gpr.xqpp) - (
                                              -gpr.ra * (-e_qpp - vscx * np.sin(delta) + vscy * np.cos(
                                          delta)) + gpr.xdpp * (-e_dpp + vscx * np.cos(delta) + vscy * np.sin(
                                          delta))) * np.cos(delta) / (gpr.ra ** 2 + gpr.xdpp * gpr.xqpp)) / (
                           Csc * npr.Sb), 0, 0, gpr.Sn * npr.wn * (
                           -gpr.ra * np.sin(delta) / (gpr.ra ** 2 + gpr.xdpp * gpr.xqpp) + gpr.xqpp * np.cos(
                       delta) / (gpr.ra ** 2 + gpr.xdpp * gpr.xqpp)) / (Csc * npr.Sb), 0, gpr.Sn * npr.wn * (
                           gpr.ra * np.cos(delta) / (gpr.ra ** 2 + gpr.xdpp * gpr.xqpp) + gpr.xdpp * np.sin(
                       delta) / (gpr.ra ** 2 + gpr.xdpp * gpr.xqpp)) / (Csc * npr.Sb), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, gpr.Sn * npr.wn * (
                           -(gpr.ra * np.sin(delta) + gpr.xdpp * np.cos(delta)) * np.sin(delta) / (
                           gpr.ra ** 2 + gpr.xdpp * gpr.xqpp) + (
                                   -gpr.ra * np.cos(delta) + gpr.xqpp * np.sin(delta)) * np.cos(delta) / (
                                   gpr.ra ** 2 + gpr.xdpp * gpr.xqpp)) / (Csc * npr.Sb), npr.wn * (gpr.Sn * (
                          (-gpr.ra * np.sin(delta) - gpr.xqpp * np.cos(delta)) * np.cos(delta) / (
                          gpr.ra ** 2 + gpr.xdpp * gpr.xqpp) - (
                                  -gpr.ra * np.cos(delta) + gpr.xdpp * np.sin(delta)) * np.sin(delta) / (
                                  gpr.ra ** 2 + gpr.xdpp * gpr.xqpp)) / npr.Sb + Csc) / Csc, 0, 0, 0, 0,
                   -npr.wn / Csc, 0, 0, 0, 0, 0],
                  [gpr.Sn * npr.wn * ((-gpr.ra * (-vscx * np.sin(delta) + vscy * np.cos(delta)) - gpr.xqpp * (
                          -vscx * np.cos(delta) - vscy * np.sin(delta))) * np.sin(delta) / (
                                              gpr.ra ** 2 + gpr.xdpp * gpr.xqpp) + (-gpr.ra * (
                          -vscx * np.cos(delta) - vscy * np.sin(delta)) + gpr.xdpp * (-vscx * np.sin(
                      delta) + vscy * np.cos(delta))) * np.cos(delta) / (gpr.ra ** 2 + gpr.xdpp * gpr.xqpp) + (
                                              -gpr.ra * (-e_dpp + vscx * np.cos(delta) + vscy * np.sin(
                                          delta)) - gpr.xqpp * (-e_qpp - vscx * np.sin(delta) + vscy * np.cos(
                                          delta))) * np.cos(delta) / (gpr.ra ** 2 + gpr.xdpp * gpr.xqpp) - (
                                              -gpr.ra * (-e_qpp - vscx * np.sin(delta) + vscy * np.cos(
                                          delta)) + gpr.xdpp * (-e_dpp + vscx * np.cos(delta) + vscy * np.sin(
                                          delta))) * np.sin(delta) / (gpr.ra ** 2 + gpr.xdpp * gpr.xqpp)) / (
                           Csc * npr.Sb), 0, 0, gpr.Sn * npr.wn * (
                           gpr.ra * np.cos(delta) / (gpr.ra ** 2 + gpr.xdpp * gpr.xqpp) + gpr.xqpp * np.sin(
                       delta) / (gpr.ra ** 2 + gpr.xdpp * gpr.xqpp)) / (Csc * npr.Sb), 0, gpr.Sn * npr.wn * (
                           gpr.ra * np.sin(delta) / (gpr.ra ** 2 + gpr.xdpp * gpr.xqpp) - gpr.xdpp * np.cos(
                       delta) / (gpr.ra ** 2 + gpr.xdpp * gpr.xqpp)) / (Csc * npr.Sb), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, npr.wn * (gpr.Sn * (
                          (gpr.ra * np.sin(delta) + gpr.xdpp * np.cos(delta)) * np.cos(delta) / (
                          gpr.ra ** 2 + gpr.xdpp * gpr.xqpp) + (
                                  -gpr.ra * np.cos(delta) + gpr.xqpp * np.sin(delta)) * np.sin(delta) / (
                                  gpr.ra ** 2 + gpr.xdpp * gpr.xqpp)) / npr.Sb - Csc) / Csc, gpr.Sn * npr.wn * (
                           (-gpr.ra * np.sin(delta) - gpr.xqpp * np.cos(delta)) * np.sin(delta) / (
                           gpr.ra ** 2 + gpr.xdpp * gpr.xqpp) + (
                                   -gpr.ra * np.cos(delta) + gpr.xdpp * np.sin(delta)) * np.cos(delta) / (
                                   gpr.ra ** 2 + gpr.xdpp * gpr.xqpp)) / (Csc * npr.Sb), 0, 0, 0, 0, 0,
                   -npr.wn / Csc, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, npr.wn, 0, 0, 0, 0, -npr.wn / Ccb, 0, npr.wn / Ccb, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, -npr.wn, 0, 0, 0, 0, 0, 0, -npr.wn / Ccb, 0, npr.wn / Ccb],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, npr.wn * wf.Sn * np.cos(thetaw) / (Ccb * npr.Sb),
                   -npr.wn * wf.Sn * np.sin(thetaw) / (Ccb * npr.Sb), 0, 0, 0, 0,
                   npr.wn * wf.Sn * (-Idw * np.sin(thetaw) - Iqw * np.cos(thetaw)) / (Ccb * npr.Sb), 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, npr.wn, 0, 0, 0, 0, -npr.wn / Ccb,
                   0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, npr.wn * wf.Sn * np.sin(thetaw) / (Ccb * npr.Sb),
                   npr.wn * wf.Sn * np.cos(thetaw) / (Ccb * npr.Sb), 0, 0, 0, 0,
                   npr.wn * wf.Sn * (Idw * np.cos(thetaw) - Iqw * np.sin(thetaw)) / (Ccb * npr.Sb), 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -npr.wn, 0, 0, 0, 0, 0, 0,
                   -npr.wn / Ccb],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, -npr.wn / Ltsc, 0, npr.wn / Ltsc, 0, 0, 0, 0, 0, -Rtsc * npr.wn / Ltsc, npr.wn, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, -npr.wn / Ltsc, 0, npr.wn / Ltsc, 0, 0, 0, 0, -npr.wn, -Rtsc * npr.wn / Ltsc, 0, 0, 0,
                   0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, -npr.wn / Ltwf, 0, 0, 0, npr.wn / Ltwf, 0, 0, 0, 0, 0, -Rtwf * npr.wn / Ltwf, npr.wn, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, -npr.wn / Ltwf, 0, 0, 0, npr.wn / Ltwf, 0, 0, 0, 0, -npr.wn, -Rtwf * npr.wn / Ltwf, 0,
                   0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, -npr.wn / Lcb, 0, npr.wn / Lcb, 0, 0, 0, 0, 0, -Rcb * npr.wn / Lcb, npr.wn],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, -npr.wn / Lcb, 0, npr.wn / Lcb, 0, 0, 0, 0, -npr.wn, -Rcb * npr.wn / Lcb],
                  ])
    # A[6,:] = A[6,:]*0
    # A[7,:] = A[7,:]*0
    # A[8,:] = A[8,:]*0
    B = np.array([[0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [npr.wn * wf.Kpc * wf.Kpp / wf.Lt, 0],
                  [0, -npr.wn * wf.Kpc * wf.Kpq * wf.Kq / wf.Lt],
                  [wf.Kpp, 0],
                  [0, -wf.Kpq * wf.Kq],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [1, 0],
                  [0, -wf.Kq],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  ])

    C = np.zeros((2, 51))

    D = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])

    Amat = A  # -B@np.linalg.inv(D)@C

    lambda_1, Phi_1 = np.linalg.eig(Amat)

    lambda_1[np.where(abs(lambda_1) < 1e-7)] = 0

    return lambda_1, Amat, A, B, C, D


def nswph_linear3(x0, u, npr, off, wf, gpr):
    delta = x0[gpr.x_idx['d']]
    w = x0[gpr.x_idx['w']]
    psi_d = x0[gpr.x_idx['psi_d']]
    psi_q = x0[gpr.x_idx['psi_q']]
    psi_fd = x0[gpr.x_idx['psi_fd']]
    psi_1d = x0[gpr.x_idx['psi_1d']]
    psi_1q = x0[gpr.x_idx['psi_1q']]
    psi_2q = x0[gpr.x_idx['psi_2q']]
    i_dg = x0[gpr.x_idx['Id']]
    i_qg = x0[gpr.x_idx['Iq']]

    Idw = x0[gpr.nx + wf.x_idx['Id']]
    Iqw = x0[gpr.nx + wf.x_idx['Iq']]
    thetaw = x0[gpr.nx + wf.x_idx['Theta']]
    # Mpll = x0[7]
    # ilx = x0[12]
    # ily = x0[13]
    # vd = x0[14]
    # vq = x0[15]
    Ido = x0[gpr.nx + wf.nx + off.x_idx['Id']]
    Iqo = x0[gpr.nx + wf.nx + off.x_idx['Iq']]
    thetao = x0[gpr.nx + wf.nx + off.x_idx['Theta']]

    inw = gpr.nx + wf.nx + off.nx
    voffx = x0[inw + 0]
    voffy = x0[inw + 1]
    vscx = x0[inw + 2]
    vscy = x0[inw + 3]
    vx2 = x0[inw + 4]
    vy2 = x0[inw + 5]
    vwfx = x0[inw + 6]
    vwfy = x0[inw + 7]
    itscx = x0[inw + 8]
    itscy = x0[inw + 9]
    itwfx = x0[inw + 10]
    itwfy = x0[inw + 11]
    icbx = x0[inw + 12]
    icby = x0[inw + 13]

    Csc = 0.01
    Chub = 0.01
    Lcb = 0.002103529614325069
    Rcb = 0.0014730639731404956
    Ccb = 0.4269650611647936 / 2
    Ltsc = 0.15 / 3
    Rtsc = 0.01 / 3

    Ltwf = 0.15 / 8
    Rtwf = 0.01 / 8

    A = np.array([[0, npr.wn, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, (-gpr.dkd - gpr.dpe / w + gpr.dpe * (w - 1) / w ** 2) / gpr.Tj, -i_qg / (gpr.Tj * gpr.cosn),
                   i_dg / (gpr.Tj * gpr.cosn), 0, 0, 0, 0, psi_q / (gpr.Tj * gpr.cosn), -psi_d / (gpr.Tj * gpr.cosn), 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [npr.wn * (-vscx * np.sin(delta) + vscy * np.cos(delta)), npr.wn * psi_q, 0, npr.wn * w, 0, 0, 0, 0,
                   gpr.ra * npr.wn, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, npr.wn * np.cos(delta), npr.wn * np.sin(delta), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [npr.wn * (-vscx * np.cos(delta) - vscy * np.sin(delta)), -npr.wn * psi_d, -npr.wn * w, 0, 0, 0, 0, 0,
                   0, gpr.ra * npr.wn, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, -npr.wn * np.sin(delta), npr.wn * np.cos(delta), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, -gpr.rfd * gpr.x1d_loop * npr.wn / gpr.xdet_d,
                   -gpr.rfd * npr.wn * (-gpr.xad - gpr.xrld) / gpr.xdet_d, 0, 0, -gpr.kfd * gpr.rfd * npr.wn, 0,
                   gpr.rfd * npr.wn / gpr.xadu, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, -gpr.r1d * npr.wn * (-gpr.xad - gpr.xrld) / gpr.xdet_d,
                   -gpr.r1d * gpr.xfd_loop * npr.wn / gpr.xdet_d, 0, 0, -gpr.k1d * gpr.r1d * npr.wn, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, -gpr.r1q * gpr.x2q_loop * npr.wn / gpr.xdet_q,
                   -gpr.r1q * npr.wn * (-gpr.xaq - gpr.xrlq) / gpr.xdet_q, 0, -gpr.k1q * gpr.r1q * npr.wn, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, -gpr.r2q * npr.wn * (-gpr.xaq - gpr.xrlq) / gpr.xdet_q,
                   -gpr.r2q * gpr.x1q_loop * npr.wn / gpr.xdet_q, 0, -gpr.k2q * gpr.r2q * npr.wn, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0],
                  [npr.wn * (vscx * np.sin(delta) - vscy * np.cos(delta)) / gpr.xdpp,
                   npr.wn * (-gpr.k1q * psi_1q - gpr.k2q * psi_2q + gpr.xqpp * i_qg) / gpr.xdpp, 0, 0, npr.wn * (
                           -gpr.k1d * gpr.r1d * (
                           -gpr.xad - gpr.xrld) / gpr.xdet_d - gpr.kfd * gpr.rfd * gpr.x1d_loop / gpr.xdet_d) / gpr.xdpp,
                   npr.wn * (-gpr.k1d * gpr.r1d * gpr.xfd_loop / gpr.xdet_d - gpr.kfd * gpr.rfd * (
                           -gpr.xad - gpr.xrld) / gpr.xdet_d) / gpr.xdpp, -gpr.k1q * npr.wn * w / gpr.xdpp,
                   -gpr.k2q * npr.wn * w / gpr.xdpp,
                   npr.wn * (-gpr.k1d ** 2 * gpr.r1d - gpr.kfd ** 2 * gpr.rfd - gpr.ra) / gpr.xdpp,
                   gpr.xqpp * npr.wn * w / gpr.xdpp, gpr.kfd * gpr.rfd * npr.wn / (gpr.xadu * gpr.xdpp), 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   -npr.wn * np.cos(delta) / gpr.xdpp, -npr.wn * np.sin(delta) / gpr.xdpp, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0],
                  [npr.wn * (vscx * np.cos(delta) + vscy * np.sin(delta)) / gpr.xqpp,
                   npr.wn * (gpr.k1d * psi_1d + gpr.kfd * psi_fd - gpr.xdpp * i_dg) / gpr.xqpp, 0, 0,
                   gpr.kfd * npr.wn * w / gpr.xqpp, gpr.k1d * npr.wn * w / gpr.xqpp, npr.wn * (
                           -gpr.k1q * gpr.r1q * gpr.x2q_loop / gpr.xdet_q - gpr.k2q * gpr.r2q * (
                           -gpr.xaq - gpr.xrlq) / gpr.xdet_q) / gpr.xqpp, npr.wn * (-gpr.k1q * gpr.r1q * (
                          -gpr.xaq - gpr.xrlq) / gpr.xdet_q - gpr.k2q * gpr.r2q * gpr.x1q_loop / gpr.xdet_q) / gpr.xqpp,
                   -gpr.xdpp * npr.wn * w / gpr.xqpp,
                   npr.wn * (-gpr.k1q ** 2 * gpr.r1q - gpr.k2q ** 2 * gpr.r2q - gpr.ra) / gpr.xqpp, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   npr.wn * np.sin(delta) / gpr.xqpp, -npr.wn * np.cos(delta) / gpr.xqpp, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1 / gpr.Te, -gpr.Kc / gpr.Te, gpr.Kc / (gpr.Tc * gpr.Te), 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1 / gpr.Tm, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, vscx / (gpr.Tm * np.sqrt(vscx ** 2 + vscy ** 2)),
                   vscy / (gpr.Tm * np.sqrt(vscx ** 2 + vscy ** 2)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, npr.wn * (-wf.Kpc - wf.Rt) / wf.Lt, 0,
                   npr.wn * wf.Kic / wf.Lt, 0, npr.wn / wf.Lt, 0, npr.wn * (
                           vwfx * np.sin(thetaw) - vwfy * np.cos(thetaw) - wf.Kp_pll * wf.Kpc * wf.Kpf * wf.Kpp * (
                           -vwfx * np.cos(thetaw) - vwfy * np.sin(thetaw))) / wf.Lt,
                   -npr.wn * wf.Ki_pll * wf.Kpc * wf.Kpf * wf.Kpp / wf.Lt, npr.wn * wf.Kif * wf.Kpc * wf.Kpp / wf.Lt,
                   npr.wn * wf.Kip * wf.Kpc / wf.Lt, 0, -npr.wn * wf.Kpc * wf.Kpp / wf.Lt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   npr.wn * (wf.Kp_pll * wf.Kpc * wf.Kpf * wf.Kpp * np.sin(thetaw) - np.cos(thetaw)) / wf.Lt,
                   npr.wn * (-wf.Kp_pll * wf.Kpc * wf.Kpf * wf.Kpp * np.cos(thetaw) - np.sin(thetaw)) / wf.Lt, 0, 0, 0,
                   0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, npr.wn * (-wf.Kpc - wf.Rt) / wf.Lt, 0,
                   npr.wn * wf.Kic / wf.Lt, 0, npr.wn / wf.Lt,
                   npr.wn * (vwfx * np.cos(thetaw) + vwfy * np.sin(thetaw)) / wf.Lt, 0, 0, 0,
                   npr.wn * wf.Kiq * wf.Kpc / wf.Lt, 0, npr.wn * wf.Kpc * wf.Kpq * wf.Kq / wf.Lt,
                   npr.wn * wf.Kpc * wf.Kpq * wf.Kv / wf.Lt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   npr.wn * np.sin(thetaw) / wf.Lt, -npr.wn * np.cos(thetaw) / wf.Lt, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
                   -wf.Kp_pll * wf.Kpf * wf.Kpp * (-vwfx * np.cos(thetaw) - vwfy * np.sin(thetaw)),
                   -wf.Ki_pll * wf.Kpf * wf.Kpp, wf.Kif * wf.Kpp, wf.Kip, 0, -wf.Kpp, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, wf.Kp_pll * wf.Kpf * wf.Kpp * np.sin(thetaw),
                   -wf.Kp_pll * wf.Kpf * wf.Kpp * np.cos(thetaw), 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, wf.Kiq, 0, wf.Kpq * wf.Kq,
                   wf.Kpq * wf.Kv, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1 / wf.Tad, 0,
                   (-vwfx * np.sin(thetaw) + vwfy * np.cos(thetaw)) / wf.Tad, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.cos(thetaw) / wf.Tad, np.sin(thetaw) / wf.Tad, 0, 0, 0, 0,
                   0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1 / wf.Tad,
                   (-vwfx * np.cos(thetaw) - vwfy * np.sin(thetaw)) / wf.Tad, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -np.sin(thetaw) / wf.Tad, np.cos(thetaw) / wf.Tad, 0, 0, 0, 0,
                   0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   npr.wn * wf.Kp_pll * (-vwfx * np.cos(thetaw) - vwfy * np.sin(thetaw)), npr.wn * wf.Ki_pll, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   -npr.wn * wf.Kp_pll * np.sin(thetaw), npr.wn * wf.Kp_pll * np.cos(thetaw), 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   -vwfx * np.cos(thetaw) - vwfy * np.sin(thetaw), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, -np.sin(thetaw), np.cos(thetaw), 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   -wf.Kp_pll * (-vwfx * np.cos(thetaw) - vwfy * np.sin(thetaw)), -wf.Ki_pll, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, wf.Kp_pll * np.sin(thetaw),
                   -wf.Kp_pll * np.cos(thetaw), 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   -wf.Kp_pll * wf.Kpf * (-vwfx * np.cos(thetaw) - vwfy * np.sin(thetaw)), -wf.Ki_pll * wf.Kpf, wf.Kif,
                   0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   wf.Kp_pll * wf.Kpf * np.sin(thetaw), -wf.Kp_pll * wf.Kpf * np.cos(thetaw), 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, wf.Kq, wf.Kv, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (vwfx * np.cos(thetaw) + vwfy * np.sin(thetaw)) / wf.Tpm,
                   (-vwfx * np.sin(thetaw) + vwfy * np.cos(thetaw)) / wf.Tpm, 0, 0, 0, 0, (
                           Idw * (-vwfx * np.sin(thetaw) + vwfy * np.cos(thetaw)) + Iqw * (
                           -vwfx * np.cos(thetaw) - vwfy * np.sin(thetaw))) / wf.Tpm, 0, 0, 0, 0, -1 / wf.Tpm,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   (Idw * np.cos(thetaw) - Iqw * np.sin(thetaw)) / wf.Tpm,
                   (Idw * np.sin(thetaw) + Iqw * np.cos(thetaw)) / wf.Tpm, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-vwfx * np.sin(thetaw) + vwfy * np.cos(thetaw)) / wf.Tpm,
                   (-vwfx * np.cos(thetaw) - vwfy * np.sin(thetaw)) / wf.Tpm, 0, 0, 0, 0, (
                           Idw * (-vwfx * np.cos(thetaw) - vwfy * np.sin(thetaw)) - Iqw * (
                           -vwfx * np.sin(thetaw) + vwfy * np.cos(thetaw))) / wf.Tpm, 0, 0, 0, 0, 0,
                   -1 / wf.Tpm, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   (-Idw * np.sin(thetaw) - Iqw * np.cos(thetaw)) / wf.Tpm,
                   (Idw * np.cos(thetaw) - Iqw * np.sin(thetaw)) / wf.Tpm, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (
                          (-2 * vwfx * np.sin(thetaw) + 2 * vwfy * np.cos(thetaw)) * (
                          vwfx * np.cos(thetaw) + vwfy * np.sin(thetaw)) / 2 + (
                                  -vwfx * np.sin(thetaw) + vwfy * np.cos(thetaw)) * (
                                  -2 * vwfx * np.cos(thetaw) - 2 * vwfy * np.sin(thetaw)) / 2) / (
                           wf.Tvm * np.sqrt((-vwfx * np.sin(thetaw) + vwfy * np.cos(thetaw)) ** 2 + (
                           vwfx * np.cos(thetaw) + vwfy * np.sin(thetaw)) ** 2)), 0, 0, 0, 0, 0, 0, -1 / wf.Tvm,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (
                           -(-vwfx * np.sin(thetaw) + vwfy * np.cos(thetaw)) * np.sin(thetaw) + (
                           vwfx * np.cos(thetaw) + vwfy * np.sin(thetaw)) * np.cos(thetaw)) / (wf.Tvm * np.sqrt(
                      (-vwfx * np.sin(thetaw) + vwfy * np.cos(thetaw)) ** 2 + (
                              vwfx * np.cos(thetaw) + vwfy * np.sin(thetaw)) ** 2)), (
                           (-vwfx * np.sin(thetaw) + vwfy * np.cos(thetaw)) * np.cos(thetaw) + (
                           vwfx * np.cos(thetaw) + vwfy * np.sin(thetaw)) * np.sin(thetaw)) / (wf.Tvm * np.sqrt(
                      (-vwfx * np.sin(thetaw) + vwfy * np.cos(thetaw)) ** 2 + (
                              vwfx * np.cos(thetaw) + vwfy * np.sin(thetaw)) ** 2)), 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   npr.wn * (-off.Kpc - off.Rt) / off.Lt, 0, npr.wn * off.Kic / off.Lt, 0, npr.wn / off.Lt, 0,
                   npr.wn * (-off.Kp_pll * off.Kpc * off.Kpf * off.Kpp * (
                           -voffx * np.cos(thetao) - voffy * np.sin(thetao)) + voffx * np.sin(
                       thetao) - voffy * np.cos(thetao)) / off.Lt,
                   -npr.wn * off.Ki_pll * off.Kpc * off.Kpf * off.Kpp / off.Lt,
                   npr.wn * off.Kif * off.Kpc * off.Kpp / off.Lt, npr.wn * off.Kip * off.Kpc / off.Lt, 0,
                   -npr.wn * off.Kpc * off.Kpp / off.Lt, 0, 0,
                   npr.wn * (off.Kp_pll * off.Kpc * off.Kpf * off.Kpp * np.sin(thetao) - np.cos(thetao)) / off.Lt,
                   npr.wn * (-off.Kp_pll * off.Kpc * off.Kpf * off.Kpp * np.cos(thetao) - np.sin(thetao)) / off.Lt, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   npr.wn * (-off.Kpc - off.Rt) / off.Lt, 0, npr.wn * off.Kic / off.Lt, 0, npr.wn / off.Lt,
                   npr.wn * (voffx * np.cos(thetao) + voffy * np.sin(thetao)) / off.Lt, 0, 0, 0,
                   npr.wn * off.Kiq * off.Kpc / off.Lt, 0, npr.wn * off.Kpc * off.Kpq * off.Kq / off.Lt,
                   npr.wn * off.Kpc * off.Kpq * off.Kv / off.Lt, npr.wn * np.sin(thetao) / off.Lt,
                   -npr.wn * np.cos(thetao) / off.Lt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
                   -off.Kp_pll * off.Kpf * off.Kpp * (-voffx * np.cos(thetao) - voffy * np.sin(thetao)),
                   -off.Ki_pll * off.Kpf * off.Kpp, off.Kif * off.Kpp, off.Kip, 0, -off.Kpp, 0, 0,
                   off.Kp_pll * off.Kpf * off.Kpp * np.sin(thetao), -off.Kp_pll * off.Kpf * off.Kpp * np.cos(thetao), 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0,
                   0, 0, 0, 0, off.Kiq, 0, off.Kpq * off.Kq, off.Kpq * off.Kv, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   -1 / off.Tad, 0, (-voffx * np.sin(thetao) + voffy * np.cos(thetao)) / off.Tad, 0, 0, 0, 0, 0, 0, 0,
                   np.cos(thetao) / off.Tad, np.sin(thetao) / off.Tad, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   -1 / off.Tad, (-voffx * np.cos(thetao) - voffy * np.sin(thetao)) / off.Tad, 0, 0, 0, 0, 0, 0, 0,
                   -np.sin(thetao) / off.Tad, np.cos(thetao) / off.Tad, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   npr.wn * off.Kp_pll * (-voffx * np.cos(thetao) - voffy * np.sin(thetao)), npr.wn * off.Ki_pll, 0, 0,
                   0, 0, 0, 0, -npr.wn * off.Kp_pll * np.sin(thetao), npr.wn * off.Kp_pll * np.cos(thetao), 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   -voffx * np.cos(thetao) - voffy * np.sin(thetao), 0, 0, 0, 0, 0, 0, 0, -np.sin(thetao),
                   np.cos(thetao), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   -off.Kp_pll * (-voffx * np.cos(thetao) - voffy * np.sin(thetao)), -off.Ki_pll, 0, 0, 0, 0, 0, 0,
                   off.Kp_pll * np.sin(thetao), -off.Kp_pll * np.cos(thetao), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   -off.Kp_pll * off.Kpf * (-voffx * np.cos(thetao) - voffy * np.sin(thetao)), -off.Ki_pll * off.Kpf,
                   off.Kif, 0, 0, -1, 0, 0, off.Kp_pll * off.Kpf * np.sin(thetao),
                   -off.Kp_pll * off.Kpf * np.cos(thetao), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, off.Kq, off.Kv, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   (voffx * np.cos(thetao) + voffy * np.sin(thetao)) / off.Tpm,
                   (-voffx * np.sin(thetao) + voffy * np.cos(thetao)) / off.Tpm, 0, 0, 0, 0, (
                           Ido * (-voffx * np.sin(thetao) + voffy * np.cos(thetao)) + Iqo * (
                           -voffx * np.cos(thetao) - voffy * np.sin(thetao))) / off.Tpm, 0, 0, 0, 0,
                   -1 / off.Tpm, 0, 0, (Ido * np.cos(thetao) - Iqo * np.sin(thetao)) / off.Tpm,
                   (Ido * np.sin(thetao) + Iqo * np.cos(thetao)) / off.Tpm, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   (-voffx * np.sin(thetao) + voffy * np.cos(thetao)) / off.Tpm,
                   (-voffx * np.cos(thetao) - voffy * np.sin(thetao)) / off.Tpm, 0, 0, 0, 0, (
                           Ido * (-voffx * np.cos(thetao) - voffy * np.sin(thetao)) - Iqo * (
                           -voffx * np.sin(thetao) + voffy * np.cos(thetao))) / off.Tpm, 0, 0, 0, 0, 0,
                   -1 / off.Tpm, 0, (-Ido * np.sin(thetao) - Iqo * np.cos(thetao)) / off.Tpm,
                   (Ido * np.cos(thetao) - Iqo * np.sin(thetao)) / off.Tpm, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (
                          (-2 * voffx * np.sin(thetao) + 2 * voffy * np.cos(thetao)) * (
                          voffx * np.cos(thetao) + voffy * np.sin(thetao)) / 2 + (
                                  -voffx * np.sin(thetao) + voffy * np.cos(thetao)) * (
                                  -2 * voffx * np.cos(thetao) - 2 * voffy * np.sin(thetao)) / 2) / (
                           off.Tvm * np.sqrt((-voffx * np.sin(thetao) + voffy * np.cos(thetao)) ** 2 + (
                           voffx * np.cos(thetao) + voffy * np.sin(thetao)) ** 2)), 0, 0, 0, 0, 0, 0,
                   -1 / off.Tvm, (-(-voffx * np.sin(thetao) + voffy * np.cos(thetao)) * np.sin(thetao) + (
                          voffx * np.cos(thetao) + voffy * np.sin(thetao)) * np.cos(thetao)) / (off.Tvm * np.sqrt(
                      (-voffx * np.sin(thetao) + voffy * np.cos(thetao)) ** 2 + (
                              voffx * np.cos(thetao) + voffy * np.sin(thetao)) ** 2)), (
                           (-voffx * np.sin(thetao) + voffy * np.cos(thetao)) * np.cos(thetao) + (
                           voffx * np.cos(thetao) + voffy * np.sin(thetao)) * np.sin(thetao)) / (
                           off.Tvm * np.sqrt((-voffx * np.sin(thetao) + voffy * np.cos(thetao)) ** 2 + (
                           voffx * np.cos(thetao) + voffy * np.sin(thetao)) ** 2)), 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   npr.wn * off.Sn * np.cos(thetao) / (Chub * npr.Sb),
                   -npr.wn * off.Sn * np.sin(thetao) / (Chub * npr.Sb), 0, 0, 0, 0,
                   npr.wn * off.Sn * (-Ido * np.sin(thetao) - Iqo * np.cos(thetao)) / (Chub * npr.Sb), 0, 0, 0, 0, 0, 0,
                   0, 0, npr.wn, 0, 0, 0, 0, 0, 0, -npr.wn / Chub, 0, -npr.wn / Chub, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   npr.wn * off.Sn * np.sin(thetao) / (Chub * npr.Sb),
                   npr.wn * off.Sn * np.cos(thetao) / (Chub * npr.Sb), 0, 0, 0, 0,
                   npr.wn * off.Sn * (Ido * np.cos(thetao) - Iqo * np.sin(thetao)) / (Chub * npr.Sb), 0, 0, 0, 0, 0, 0,
                   0, -npr.wn, 0, 0, 0, 0, 0, 0, 0, 0, -npr.wn / Chub, 0, -npr.wn / Chub, 0, 0],
                  [gpr.Sn * npr.wn * (-i_dg * np.sin(delta) - i_qg * np.cos(delta)) / (Csc * npr.Sb), 0, 0, 0, 0, 0, 0,
                   0, gpr.Sn * npr.wn * np.cos(delta) / (Csc * npr.Sb),
                   -gpr.Sn * npr.wn * np.sin(delta) / (Csc * npr.Sb), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, npr.wn, 0, 0, 0, 0, npr.wn / Csc, 0, 0, 0, 0, 0],
                  [gpr.Sn * npr.wn * (i_dg * np.cos(delta) - i_qg * np.sin(delta)) / (Csc * npr.Sb), 0, 0, 0, 0, 0, 0,
                   0, gpr.Sn * npr.wn * np.sin(delta) / (Csc * npr.Sb),
                   gpr.Sn * npr.wn * np.cos(delta) / (Csc * npr.Sb), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -npr.wn, 0, 0, 0, 0, 0, 0, npr.wn / Csc, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, npr.wn, 0, 0, 0, 0, npr.wn / Ccb, 0, -npr.wn / Ccb, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -npr.wn, 0, 0, 0, 0, 0, 0, npr.wn / Ccb, 0, -npr.wn / Ccb],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, npr.wn * wf.Sn * np.cos(thetaw) / (Ccb * npr.Sb),
                   -npr.wn * wf.Sn * np.sin(thetaw) / (Ccb * npr.Sb), 0, 0, 0, 0,
                   npr.wn * wf.Sn * (-Idw * np.sin(thetaw) - Iqw * np.cos(thetaw)) / (Ccb * npr.Sb), 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, npr.wn, 0, 0, 0, 0, npr.wn / Ccb,
                   0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, npr.wn * wf.Sn * np.sin(thetaw) / (Ccb * npr.Sb),
                   npr.wn * wf.Sn * np.cos(thetaw) / (Ccb * npr.Sb), 0, 0, 0, 0,
                   npr.wn * wf.Sn * (Idw * np.cos(thetaw) - Iqw * np.sin(thetaw)) / (Ccb * npr.Sb), 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -npr.wn, 0, 0, 0, 0, 0, 0, npr.wn / Ccb],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, npr.wn / Ltsc, 0, -npr.wn / Ltsc, 0, 0, 0, 0, 0, -Rtsc * npr.wn / Ltsc, npr.wn,
                   0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, npr.wn / Ltsc, 0, -npr.wn / Ltsc, 0, 0, 0, 0, -npr.wn, -Rtsc * npr.wn / Ltsc,
                   0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, npr.wn / Ltwf, 0, 0, 0, -npr.wn / Ltwf, 0, 0, 0, 0, 0, -Rtwf * npr.wn / Ltwf,
                   npr.wn, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, npr.wn / Ltwf, 0, 0, 0, -npr.wn / Ltwf, 0, 0, 0, 0, -npr.wn,
                   -Rtwf * npr.wn / Ltwf, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, npr.wn / Lcb, 0, -npr.wn / Lcb, 0, 0, 0, 0, 0, -Rcb * npr.wn / Lcb,
                   npr.wn],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, npr.wn / Lcb, 0, -npr.wn / Lcb, 0, 0, 0, 0, -npr.wn,
                   -Rcb * npr.wn / Lcb],
                  ])

    B = np.array([[0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [npr.wn * wf.Kpc * wf.Kpp / wf.Lt, 0],
                  [0, -npr.wn * wf.Kpc * wf.Kpq * wf.Kq / wf.Lt],
                  [wf.Kpp, 0],
                  [0, -wf.Kpq * wf.Kq],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [1, 0],
                  [0, -wf.Kq],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  [0, 0],
                  ])

    C = np.zeros((2, 51))

    D = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])

    Amat = A  # -B@np.linalg.inv(D)@C

    lambda_1, Phi_1 = np.linalg.eig(Amat)

    lambda_1[np.where(abs(lambda_1) < 1e-7)] = 0

    return lambda_1, Amat, A, B, C, D
