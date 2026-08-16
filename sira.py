"""
sira.py -- SIRA model with awareness-driven accessibility.

This module is the plain, importable, plotting-free core of the code
accompanying:

    A. Alsammani and M. Mohammed, "Long-term Coexistence of Epidemics and
    Risk Awareness: Impacts of Adaptive Human Response and Fatigue,"
    submitted to BIOMATH.

It is extracted verbatim from Sections 2-3 of SIRA_reproducibility.ipynb,
so importing this module and running the notebook use exactly the same
code. See the notebook for full derivations and commentary.

Public API
----------
    DEFAULT     : baseline parameter dictionary (Table 1 of the manuscript)
    simulate(p, T, n, y0) : integrate a trajectory (LSODA, rtol=1e-10, atol=1e-12)
    R0(p)       : basic reproduction number beta / (gamma + mu)
    Phi(A, p)   : sequestration factor, Eq. (7)
    endemic(p)  : unique endemic equilibrium via bisection on the strictly
                  monotone scalar function G of Eqs. (18)-(19); None if
                  R0 <= 1 (Theorem 3.6)
    A_infty(p)  : low-fatigue awareness plateau, root of Eq. (26)
    spectrum(p) : eigenvalues of the Jacobian of the full six-dimensional
                  system, evaluated at the endemic equilibrium
    eqcurve(key, xs) : endemic (I*, A*) along a one-parameter sweep

State ordering throughout: y = (Sa, Su, Ia, Iu, R, A).
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

DEFAULT = dict(beta=0.6, gamma=0.2, mu=0.02,
               k1=0.4, k2=0.4, d1=0.2, d2=0.2,
               rho=0.5, omega=0.01)

Y0 = np.array([0.9, 0.05, 0.05, 0.0, 0.0, 0.0])   # (Sa, Su, Ia, Iu, R, A)


def rhs(t, y, p):
    '''Vector field of Eqs. (1)-(6); state order (Sa, Su, Ia, Iu, R, A).'''
    Sa, Su, Ia, Iu, R, A = y
    k1 = p['k1'] * A
    k2 = p['k2'] * (1.0 - A)
    d1 = p['d1'] * A
    d2 = p['d2'] * (1.0 - A)
    b, g, mu = p['beta'], p['gamma'], p['mu']
    dSa = (1.0 - A) * mu - b * Sa * Ia + k2 * Su - k1 * Sa - mu * Sa
    dSu = A * mu + k1 * Sa - k2 * Su - mu * Su
    dIa = b * Sa * Ia - d1 * Ia + d2 * Iu - (g + mu) * Ia
    dIu = d1 * Ia - d2 * Iu - (g + mu) * Iu
    dR = g * (Ia + Iu) - mu * R
    dA = p['rho'] * (Ia + Iu) * (1.0 - A) - p['omega'] * A
    return (dSa, dSu, dIa, dIu, dR, dA)


def simulate(p, T=800.0, n=4000, y0=None):
    '''Integrate the model; returns (t, Y) with Y of shape (6, n).'''
    if y0 is None:
        y0 = Y0
    t = np.linspace(0.0, T, n)
    sol = solve_ivp(rhs, (0.0, T), np.asarray(y0, dtype=float),
                    t_eval=t, args=(p,), method='LSODA',
                    rtol=1e-10, atol=1e-12)
    return sol.t, sol.y


def R0(p):
    return p['beta'] / (p['gamma'] + p['mu'])


def Phi(A, p):
    '''Sequestration factor of Eq. (7).'''
    return 1.0 + p['d1'] * A / (p['d2'] * (1.0 - A) + p['gamma'] + p['mu'])


def _A_of_I(I, p):
    if p['rho'] == 0.0:
        return 0.0
    return p['rho'] * I / (p['rho'] * I + p['omega'])


def _S_of_I(I, p):
    '''Accessible susceptible fraction S_a^*(I) of Eq. (18).'''
    g, mu = p['gamma'], p['mu']
    a = _A_of_I(I, p)
    s = 1.0 - (g + mu) / mu * I
    N = (p['k2'] * (1.0 - a) + mu) * s - mu * a
    D = p['k1'] * a + p['k2'] * (1.0 - a) + mu
    return N / D


def G(I, p):
    '''Strictly decreasing scalar function of Eq. (19).'''
    g, mu = p['gamma'], p['mu']
    return p['beta'] * _S_of_I(I, p) - (g + mu) * Phi(_A_of_I(I, p), p)


def endemic(p):
    '''Unique endemic equilibrium (Theorem 3.6), or None when R0 <= 1.'''
    g, mu = p['gamma'], p['mu']
    Imax = mu / (g + mu)
    if G(1e-14, p) <= 0.0:
        return None
    I = brentq(G, 1e-14, Imax * (1.0 - 1e-12), args=(p,), xtol=1e-16)
    a = _A_of_I(I, p)
    D = p['d1'] * a + p['d2'] * (1.0 - a) + g + mu
    Ia = (p['d2'] * (1.0 - a) + g + mu) / D * I
    Iu = p['d1'] * a / D * I
    S = 1.0 - (g + mu) / mu * I
    Sa = _S_of_I(I, p)
    return dict(I=I, A=a, S=S, Sa=Sa, Su=S - Sa, Ia=Ia, Iu=Iu, R=g / mu * I)


def A_infty(p):
    '''Low-fatigue awareness plateau: the root of Eq. (26).'''
    g, mu = p['gamma'], p['mu']

    def H(A):
        N = (p['k2'] * (1.0 - A) + mu) - mu * A
        D = p['k1'] * A + p['k2'] * (1.0 - A) + mu
        return p['beta'] * N / D - (g + mu) * Phi(A, p)

    return brentq(H, 1e-12, 1.0 - 1e-9, xtol=1e-15)


def jacobian(p, y, h=1e-7):
    '''Central-difference Jacobian of the full six-dimensional vector field.'''
    y = np.asarray(y, dtype=float)
    J = np.zeros((6, 6))
    for j in range(6):
        e = np.zeros(6)
        e[j] = h
        J[:, j] = (np.asarray(rhs(0.0, y + e, p)) -
                   np.asarray(rhs(0.0, y - e, p))) / (2.0 * h)
    return J


def spectrum(p):
    '''Eigenvalues of J(E*) sorted by descending real part, with E*;
    None when no endemic equilibrium exists.'''
    e = endemic(p)
    if e is None:
        return None
    y = np.array([e['Sa'], e['Su'], e['Ia'], e['Iu'], e['R'], e['A']])
    lam = np.linalg.eigvals(jacobian(p, y))
    return lam[np.argsort(-lam.real)], e


def eqcurve(key, xs):
    '''Endemic (I*, A*) along a one-parameter sweep.'''
    I, A = [], []
    for x in xs:
        e = endemic({**DEFAULT, key: x})
        I.append(e['I']); A.append(e['A'])
    return np.array(I), np.array(A)
