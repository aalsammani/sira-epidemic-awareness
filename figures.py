"""
figures.py -- regenerate the nine manuscript figures (and the held-out
response Figure R2) as a standalone script, without Jupyter.

Extracted verbatim from SIRA_reproducibility.ipynb Sections 5-7, so this
script and the notebook produce byte-identical output. Requires sira.py
in the same directory (or on the Python path).

Usage:
    python figures.py            # all nine figures + R2
    python figures.py M2 M5      # only Manuscript Figures 2 and 5
    python figures.py R2         # only the held-out accessibility panel

File name -> figure mapping (matches the manuscript's includegraphics calls):
    figure_M2 -> figs/fig5.pdf   (Manuscript Figure 2)
    figure_M3 -> figs/fig6.pdf   (Manuscript Figure 3)
    figure_M4 -> figs/fig7.pdf   (Manuscript Figure 4)
    figure_M5 -> figs/fig8.pdf   (Manuscript Figure 5)
    figure_M6 -> figs/fig2.pdf   (Manuscript Figure 6)
    figure_M7 -> figs/fig3.pdf   (Manuscript Figure 7)
    figure_M8 -> figs/figA1.pdf  (Manuscript Figure 8)
    figure_M9 -> figs/figA2.pdf  (Manuscript Figure 9)
    figure_R2 -> figs/fig4.pdf   (Response Figure R2)
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import brentq
from sira import *

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8.5,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'lines.linewidth': 1.2,
    'grid.linewidth': 0.4,
    'grid.alpha': 0.35,
    'figure.dpi': 400,
    'savefig.dpi': 400,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})
plt.rcParams['figure.dpi'] = 115   # inline preview only; saved files use savefig.dpi = 400

CS, CI, CR, CA = '#1f6fb4', '#e2481f', '#3aa03a', '#111111'   # S, I, R, A
OUT = 'figs/'
os.makedirs(OUT, exist_ok=True)
SEED = 20260811                                               # fixes every random draw below

print(f'Python {sys.version.split()[0]} | NumPy {np.__version__} | '
      f'SciPy {scipy.__version__} | Matplotlib {matplotlib.__version__}')


def panel(ax):
    ax.grid(True, ls='-', color='0.85')
    ax.set_axisbelow(True)


def figure_M2():
    '''Manuscript Figure 2 (file figs/fig5.pdf).'''
    fig, axs = plt.subplots(2, 3, figsize=(7.0, 3.7))
    specs = [('rho', np.linspace(0.05, 1.0, 300), r'Awareness response $\rho$'),
             ('omega', np.linspace(0.001, 0.5, 400), r'Behavioral fatigue $\omega$'),
             ('beta', np.linspace(0.23, 2.0, 400), r'Transmission rate $\beta$')]
    for j, (k, xs, lab) in enumerate(specs):
        I, A = eqcurve(k, xs)
        axs[0, j].plot(xs, I, color=CI)
        axs[1, j].plot(xs, A, color=CA)
        for i in (0, 1):
            axs[i, j].set_xlim(xs[0], xs[-1]); panel(axs[i, j])
        axs[1, j].set_xlabel(lab)
    p0 = dict(DEFAULT)
    axs[0, 0].set_ylabel(r'Endemic prevalence $I^*$')
    axs[1, 0].set_ylabel(r'Endemic awareness $A^*$')
    for ax, t in zip(axs.ravel(), ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']):
        ax.set_title(t, pad=2, loc='left')
    bt = (p0['gamma'] + p0['mu'])
    for i in (0, 1):
        axs[i, 2].axvline(bt, color='0.45', ls=':', lw=0.8)
    axs[0, 2].text(bt + 0.03, 0.9 * axs[0, 2].get_ylim()[1], r'$\mathcal{R}_0=1$',
                   fontsize=6.5, color='0.35')
    fig.tight_layout(pad=0.4, w_pad=1.0, h_pad=0.7)
    fig.savefig(OUT + 'fig5.pdf')
    plt.show()

def grid_eq(kx, xs, ky, ys):
    I = np.zeros((len(ys), len(xs))); A = np.zeros_like(I)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            e = endemic({**DEFAULT, kx: x, ky: y})
            if e is None:
                I[i, j] = np.nan; A[i, j] = np.nan
            else:
                I[i, j] = e['I']; A[i, j] = e['A']
    return I, A


def figure_M3():
    '''Manuscript Figure 3 (file figs/fig6.pdf).'''
    n = 140
    w = np.linspace(0.001, 0.5, n); r = np.linspace(0.05, 1.0, n); b = np.linspace(0.23, 2.0, n)
    I1, A1 = grid_eq('omega', w, 'rho', r)
    I2, A2 = grid_eq('beta', b, 'rho', r)
    I3, A3 = grid_eq('beta', b, 'omega', w)
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 3.9))
    data = [(I1, w, r, r'Behavioral fatigue $\omega$', r'Awareness response $\rho$'),
            (I2, b, r, r'Transmission rate $\beta$', r'Awareness response $\rho$'),
            (I3, b, w, r'Transmission rate $\beta$', r'Behavioral fatigue $\omega$'),
            (A1, w, r, r'Behavioral fatigue $\omega$', r'Awareness response $\rho$'),
            (A2, b, r, r'Transmission rate $\beta$', r'Awareness response $\rho$'),
            (A3, b, w, r'Transmission rate $\beta$', r'Behavioral fatigue $\omega$')]
    vmaxI = np.nanmax([I1, I2, I3]); vmaxA = np.nanmax([A1, A2, A3])
    ims = []
    for k, (Z, xs, ys, xl, yl) in enumerate(data):
        ax = axs.ravel()[k]
        cmap = 'magma' if k < 3 else 'viridis'
        im = ax.pcolormesh(xs, ys, Z, cmap=cmap, shading='gouraud',
                           vmin=0, vmax=(vmaxI if k < 3 else vmaxA))
        ims.append(im)
        ax.set_xlabel(xl)
        if k % 3 == 0: ax.set_ylabel(yl)
        ax.set_title('(%s)' % 'abcdef'[k], pad=2, loc='left')
        if k in (0, 3):
            RATIO = np.array([[y / x for x in xs] for y in ys])
            ax.contour(xs, ys, RATIO, levels=[2, 5, 20], colors='w',
                       linewidths=0.6, linestyles='--')
    fig.tight_layout(pad=0.4, w_pad=0.9, h_pad=0.8)
    fig.subplots_adjust(right=0.88)
    cax1 = fig.add_axes([0.90, 0.56, 0.017, 0.36])
    cax2 = fig.add_axes([0.90, 0.09, 0.017, 0.36])
    fig.colorbar(ims[0], cax=cax1).set_label(r'$I^*$', fontsize=8)
    fig.colorbar(ims[3], cax=cax2).set_label(r'$A^*$', fontsize=8)
    fig.savefig(OUT + 'fig6.pdf')
    plt.show()

def figure_M4():
    '''Manuscript Figure 4 (file figs/fig7.pdf).'''
    fig, axs = plt.subplots(2, 2, figsize=(5.8, 4.2))
    # (a) collapse onto omega/rho
    ax = axs[0, 0]
    thetas = np.logspace(-4, 0, 300)
    for rho, mk in zip([0.1, 0.5, 2.0], ['o', 's', '^']):
        Is = []
        for th in thetas[::12]:
            Is.append(endemic({**DEFAULT, 'rho': rho, 'omega': th * rho})['I'])
        ax.plot(thetas[::12], Is, mk, ms=2.6, mfc='none', mew=0.7,
                label=r'$\rho=%.1f$' % rho)
    Iref = [endemic({**DEFAULT, 'rho': 1.0, 'omega': th})['I'] for th in thetas]
    ax.plot(thetas, Iref, '-', color='0.3', lw=1.0, zorder=0, label='common curve')
    ax.set_xscale('log'); ax.set_xlabel(r'$\theta=\omega/\rho$')
    ax.set_ylabel(r'Endemic prevalence $I^*$')
    ax.legend(frameon=False, loc='upper left', handlelength=1.2)
    panel(ax)
    # (b) small-theta asymptotics
    ax = axs[0, 1]
    Ainf = A_infty(dict(DEFAULT))
    th = np.logspace(-5, -0.5, 200)
    Ie = [endemic({**DEFAULT, 'rho': 1.0, 'omega': t})['I'] for t in th]
    ax.loglog(th, Ie, color=CI, label=r'exact $I^*(\theta)$')
    ax.loglog(th, th * Ainf / (1 - Ainf), '--', color='0.3', lw=0.9,
              label=r'$\theta\,A_\infty/(1-A_\infty)$')
    ax.set_xlabel(r'$\theta=\omega/\rho$'); ax.set_ylabel(r'$I^*$')
    ax.legend(frameon=False, loc='upper left', handlelength=1.6)
    panel(ax)
    # (c) exact decomposition of burden reduction
    ax = axs[1, 0]
    th = np.linspace(1e-4, 0.3, 300)
    p0 = dict(DEFAULT); r0 = R0(p0); c = p0['mu'] / (p0['gamma'] + p0['mu'])
    ISIR = c * (1 - 1 / r0)
    ch1, ch2, Iv = [], [], []
    for t in th:
        e = endemic({**DEFAULT, 'rho': 1.0, 'omega': t})
        ch1.append(c * e['Su'])
        ch2.append(c * (Phi(e['A'], {**DEFAULT, 'rho': 1.0, 'omega': t}) - 1) / r0)
        Iv.append(e['I'])
    ch1, ch2, Iv = map(np.array, (ch1, ch2, Iv))
    ax.fill_between(th, 0, Iv, color=CI, alpha=.55, lw=0, label=r'residual $I^*$')
    ax.fill_between(th, Iv, Iv + ch1, color='#4c78a8', alpha=.75, lw=0,
                    label='susceptible withdrawal')
    ax.fill_between(th, Iv + ch1, Iv + ch1 + ch2, color='#9ecae1', alpha=.9, lw=0,
                    label='infectious sequestration')
    ax.axhline(ISIR, color='k', lw=0.8)
    ax.set_xlim(0, 0.3); ax.set_ylim(0, ISIR * 1.08)
    ax.set_xlabel(r'$\theta=\omega/\rho$'); ax.set_ylabel(r'Prevalence budget')
    ax.text(0.005, ISIR * 1.01, r'$I^*_{\rm SIR}$ (no awareness)', fontsize=6.5)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], loc='center right', frameon=True, framealpha=0.9,
              edgecolor='none', handlelength=1.0, labelspacing=0.25, fontsize=6.5)
    panel(ax)
    # (d) relative suppression vs beta
    ax = axs[1, 1]
    bs = np.linspace(0.25, 2.0, 300)
    for t, ls in zip([0.005, 0.02, 0.1], ['-', '--', ':']):
        rel = []
        for b in bs:
            q = {**DEFAULT, 'beta': b, 'rho': 1.0, 'omega': t}
            IS = c * (1 - 1 / R0(q))
            rel.append(endemic(q)['I'] / IS)
        ax.plot(bs, rel, ls, color=CA, lw=1.0, label=r'$\theta=%.3f$' % t)
    ax.set_xlabel(r'Transmission rate $\beta$')
    ax.set_ylabel(r'$I^*/I^*_{\rm SIR}$')
    ax.set_xlim(0.25, 2.0); ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc='lower right', handlelength=1.6)
    panel(ax)
    for ax, t in zip(axs.ravel(), ['(a)', '(b)', '(c)', '(d)']):
        ax.set_title(t, pad=2, loc='left')
    fig.tight_layout(pad=0.4, w_pad=1.0, h_pad=0.9)
    fig.savefig(OUT + 'fig7.pdf')
    plt.show()

HOPF = dict(DEFAULT); HOPF.update(beta=1.8, d2=0.01, rho=1.0, omega=0.01)


def maxre(p):
    r = spectrum(p)
    return np.nan if r is None else r[0].real.max()


def red_maxre(p):
    '''Fast-relaxation limit of Proposition 3.13: 3-dimensional reduced system.'''
    g, mu = p['gamma'], p['mu']

    def gS(A): return p['k2'] * (1 - A) / (p['k1'] * A + p['k2'] * (1 - A))

    def gI(A):
        den = p['d1'] * A + p['d2'] * (1 - A)
        return p['d2'] * (1 - A) / den if den > 0 else 1.0

    def b(A): return p['beta'] * gS(A) * gI(A)
    Imax = mu / (g + mu)

    def H(I):
        A = p['rho'] * I / (p['rho'] * I + p['omega'])
        return b(A) * (1 - (g + mu) / mu * I) - (g + mu)
    if H(1e-14) <= 0: return np.nan
    I = brentq(H, 1e-14, Imax * (1 - 1e-12), xtol=1e-16)
    A = p['rho'] * I / (p['rho'] * I + p['omega']); S = 1 - (g + mu) / mu * I

    def f(y):
        S, I, A = y
        return np.array([mu - b(A) * S * I - mu * S, b(A) * S * I - (g + mu) * I,
                         p['rho'] * I * (1 - A) - p['omega'] * A])
    y = np.array([S, I, A]); J = np.zeros((3, 3)); h = 1e-7
    for j in range(3):
        e = np.zeros(3); e[j] = h
        J[:, j] = (f(y + e) - f(y - e)) / (2 * h)
    return np.linalg.eigvals(J).real.max()


def figure_M5():
    '''Manuscript Figure 5 (file figs/fig8.pdf).'''
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.2))
    n = 90
    d2s = np.linspace(0.005, 0.35, n); d1s = np.linspace(0.02, 1.0, n)
    Zf = np.zeros((n, n)); Zr = np.zeros((n, n))
    for i, d1 in enumerate(d1s):
        for j, d2 in enumerate(d2s):
            q = dict(HOPF); q['d1'] = d1; q['d2'] = d2
            Zf[i, j] = maxre(q); Zr[i, j] = red_maxre(q)
    v = np.nanmax(np.abs(Zf))
    for ax, Z, ttl in [(axs[0, 0], Zf, 'SIRA model'),
                       (axs[0, 1], Zr, 'Fast-relaxation limit')]:
        im = ax.pcolormesh(d2s, d1s, Z, cmap='RdBu_r', vmin=-v, vmax=v, shading='gouraud')
        if np.nanmax(Z) > 0:
            ax.contour(d2s, d1s, Z, levels=[0], colors='k', linewidths=0.9)
        ax.set_xlabel(r'$\delta_2^0$'); ax.set_ylabel(r'$\delta_1^0$')
        ax.set_title(ttl, pad=3)
        fig.colorbar(im, ax=ax, pad=0.02).set_label(r'$\max\mathrm{Re}\,\lambda$', fontsize=7)
    axs[0, 1].text(0.5, 0.5, 'stable everywhere', transform=axs[0, 1].transAxes,
                   ha='center', fontsize=7.5, color='0.15')
    # (c) (omega,rho) plane
    ax = axs[0, 2]
    ws = np.linspace(0.002, 0.12, n); rs = np.linspace(0.05, 3.0, n)
    Z = np.zeros((n, n))
    for i, r in enumerate(rs):
        for j, w in enumerate(ws):
            q = dict(HOPF); q['d1'] = 0.5; q['rho'] = r; q['omega'] = w
            Z[i, j] = maxre(q)
    v2 = np.nanmax(np.abs(Z))
    im = ax.pcolormesh(ws, rs, Z, cmap='RdBu_r', vmin=-v2, vmax=v2, shading='gouraud')
    ax.contour(ws, rs, Z, levels=[0], colors='k', linewidths=0.9)
    ax.set_xlabel(r'Behavioral fatigue $\omega$'); ax.set_ylabel(r'Awareness response $\rho$')
    ax.set_title(r'SIRA model, $\delta_1^0=0.5$', pad=3)
    fig.colorbar(im, ax=ax, pad=0.02).set_label(r'$\max\mathrm{Re}\,\lambda$', fontsize=7)
    # (d) bifurcation diagram
    ax = axs[1, 0]
    d1grid = np.linspace(0.25, 0.9, 40)
    lo, hi, eq = [], [], []
    for d1 in d1grid:
        q = dict(HOPF); q['d1'] = d1
        e = endemic(q); eq.append(e['I'])
        y0 = [0, e['Su'], e['Ia'] * 1.01, e['Iu'], e['R'], e['A']]
        y0[0] = 1 - sum(y0[1:5])
        t, Y = simulate(q, y0=y0, T=30000, n=90000)
        I = (Y[2] + Y[3])[t > 25000]
        lo.append(I.min()); hi.append(I.max())
    lo, hi, eq = map(np.array, (lo, hi, eq))
    stable = (hi - lo) < 1e-6
    ax.plot(d1grid[stable], eq[stable], color=CA, lw=1.1)
    ax.plot(d1grid[~stable], eq[~stable], color=CA, lw=1.1, ls='--')
    ax.plot(d1grid[~stable], lo[~stable], color=CI, lw=1.1)
    ax.plot(d1grid[~stable], hi[~stable], color=CI, lw=1.1)
    ax.fill_between(d1grid[~stable], lo[~stable], hi[~stable], color=CI, alpha=0.15, lw=0)
    ax.set_xlabel(r'$\delta_1^0$'); ax.set_ylabel(r'Attractor range of $I$')
    ax.set_xlim(d1grid[0], d1grid[-1])
    panel(ax)
    # (e,f) time series
    for ax, d1, lab in [(axs[1, 1], 0.30, 'damped'), (axs[1, 2], 0.60, 'sustained')]:
        q = dict(HOPF); q['d1'] = d1
        t, Y = simulate(q, T=2500, n=12000)
        ax.plot(t, Y[0] + Y[1], color=CS, label='$S$')
        ax.plot(t, Y[2] + Y[3], color=CI, label='$I$')
        ax.plot(t, Y[5], color=CA, label='$A$')
        ax.set_xlim(0, 2500); ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel('Time (days)')
        ax.set_title(r'$\delta_1^0=%.2f$ (%s)' % (d1, lab), pad=3)
        panel(ax)
        if lab == 'damped':
            ax.set_ylabel('Fraction of population')
            ax.legend(frameon=False, loc='center right', handlelength=1.1)
    for ax, t in zip(axs.ravel(), ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']):
        ax.set_title(ax.get_title(), pad=3)
        ax.text(-0.02, 1.14, t, transform=ax.transAxes, fontsize=8.5, va='top')
    fig.tight_layout(pad=0.4, w_pad=1.1, h_pad=1.1)
    fig.savefig(OUT + 'fig8.pdf')
    plt.show()

def traj_panel(ax, p, T=800, legend=False, title=None):
    t, Y = simulate(p, T=T, n=4000)
    S, I, R, A = Y[0] + Y[1], Y[2] + Y[3], Y[4], Y[5]
    ax.plot(t, S, color=CS, label='$S$')
    ax.plot(t, I, color=CI, label='$I$')
    ax.plot(t, R, color=CR, label='$R$')
    ax.plot(t, A, color=CA, label='$A$')
    ax.set_xlim(0, T); ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_xticks([0, 200, 400, 600, 800])
    panel(ax)
    if title: ax.set_title(title, pad=3)
    if legend: ax.legend(loc='center right', frameon=False, handlelength=1.1,
                         labelspacing=0.25, borderpad=0.2)


def figure_M6():
    '''Manuscript Figure 6 (file figs/fig2.pdf).'''
    fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.4), sharex=True, sharey=True)
    for j, r in enumerate([0.2, 0.4, 0.6, 0.8]):
        traj_panel(axs[0, j], {**DEFAULT, 'rho': r}, legend=(j == 0),
                   title=r'(%s) $\rho=%.1f$' % ('abcd'[j], r))
    for j, w in enumerate([0.005, 0.01, 0.03, 0.06]):
        traj_panel(axs[1, j], {**DEFAULT, 'omega': w},
                   title=r'(%s) $\omega=%.3f$' % ('efgh'[j], w))
    for ax in axs[1]: ax.set_xlabel('Time (days)')
    for ax in axs[:, 0]: ax.set_ylabel('Fraction of population')
    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.6)
    fig.savefig(OUT + 'fig2.pdf')
    plt.show()

def figure_M7():
    '''Manuscript Figure 7 (file figs/fig3.pdf).'''
    fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.4), sharex=True, sharey=True)
    for j, k in enumerate([0.01, 0.2, 0.4, 0.6]):
        traj_panel(axs[0, j], {**DEFAULT, 'k1': k}, legend=(j == 0),
                   title=r'(%s) $\kappa_1^0=%.2f$' % ('abcd'[j], k))
    for j, k in enumerate([0.01, 0.2, 0.4, 0.6]):
        traj_panel(axs[1, j], {**DEFAULT, 'k2': k},
                   title=r'(%s) $\kappa_2^0=%.2f$' % ('efgh'[j], k))
    for ax in axs[1]: ax.set_xlabel('Time (days)')
    for ax in axs[:, 0]: ax.set_ylabel('Fraction of population')
    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.6)
    fig.savefig(OUT + 'fig3.pdf')
    plt.show()

def figure_M8():
    '''Manuscript Figure 8 (file figs/figA1.pdf).'''
    fig, axs = plt.subplots(2, 2, figsize=(5.6, 3.6), sharex=True, sharey=True)
    for ax, b, lab in zip(axs.ravel(), [0.4, 0.55, 0.7, 0.85], 'abcd'):
        traj_panel(ax, {**DEFAULT, 'beta': b}, legend=(lab == 'a'),
                   title=r'(%s) $\beta=%.2f$' % (lab, b))
    for ax in axs[1]: ax.set_xlabel('Time (days)')
    for ax in axs[:, 0]: ax.set_ylabel('Fraction of population')
    fig.tight_layout(pad=0.4, w_pad=0.6, h_pad=0.6)
    fig.savefig(OUT + 'figA1.pdf')
    plt.show()

def figure_M9():
    '''Manuscript Figure 9 (file figs/figA2.pdf).'''
    fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.4), sharex=True, sharey=True)
    for j, d in enumerate([0.01, 0.1, 0.2, 0.3]):
        traj_panel(axs[0, j], {**DEFAULT, 'd1': d}, legend=(j == 0),
                   title=r'(%s) $\delta_1^0=%.2f$' % ('abcd'[j], d))
    for j, d in enumerate([0.01, 0.1, 0.2, 0.3]):
        traj_panel(axs[1, j], {**DEFAULT, 'd2': d},
                   title=r'(%s) $\delta_2^0=%.2f$' % ('efgh'[j], d))
    for ax in axs[1]: ax.set_xlabel('Time (days)')
    for ax in axs[:, 0]: ax.set_ylabel('Fraction of population')
    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.6)
    fig.savefig(OUT + 'figA2.pdf')
    plt.show()

def figure_R2():
    '''Response Figure R2 (file figs/fig4.pdf).'''
    fig, axs = plt.subplots(2, 2, figsize=(5.4, 3.8))
    specs = [('k1', np.linspace(0.01, 0.8, 300), r'$\kappa_1^0$  (susceptible withdrawal)'),
             ('k2', np.linspace(0.01, 0.8, 300), r'$\kappa_2^0$  (susceptible return)'),
             ('d1', np.linspace(0.01, 0.6, 300), r'$\delta_1^0$  (infectious withdrawal)'),
             ('d2', np.linspace(0.01, 0.6, 300), r'$\delta_2^0$  (infectious return)')]
    p0 = dict(DEFAULT)
    ISIR = p0['mu'] * (1 - 1 / R0(p0)) / (p0['gamma'] + p0['mu'])
    for ax, (k, xs, lab) in zip(axs.ravel(), specs):
        I, A = eqcurve(k, xs)
        ax.plot(xs, I, color=CI)
        ax.axhline(ISIR, color='0.45', ls='--', lw=0.8)
        ax.set_xlabel(lab); ax.set_xlim(xs[0], xs[-1])
        ax.set_ylim(0, 1.10 * ISIR)
        panel(ax)
    axs[0, 0].set_ylabel(r'Endemic prevalence $I^*$')
    axs[1, 0].set_ylabel(r'Endemic prevalence $I^*$')
    for ax, t in zip(axs.ravel(), ['(a)', '(b)', '(c)', '(d)']):
        ax.set_title(t, pad=2, loc='left')
    for ax in axs.ravel():
        ax.text(0.98, ISIR, r'$I^*_{\rm SIR}$', fontsize=6.5, color='0.35',
                va='bottom', ha='right', transform=ax.get_yaxis_transform())
    fig.tight_layout(pad=0.4, w_pad=1.0, h_pad=0.9)
    fig.savefig(OUT + 'fig4.pdf')
    plt.show()

if __name__ == '__main__':
    import sys
    fns = {'M2': figure_M2, 'M3': figure_M3, 'M4': figure_M4, 'M5': figure_M5,
           'M6': figure_M6, 'M7': figure_M7, 'M8': figure_M8, 'M9': figure_M9,
           'R2': figure_R2}
    which = sys.argv[1:] or list(fns)
    for w in which:
        print('figure', w, flush=True)
        fns[w]()
    print('done')
