"""
response_figures.py -- regenerate the four supplementary figures and the
two numerical scans prepared for the point-by-point response to the
BIOMATH referee report, as a standalone script, without Jupyter.

Extracted from SIRA_reproducibility.ipynb Sections 8-10, so this script
and the notebook produce byte-identical output. Requires sira.py in the
same directory (or on the Python path). The random seed matches the one
used throughout the notebook and the manuscript's response letter, so
every number printed below reproduces the letter to the digit.

  figR1  : no-awareness (rho = 0) control against the baseline SIRA
           trajectory, with periods and damping ratios annotated
  figR3  : LHS/PRCC global sensitivity of I* and A* over the Table 1
           parameter ranges (N = 1500)                        -> prcc_values.json
  figR4  : quantified transient metrics against rho and omega  -> resurgence_values.json
  scan   : maximum of max Re lambda(J(E*)) over 4e4 uniform draws from
           the Table 1 ranges and from the wider ranges         -> scan_report.json
  multi  : multistability probe from 25 initial conditions in three
           regimes                                       -> multistability_report.json

Usage:
    python response_figures.py             # everything
    python response_figures.py R1 R3       # a subset
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import qmc, rankdata
from sira import *

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'lines.linewidth': 1.2, 'grid.linewidth': 0.4, 'grid.alpha': 0.35,
    'figure.dpi': 400, 'savefig.dpi': 400,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})

CS, CI, CR, CA = '#1f6fb4', '#e2481f', '#3aa03a', '#111111'
OUT = 'figs/'
os.makedirs(OUT, exist_ok=True)
SEED = 20260811

TABLE1_RANGES = dict(beta=(0.2, 2.0), k1=(0.01, 0.8), k2=(0.01, 0.8),
                     d1=(0.01, 0.6), d2=(0.01, 0.6),
                     rho=(0.05, 1.0), omega=(0.001, 0.5))
WIDE_RANGES = dict(beta=(0.25, 3.0), gamma=(0.05, 0.5), mu=(0.002, 0.1),
                   k1=(0.0, 1.0), k2=(0.0, 1.0), d1=(0.0, 1.0), d2=(0.0, 1.0),
                   rho=(0.01, 3.0), omega=(1e-4, 1.0))
HOPF = dict(DEFAULT); HOPF.update(beta=1.8, d2=0.01, rho=1.0, omega=0.01)


def panel(ax):
    ax.grid(True, ls='-', color='0.85')
    ax.set_axisbelow(True)


# ================================================================== Fig R1
def figure_R1():
    '''Response Figure R1 (file figs/figR1.pdf): the no-awareness control.'''
    p = dict(DEFAULT)
    p0 = dict(DEFAULT); p0['rho'] = 0.0
    T = 800
    t, Y = simulate(p, T=T, n=6000)
    t0, Y0v = simulate(p0, T=T, n=6000)
    I = Y[2] + Y[3]; I0 = Y0v[2] + Y0v[3]

    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.4))
    ax = axs[0]
    ax.plot(t0, I0, color='0.45', ls='--', label=r'control ($\rho=0$)')
    ax.plot(t, I, color=CI, label='SIRA (baseline)')
    ax.set_xlabel('Time (days)'); ax.set_ylabel('Prevalence $I$')
    ax.set_xlim(0, T)
    ax.legend(frameon=False, handlelength=1.6, loc='upper right')
    panel(ax)

    ax = axs[1]
    ax.semilogy(t0, np.maximum(I0, 1e-12), color='0.45', ls='--',
                label=r'control ($\rho=0$)')
    ax.semilogy(t, np.maximum(I, 1e-12), color=CI, label='SIRA (baseline)')
    ax.set_xlabel('Time (days)'); ax.set_ylabel('Prevalence $I$ (log)')
    ax.set_xlim(0, T); ax.set_ylim(1e-6, 0.5)
    ax.text(0.03, 0.16,
            'control: period 76 d, damping 0.329\nSIRA: period 105 d, damping 0.143',
            transform=ax.transAxes, fontsize=6.5,
            bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.5))
    panel(ax)
    for ax, lab in zip(axs, ['(a)', '(b)']):
        ax.set_title(lab, pad=2, loc='left')
    fig.tight_layout(pad=0.4, w_pad=1.0)
    fig.savefig(OUT + 'figR1.pdf')
    plt.close(fig)


# ================================================================== Fig R3
def prcc(X, y):
    '''Partial rank correlation coefficient of each column of X with y.'''
    n, d = X.shape
    Xr = np.column_stack([rankdata(X[:, j]) for j in range(d)])
    yr = rankdata(y)
    out = np.zeros(d)
    for j in range(d):
        others = np.delete(np.arange(d), j)
        Zo = np.column_stack([np.ones(n), Xr[:, others]])
        rx = Xr[:, j] - Zo @ np.linalg.lstsq(Zo, Xr[:, j], rcond=None)[0]
        ry = yr - Zo @ np.linalg.lstsq(Zo, yr, rcond=None)[0]
        out[j] = np.corrcoef(rx, ry)[0, 1]
    return out


def figure_R3(N=1500):
    '''Response Figure R3 (file figs/figR3.pdf): LHS/PRCC sensitivity.'''
    keys = list(TABLE1_RANGES)
    sampler = qmc.LatinHypercube(d=len(keys), seed=SEED)
    U = sampler.random(N)
    X = np.zeros_like(U)
    for j, k in enumerate(keys):
        lo, hi = TABLE1_RANGES[k]
        X[:, j] = lo + (hi - lo) * U[:, j]
    Ist = np.zeros(N); Ast = np.zeros(N)
    for i in range(N):
        e = endemic({**DEFAULT, **{k: X[i, j] for j, k in enumerate(keys)}})
        Ist[i] = 0.0 if e is None else e['I']
        Ast[i] = 0.0 if e is None else e['A']
    pI, pA = prcc(X, Ist), prcc(X, Ast)

    labels = [r'$\beta$', r'$\kappa_1^0$', r'$\kappa_2^0$',
              r'$\delta_1^0$', r'$\delta_2^0$', r'$\rho$', r'$\omega$']
    x = np.arange(len(keys))
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.3), sharey=True)
    for ax, v, ttl, col in [(axs[0], pI, r'PRCC for $I^*$', CI),
                            (axs[1], pA, r'PRCC for $A^*$', CA)]:
        ax.bar(x, v, width=0.62, color=col, alpha=0.85)
        ax.axhline(0, color='k', lw=0.6)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylim(-1, 1); ax.set_title(ttl, pad=3)
        panel(ax)
    axs[0].set_ylabel('PRCC')
    fig.tight_layout(pad=0.4, w_pad=1.0)
    fig.savefig(OUT + 'figR3.pdf')
    plt.close(fig)

    vals = {k: (round(float(a), 3), round(float(b), 3))
            for k, a, b in zip(keys, pI, pA)}
    json.dump(vals, open('prcc_values.json', 'w'), indent=1)
    print('PRCC (I*, A*) per parameter:')
    for k, v in vals.items():
        print(f'  {k:6s}: {v[0]:+.3f}, {v[1]:+.3f}')
    return keys, pI, pA


# ================================================================== Fig R4
def metrics(p, T=1200, horizon=800.0):
    '''(time to 2nd peak, 2nd-peak prevalence, cum. incidence, damping ratio).'''
    t, Y = simulate(p, T=T, n=12000)
    I = Y[2] + Y[3]
    pk, _ = find_peaks(I, prominence=1e-4)
    t2 = t[pk[1]] if len(pk) >= 2 else np.nan
    I2 = I[pk[1]] if len(pk) >= 2 else np.nan
    mask = t <= horizon
    inc = np.trapezoid(p['beta'] * Y[0][mask] * Y[2][mask], t[mask])
    lam, _ = spectrum(p)
    damp = abs(lam[0].real) / abs(lam[0].imag) if lam[0].imag != 0 else np.nan
    return t2, I2, inc, damp


def figure_R4(n=25):
    '''Response Figure R4 (file figs/figR4.pdf): quantified transient metrics.'''
    rhos = np.linspace(0.1, 1.0, n)
    omegas = np.linspace(0.003, 0.06, n)
    Mr = np.array([metrics({**DEFAULT, 'rho': r}) for r in rhos])
    Mw = np.array([metrics({**DEFAULT, 'omega': w}) for w in omegas])

    fig, axs = plt.subplots(2, 4, figsize=(7.4, 3.6))
    cols = [(r'Time to second peak (d)', 0), (r'Second-peak prevalence', 1),
            (r'Cum. incidence (800 d)', 2), (r'Damping ratio at $E^*$', 3)]
    for j, (lab, k) in enumerate(cols):
        axs[0, j].plot(rhos, Mr[:, k], color=CI)
        axs[1, j].plot(omegas, Mw[:, k], color=CA)
        axs[0, j].set_title(lab, pad=3, fontsize=7.5)
        axs[1, j].set_xlabel(r'Behavioral fatigue $\omega$')
        panel(axs[0, j]); panel(axs[1, j])
    for j in range(4):
        axs[0, j].set_xlabel(r'Awareness response $\rho$')
    for ax, t in zip(axs.ravel(), ['(a)', '(b)', '(c)', '(d)',
                                   '(e)', '(f)', '(g)', '(h)']):
        ax.text(-0.04, 1.2, t, transform=ax.transAxes, fontsize=8.5, va='top')
    fig.tight_layout(pad=0.4, w_pad=0.9, h_pad=1.2)
    fig.savefig(OUT + 'figR4.pdf')
    plt.close(fig)

    json.dump(dict(rho=list(map(float, rhos)),
                   t2_rho=[None if np.isnan(v) else round(float(v), 1) for v in Mr[:, 0]],
                   omega=list(map(float, omegas)),
                   t2_omega=[None if np.isnan(v) else round(float(v), 1) for v in Mw[:, 0]]),
              open('resurgence_values.json', 'w'), indent=1)
    print(f'time to second peak: {Mr[0,0]:.1f} d at rho=0.1  ->  {Mr[-1,0]:.1f} d at rho=1.0'
          '   (letter: 106 -> 156 d)')
    print(f'time to second peak: {Mw[0,0]:.1f} d at omega=0.003  ->  {Mw[-1,0]:.1f} d at omega=0.06'
          '   (letter: 323 -> 72 d)')
    return rhos, omegas, Mr, Mw


# ==================================================================== scans
def scan_maxre(N=40000, wide=False, seed_off=0):
    '''(proportion unstable, max of max Re lambda, parameter set at the max).'''
    rng = np.random.default_rng(SEED + seed_off)
    ranges = WIDE_RANGES if wide else TABLE1_RANGES
    keys = list(ranges)
    unstable = 0
    mx, argmax = -np.inf, None
    for i in range(N):
        p = dict(DEFAULT)
        for k in keys:
            lo, hi = ranges[k]
            p[k] = rng.uniform(lo, hi)
        r = spectrum(p)
        if r is None:
            continue
        m = r[0][0].real
        if m > 0:
            unstable += 1
        if m > mx:
            mx, argmax = m, dict(p)
    return unstable / N, mx, argmax


def run_scans(N=40000):
    '''Seeded eigenvalue scans (referee comment M5a) -> scan_report.json.'''
    scan = {}
    for name, wide, off in [('table1', False, 1), ('wide', True, 2)]:
        prop, mx, arg = scan_maxre(N, wide=wide, seed_off=off)
        scan[name] = dict(proportion_unstable=prop, max_maxRe=float(mx),
                          argmax={k: round(v, 4) for k, v in arg.items()})
        print(f'{name:7s}: {100*prop:.4f}% unstable, max Re lambda = {mx:+.2e}')
        print('         attained at ' +
              ', '.join(f'{k}={v:.3g}' for k, v in arg.items()
                        if k in ('beta', 'k1', 'k2', 'd1', 'd2', 'rho', 'omega')))
    json.dump(scan, open('scan_report.json', 'w'), indent=1)
    return scan


# ============================================================ multistability
def multistability_probe(n_ic=25):
    '''25 initial conditions in three regimes -> multistability_report.json.'''
    rng = np.random.default_rng(SEED)
    regimes = {
        'baseline': dict(DEFAULT),
        'hopf_stable_d1_0.30': {**HOPF, 'd1': 0.30},
        'hopf_cycle_d1_0.60': {**HOPF, 'd1': 0.60},
    }
    report = {}
    for name, p in regimes.items():
        finals = []
        for _ in range(n_ic):
            w = rng.dirichlet(np.ones(5))          # random point on the simplex
            a0 = rng.uniform(0, 1)
            y0 = [w[0], w[1], max(w[2], 1e-3), w[3], w[4], a0]
            s = sum(y0[:5]); y0 = [v / s for v in y0[:5]] + [a0]
            t, Y = simulate(p, y0=y0, T=20000, n=20000)
            I = (Y[2] + Y[3])[t > 16000]
            finals.append((float(I.min()), float(I.max())))
        lo = min(f[0] for f in finals); hi = max(f[1] for f in finals)
        spread_lo = max(f[0] for f in finals) - lo
        spread_hi = hi - min(f[1] for f in finals)
        report[name] = dict(attractor_range=[round(lo, 6), round(hi, 6)],
                            across_ic_spread=[round(spread_lo, 8),
                                              round(spread_hi, 8)])
    json.dump(report, open('multistability_report.json', 'w'), indent=1)
    for name, r in report.items():
        print(f"{name:22s}: attractor I in [{r['attractor_range'][0]:.6f}, "
              f"{r['attractor_range'][1]:.6f}], across-IC spread "
              f"{max(r['across_ic_spread']):.1e}")
    return report


if __name__ == '__main__':
    which = sys.argv[1:] or ['R1', 'R3', 'R4', 'scan', 'multi']
    for w in which:
        print('task', w, flush=True)
        if w == 'R1':
            figure_R1()
        elif w == 'R3':
            figure_R3()
        elif w == 'R4':
            figure_R4()
        elif w == 'scan':
            run_scans()
        elif w == 'multi':
            multistability_probe()
    print('done')
