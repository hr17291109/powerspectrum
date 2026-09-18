import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from getdist import MCSamples, plots

LABELS = {"delta_v": r"\Delta V_{\rm max}", "Vmax_threshould": r"V_{\rm max,th}"}

class MCMCResult:
    labels: list
    params: list
    chi2_col: str
    X: np.ndarray               # accepted galaxy-halo connection parameter list
    chi2: np.ndarray            # accepted chi square list
    Xr: np.ndarray              # rejected galaxy-halo connection parameter list
    chi2r: np.ndarray           # rejected chi square list
    xmin: np.ndarray
    chi2min: float
    mean: np.ndarray
    cov: np.ndarray
    n_reject: int
    n_accept: int

    def __init__(self, chain_path, others_path):
        ch = pd.read_csv(chain_path, sep=r"\s+")
        self.chi2_col = ch.columns[-1]
        self.params = [c for c in ch.columns if c != self.chi2_col]
        self.labels = [LABELS.get(p, p) for p in self.params]
 
        self.X = ch[self.params].values.astype(float)
        self.chi2 = ch[self.chi2_col].values.astype(float)
        
        ot = pd.read_csv(others_path, sep=r"\s+")
        self.Xr = ot[self.params].values.astype(float)
        self.chi2r = ot[self.chi2_col].values.astype(float)
 
        imin = int(np.argmin(self.chi2))
        self.xmin, self.chi2min = self.X[imin], self.chi2[imin]
        self.mean = self.X.mean(axis=0)
        self.cov = np.cov(self.X, rowvar=False)

        n_rej = len(self.Xr)
        self.n_reject, self.n_accept = n_rej, len(self.X) - 1 - n_rej
        self.acceptance = self.n_accept / (len(self.X) - 1)
        dup = int((self.X[1:] == self.X[:-1]).all(axis=1).sum())
        if dup != n_rej:
            raise ValueError(f"duplicated rows ({dup}) != others rows ({n_rej})")

    def prop_next_run(self, n_start=4, spread=2.0, jitter=1.0):
        d = self.X.shape[1]
        Sigma_prop = (2.38 ** 2 / d) * self.cov
 
        w, v = np.linalg.eigh(self.cov)
        sig = np.sqrt(w)
        wide, narrow = -1, 0
        offs = np.linspace(-spread, spread, n_start)
        sgn = np.where(np.arange(n_start) % 2 == 0, 1.0, -1.0)
        starts = np.array([self.xmin
                           + o * sig[wide] * v[:, wide]
                           + s * jitter * sig[narrow] * v[:, narrow]
                           for o, s in zip(offs, sgn)])
        return Sigma_prop, starts

    def mcplot(self, pair=(0, 1), ax=None, savefig=None):
        i, j = [self.params.index(p) if isinstance(p, str) else p for p in pair]
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
        sel = np.ix_([i, j], [i, j])
 
        ax.scatter(self.Xr[:, i], self.Xr[:, j], s=14, c="0.65",
                    alpha=0.6, edgecolors="none",
                    label=f"rejected ({len(self.Xr)})")
        ax.scatter(self.X[:, i], self.X[:, j], s=26, c="royalblue",
                   edgecolors="k", linewidths=0.25, zorder=3,
                   label=f"accepted ({len(self.X)})")
        ax.plot(self.xmin[i], self.xmin[j], marker="*", ms=24, mfc="red",
                mec="k", mew=1.2, zorder=5,
                label=rf"minimum $\chi^2={self.chi2min:.3f}$")
 
        w, v = np.linalg.eigh(self.cov[sel])
        angle = np.degrees(np.arctan2(v[1, -1], v[0, -1]))
        for k, ls in [(1, "-"), (2, "--")]:
            ax.add_patch(Ellipse((self.mean[i], self.mean[j]),
                                 2 * k * np.sqrt(w[-1]), 2 * k * np.sqrt(w[0]),
                                 angle=angle, fill=False,
                                 color="crimson", lw=2, ls=ls, zorder=4))
        ax.plot([], [], color="crimson", label=r"chain 1, 2$\sigma$")
 
        ax.set_xlabel(f"${self.labels[i]}$")
        ax.set_ylabel(f"${self.labels[j]}$")
        ax.legend(fontsize=9, loc="best")
        if savefig:
            ax.figure.tight_layout()
            ax.figure.savefig(savefig, dpi=140, bbox_inches="tight")
        return ax

    def __repr__(self):
        return (f"MCMCResult(n={len(self.X)}, acceptance={self.acceptance:.3f}, "
                f"chi2min={self.chi2min:.3f})")

def gd_mcplot(chains, nburnin=0, savefig=None):
    names = chains[0].params
    samples = [MCSamples(samples=c.X[nburnin:], names=c.params, labels=c.labels, label=f"chain {i}")
               for i, c in enumerate(chains, 1)]

    g = plots.get_single_plotter(width_inch=5, ratio=1)
    g.plot_2d(samples, names[0], names[1], filled=True)
    if savefig:
        g.export(savefig)
    return g

