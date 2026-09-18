import numpy as np
from blackjax.diagnostics import potential_scale_reduction
import plot_ghc_mcmc as pgm
#from importlib import reload
#reload(pgm)

def main():
    chain1 = pgm.MCMCResult('output/Q1/HIGHZ_NGC_Q1_500_k03_cov_chi2.dat', 'output/Q1/HIGHZ_NGC_Q1_500_k03_others_cov_chi2.dat')
    sigma_prop, starts = chain1.prop_next_run(n_start=4, spread=2.0, jitter=1.0)
    print(f"acceptance = {chain1.acceptance:.4f}")
    print("proposal covariance =\n", sigma_prop)
    print("start points =\n", starts)
    chain1.mcplot(pair=(0, 1), ax=None, savefig=None)

    chains = [chain1]
    nburnin = 0
    pgm.gd_mcplot(chains, nburnin=nburnin, savefig=None)
    if len(chains) >= 2:
        arr = np.stack([c.X[nburnin:] for c in chains])
        print("R-hat =", potential_scale_reduction(arr))
    else:
        print("R-hat: needs 2 or more chains")

if __name__ == "__main__":
    main()