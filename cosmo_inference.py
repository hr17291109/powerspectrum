import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler
from getdist import MCSamples, plots
from scipy.stats import multivariate_normal
from tqdm import tqdm

def read_paramlist(filename):
    f=open(filename,'r')
    paramdic = pd.Series()
    for line0 in f:
        columns = line0.split(' ')
        for i in columns:
            paramdic[columns[0]] = float(columns[-1].split('\n')[0])
    f.close()
    return paramdic

def combine_data(ns, highlow):
    combined_data = []
    for i in range(30):
        data = np.loadtxt('output/Q'+ str(i) + '/'+highlow+'Z_'+ns+'_Q' + str(i) + '_1000_cov_chi2.dat', skiprows=1)
        p = read_paramlist(f"params/Q{str(i).zfill(4)}_input_params.ini")
        best_row = data[np.argmin(data[:,2])]

        combined_data.append({
            'set_id': f'Q{i}',
            'omega_b':   p['omega_b'],
            'omega_cdm': p['omega_cdm'],
            'omega_m': p['Omega_m'],
            'w0': p['w0'],
            'As': p['ln(10^10As)'],
            'ns': p['ns'],
            'delv': best_row[0],
            'vmax': best_row[1],
            'chi':  best_row[2],
            'logz': -best_row[2] / 2
        })

    return combined_data

def prior(theta, mu_prior, cov_prior):
    return multivariate_normal.pdf(theta, mean=mu_prior, cov=cov_prior)

def prior(theta, prior_lo, prior_hi):
    if np.all(theta > prior_lo) and np.all(theta < prior_hi):
        return 1.0
    return 0.0

def predict_chi(theta_array, scaler_X, scaler_y, gpr):
    X_s = scaler_X.transform(np.atleast_2d(theta_array))
    y_s, y_std_s = gpr.predict(X_s, return_std=True)
    chi_pred = scaler_y.inverse_transform(y_s.reshape(-1, 1)).ravel()
    chi_std  = y_std_s * scaler_y.scale_[0]
    return chi_pred, chi_std

def cosomo_mcmc(rng, covmat, current_theta, current_chi2, other_arr, theta_arr, scaler_X, scaler_y, gpr, prior_lo, prior_hi, alpha=1.0):
    proposed_theta = rng.multivariate_normal(current_theta, covmat)
    chi_pred_arr, chi2_std = predict_chi(proposed_theta, scaler_X, scaler_y, gpr)
    proposed_chi2 = chi_pred_arr[0] + alpha * chi2_std[0]
    prior_prop = prior(proposed_theta, prior_lo, prior_hi)
    prior_curr = prior(current_theta, prior_lo, prior_hi)

    log_r = -(proposed_chi2 - current_chi2) / 2.0
    if np.log(rng.random()) < log_r:
        th_copy = proposed_theta.copy()
        th_copy = th_copy.tolist()
        th_copy.append(proposed_chi2)
        theta_arr.append(th_copy)
        return proposed_theta, proposed_chi2
    else:
        th_copy = current_theta.tolist() if isinstance(current_theta, np.ndarray) else list(current_theta)
        th_copy.append(current_chi2)
        theta_arr.append(th_copy)
        th_copy1 = proposed_theta.copy()
        th_copy1 = th_copy1.tolist()
        th_copy1.append(proposed_chi2)
        other_arr.append(th_copy1)
        return current_theta, current_chi2

def GPmcmc(df, Nsteps, nburnin, kn=1, step=0.05, seed=12345):
    rng = np.random.default_rng(seed)
    param_names = ["omega_m", "w0", "As", "ns"]
    X_raw = df[param_names].values
    y_raw = df["chi"].values

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X_raw)
    y_scaled = scaler_y.fit_transform(y_raw.reshape(-1, 1)).ravel()

    if kn == 1:
        kernel = (
            ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3))
            * RBF(length_scale=np.ones(X_scaled.shape[1]),
                length_scale_bounds=(1e-2, 1e2))
        )
    elif kn == 2:
        kernel = (
            ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3))
            * RBF(length_scale=np.ones(X_scaled.shape[1]),
                length_scale_bounds=(1e-2, 1e3))
        )
    elif kn == 3:
        kernel = (
            ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3))
            * RBF(length_scale=np.ones(X_scaled.shape[1]),
                length_scale_bounds=(1e-1, 1e7))
        )
    elif kn == 4:
        kernel = (
            ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3))
            * RBF(length_scale=np.ones(X_scaled.shape[1]),
                length_scale_bounds=(1e-2, 1e3))
        )
    elif kn == 5:
        kernel = (
            ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3))
            * Matern(length_scale=np.ones(X_scaled.shape[1]),
                    length_scale_bounds=(1e-2, 1e2),
                    nu=2.5)
        )
    elif kn == 6:
        kernel = (
            ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3))
            * Matern(length_scale=np.ones(X_scaled.shape[1]),
                    length_scale_bounds=(1e-2, 1e7),
                    nu=2.5)
        )

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=20,
        normalize_y=False,
        random_state=42
    )
    gpr.fit(X_scaled, y_scaled)

    #mu_prior    = X_raw.mean(axis=0)
    #cov_prior   = np.cov(X_raw.T)

    data_range = X_raw.max(axis=0) - X_raw.min(axis=0)
    margin = 0.1 * data_range
    prior_lo = X_raw.min(axis=0) - margin
    prior_hi = X_raw.max(axis=0) + margin

    covmat = np.cov(X_raw.T) * step
    #covmat = cov_prior*step

    best_row = df.loc[df["chi"].idxmin()]
    current_theta = [best_row['omega_m'], best_row['w0'], best_row['As'], best_row['ns']]
    current_chi2  = best_row['chi']
    #best_om = df.loc[df["chi"].idxmin()]['omega_m']
    #best_w0 = df.loc[df["chi"].idxmin()]['w0']
    #best_As = df.loc[df["chi"].idxmin()]['As']
    #best_ns = df.loc[df["chi"].idxmin()]['ns']
    #best_chi2 = df.loc[df["chi"].idxmin()]['chi']
    #current_theta = [best_om, best_w0, best_As, best_ns]
    theta_arr = []
    th_copy = current_theta.copy()
    th_copy.append(current_chi2)
    theta_arr.append(th_copy)
    #current_chi2 = best_chi2
    other_arr = []

    for i in tqdm(range(Nsteps)):
        current_theta, current_chi2 = cosomo_mcmc(rng, covmat, current_theta, current_chi2, other_arr, theta_arr, scaler_X, scaler_y, gpr, prior_lo, prior_hi, alpha=0.5)

    theta_arr = np.array(theta_arr)

    labels  = [r"\Omega_m", r"w_0", r"A_s", r"n_s"]
    names   = ["omega_m",   "w0",  "As",  "ns"]

    mc_samples = MCSamples(
        samples=theta_arr[nburnin:,:4],
        names=names,
        labels=labels,
        # settings={'smooth_scale_1D': 0.3, 'smooth_scale_2D': 0.3},
        # label="Posterior"
    )

    total_steps = len(theta_arr) - 1 
    rejected_steps = len(other_arr)
    accepted_steps = total_steps - rejected_steps
    acceptance_rate = accepted_steps / total_steps

    print(f"総ステップ数: {total_steps}")
    print(f"採択回数: {accepted_steps}")
    print(f"棄却回数: {rejected_steps}")
    print(f"平均採択率: {acceptance_rate:.4f} ({acceptance_rate * 100:.1f}%)")
    return mc_samples    