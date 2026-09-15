import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional, List
from scipy.linalg import cho_factor, cho_solve
import pk_tools

class PowerSpectrumData:
    def __init__(self, z_type: str, ns_type: str):
        self.z_type = z_type         # 'LOW' or 'HIGH'
        self.ns_type = ns_type       # 'SGC' or 'NGC'
        self.k = None                # wavenumber
        self.obs_power = None        # BOSS data power spectrum multipoles
        self.M = None                # the wide-angle transformation matrix
        self.cov_mat = None          # the covariance matrix
        self.window_mat = None       # the window function matrix
        self.cov_inv_mat = None
        self.k_mask = None

        if (self.z_type == 'LOW'):
            self.z = 1               #z = 0.38
        elif (self.z_type == 'HIGH'):
            self.z = 3               #z = 0.61
        else:
            raise ValueError("z_type must be 'LOW' or 'HIGH'")

        if self.ns_type not in ['NGC', 'SGC']:
            raise ValueError("ns_type must be 'NGC' or 'SGC'")

    def load_boss_data(self, base_path: str, k_max: float = 0.40):
        pk = pk_tools.read_power(f"{base_path}ps1D_BOSS_DR12_{self.ns_type}_z{str(self.z)}_COMPnbar_TSC_700_700_700_400_renorm.dat")
        self.M = np.loadtxt(f"{base_path}M_BOSS_DR12_{self.ns_type}_z{str(self.z)}_V6C_1_1_1_1_1_1200_2000.matrix")
        self.window_mat = np.loadtxt(f"{base_path}W_BOSS_DR12_{self.ns_type}_z{str(self.z)}_V6C_1_1_1_1_1_10_200_2000_averaged_v1.matrix")
        self.cov_mat = np.loadtxt(f"{base_path}C_2048_BOSS_DR12_{self.ns_type}_z{str(self.z)}_V6C_1_1_1_1_1_10_200_200_prerecon.matrix")

        self.k = pk['k']
        n_bins = len(self.k)

        self.k_mask = self.k <= k_max
        valid_k_indices = np.where(self.k_mask)[0]

        # select rows and cols to delete
        rows_to_delete = []
        for l_idx in range(5):
            if l_idx in [1, 3]:
                rows_to_delete.extend(range(l_idx * n_bins, (l_idx + 1) * n_bins))
            else:
                invalid_k = np.where(~self.k_mask)[0]
                rows_to_delete.extend([i + (l_idx * n_bins) for i in invalid_k])

        rows_to_delete = np.array(rows_to_delete)
        cols_to_delete = rows_to_delete.copy()

        #rows_to_delete = np.r_[n_bins : 2*n_bins, 3*n_bins : 4*n_bins]
        #cols_to_delete = np.r_[n_bins : 2*n_bins, 3*n_bins : 4*n_bins]

        # create indexes except rows and cols
        total_bins = 5 * n_bins
        rows_to_keep = np.delete(np.arange(total_bins), rows_to_delete)
        cols_to_keep = np.delete(np.arange(total_bins), cols_to_delete)

        # create a new matrix from the indexes
        C2 = self.cov_mat[rows_to_keep][:, cols_to_keep]
        #self.cov_inv_mat = np.linalg.inv(C2)
        c, low = cho_factor(C2)
        self.cov_inv_mat = cho_solve((c, low), np.eye(C2.shape[0]))

        self.obs_power = np.concatenate((
            pk['pk0'][valid_k_indices], 
            pk['pk2'][valid_k_indices], 
            pk['pk4'][valid_k_indices]
        ))

        #self.obs_power = np.concatenate(pk['pk0'], pk['pk2'], pk['pk4'])
        self.k_valid = self.k[valid_k_indices]

@dataclass
class PowerSpectrumModel:
    vmax: float
    delta_vmax: float
    Q_id: str
    k_sim: Optional[np.ndarray] = field(default=None, init=False)
    psim0: Optional[np.ndarray] = field(default=None, init=False)
    psim2: Optional[np.ndarray] = field(default=None, init=False)
    psim4: Optional[np.ndarray] = field(default=None, init=False)
    psim:  Optional[np.ndarray] = field(default=None, init=False)

    def _get_padding_zeros(self, z_types: str) -> List[float]:
        pad_2_qs = [1, 4, 10, 13, 1.2]
        if (self.Q_id in pad_2_qs) or (z_types == 'HIGH' and self.Q_id in [17, 24]) or (z_types == 'LOW' and self.Q_id in [19]):
            return [0.000, 0.000]
        elif (z_types == 'HIGH' and self.Q_id in [9, 12, 22]) or (z_types == 'LOW' and self.Q_id in [26]) or (self.Q_id in [16, 29]):
            return [0.000, 0.000, 0.000, 0.000]
        else:
            return [0.000, 0.000, 0.000]

    def load_sim_data(self, obs_data, base_path: str):
        file_prefix = f"{base_path}/Q{self.Q_id}/{obs_data.z_type}Z_{obs_data.ns_type}_Q{self.Q_id}_{self.vmax}_{self.delta_vmax}_"
        psim0_raw = np.loadtxt(f"{file_prefix}pk0.dat")
        psim2_raw = np.loadtxt(f"{file_prefix}pk2.dat")
        psim4_raw = np.loadtxt(f"{file_prefix}pk4.dat")

        pad_zeros = self._get_padding_zeros(obs_data.z_type)
        pk00 = np.insert(psim0_raw.T[3], 0, pad_zeros)
        pk02 = np.insert(psim2_raw.T[3], 0, pad_zeros)
        pk04 = np.insert(psim4_raw.T[3], 0, pad_zeros)

        ptf = np.concatenate((pk00, pk02, pk04))
        wmp = obs_data.window_mat @ obs_data.M @ ptf

        n_bins = len(obs_data.k)
        self.k_sim = np.arange(0, 0.4, 0.01)
        self.psim0 = wmp[0 : n_bins]
        self.psim2 = wmp[n_bins*2 : n_bins*3]
        self.psim4 = wmp[n_bins*4 : n_bins*5]

        valid_mask = obs_data.k_mask
        self.psim = np.concatenate((self.psim0[valid_mask], self.psim2[valid_mask], self.psim4[valid_mask]))

def calc_chi2(sim_data, obs_data):
    diff = obs_data.obs_power - sim_data.psim
    x2 = diff.T @ obs_data.cov_inv_mat @ diff
    return x2


def plot_power_spectrum(obs_data=None, sim_data=None, save_path = None):
    if obs_data is None and sim_data is None:
        raise ValueError("Must provide at least one of 'obs_data' or 'sim_data'.")

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlabel(r"$k[h{\rm Mpc}^{-1}]$", fontsize=20)
    ax.set_ylabel(r"$kP_\ell(k)[h^{-2}{\rm Mpc}^2]$", fontsize=20)
    ax.tick_params(labelsize=15)

    colors = {0: 'blue', 2: 'red', 4: 'green'}

    if obs_data is not None:
        k_obs = obs_data.k_valid
        n_bins_original = len(obs_data.k)
        variances = np.diagonal(obs_data.cov_mat)

        for i, l in enumerate([0, 2, 4]):
            offset = (i * 2 if l != 4 else 4) * n_bins_original
            err = np.sqrt(variances[offset : offset + n_bins_original])[obs_data.k_mask]
            
            len_k = len(k_obs)
            p_obs = obs_data.obs_power[i * len_k : (i + 1) * len_k]

            ax.errorbar(k_obs, k_obs * p_obs, yerr=k_obs * err, capsize=5, fmt='o', 
                        color='w', markeredgecolor='black', ecolor='black')
            ax.plot(k_obs, k_obs * p_obs, c=colors[l], label=rf"BOSS $\ell={l}$")

    if sim_data is not None:
        for i, l in enumerate([0, 2, 4]):
            p_sim_full = getattr(sim_data, f"psim{l}")
            
            if obs_data is not None:
                k_obs = obs_data.k_valid
                p_sim_masked = p_sim_full[obs_data.k_mask]
                ax.plot(k_obs, k_obs * p_sim_masked, '--', c=colors[l], label=rf"Sim $\ell={l}$")
            else:
                k_sim = sim_data.k_sim
                ax.plot(k_sim, k_sim * p_sim_full, '--', c=colors[l], label=rf"Sim $\ell={l}$")

    ax.legend(fontsize=15)
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()

