import pk_compare as pkc
import matplotlib.pyplot as plt
from importlib import reload
reload(pkc)

obs_data = pkc.PowerSpectrumData(z_type='HIGH', ns_type='NGC')
obs_data.load_boss_data(base_path="BOSSmultipoles/", k_max=0.40)

model1 = pkc.PowerSpectrumModel(vmax=455.20, delta_vmax=82.60, Q_id=0)
model1.load_sim_data(obs_data=obs_data, base_path="psout")

pkc.plot_power_spectrum(obs_data=obs_data, sim_data=model1, save_path = "plot/Q0/test.png")
