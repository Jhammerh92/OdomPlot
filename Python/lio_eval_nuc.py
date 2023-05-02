from PlotClass import PlotOdom
import matplotlib.pyplot as plt
import os
import numpy as np


dir_data = os.path.realpath("data")
data_set_1 = ""
# data_set_2 = "lobby_01"
abs_path = "/home/slamnuc/temp_saved_odometry_data/odometry"
files = sorted(os.listdir(abs_path))
print(files)
file_path = os.path.join(abs_path,files[-1])
file_path = os.path.join(abs_path, "20230428-173134_run_data.csv")
Plotter1 = PlotOdom(abs_csv_path=file_path,  name=data_set_1, save_plots=False)
# Plotter2 = PlotOdom(name=data_set_2)

Plotter1.plot_acc_bias()
Plotter1.plot_ang_bias()

# fig, axes = plt.subplots(1,2, num="Boxplots", sharey=True)
# Plotter1.plot_fitness_boxplot()
Plotter1.plot_fitness()

vel_axes = Plotter1.plot_velocity(label="LO")
Plotter1.plot_observer_velocity(vel_axes, label="GO")



Plotter1.plot_translation_cov()
# Plotter2.plot_fitness_boxplot(axes[1])
# fig.set_figwidth(6)

fig,axes = Plotter1.plot_odometry_2D(label="P2Pl")

fig,axes = Plotter1.plot_odometry_2D_timewise(label="P2Pl")

# fig, axes = plt.subplots(3,1, num="Odometry Fitness overlay", sharey=True, sharex=True)
# Plotter1.plot_odometry_colorbar_2D(c_values=np.log(Plotter1.fitness),ax=axes[0],cbar_label="ICP Fitness", cbar_unit='[log(m)]')
# Plotter1.plot_odometry_colorbar_2D(c_values=np.log(Plotter1.residual_norm),ax=axes[1],cbar_label="Translation Residual     _", cbar_unit='[log(m)]')
# Plotter1.plot_odometry_colorbar_2D(c_values=np.log(Plotter1.residual_qnorm),ax=axes[2],cbar_label="Rotational Residual     _", cbar_unit='[log(a.u.)]')

Plotter1.plot_odometry_colorbar_2D(c_values=Plotter1.time,ax=None,cbar_label="Time     _", cbar_unit='[s]')
# Plotter2.plot_odometry_2D(axes, label="NDT")
# axes[0].legend([, "NDT"])

# Plotter1.save_figs(r"/Users/jhh/Library/CloudStorage/Dropbox/Apps/Overleaf/LiDAR and IMU Integration for Robust Navigation in GNSS Denied Environments - Master's Thesis/fig/lio_eval/plots")
# Plotter1.save_figs()
plt.show()
# plt.close('all')
