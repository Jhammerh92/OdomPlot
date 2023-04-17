from PlotClass import PlotOdom
import matplotlib.pyplot as plt
import os

dir_data = os.path.realpath("data")
# data_set_1 = "lobby_01_failure"
data_set_1 = "328_failure"
# data_set_2 = "lobby_01"

Plotter1 = PlotOdom(name=data_set_1)
# Plotter2 = PlotOdom(name=data_set_2)

Plotter1.plot_acc_bias()
Plotter1.plot_ang_bias()

# fig, axes = plt.subplots(1,2, num="Boxplots", sharey=True)
Plotter1.plot_fitness_boxplot()
Plotter1.plot_fitness()

Plotter1.plot_translation_cov()
# Plotter2.plot_fitness_boxplot(axes[1])
# fig.set_figwidth(6)

fig,axes = Plotter1.plot_odometry_2D(label="P2Pl")

fig,axes = Plotter1.plot_odometry_2D_timewise(label="P2Pl")

fig, axes = plt.subplots(3,1, num="Odometry Fitness overlay", sharey=True, sharex=True)
Plotter1.plot_odometry_colorbar_2D(c_values=Plotter1.fitness,ax=axes[0],cbar_label="ICP Fitness", cbar_unit='[m]')
Plotter1.plot_odometry_colorbar_2D(c_values=Plotter1.residual_norm,ax=axes[1],cbar_label="Translation Residual     _", cbar_unit='[m]')
Plotter1.plot_odometry_colorbar_2D(c_values=Plotter1.residual_qnorm,ax=axes[2],cbar_label="Rotational Residual     _", cbar_unit='[a.u.]')

Plotter1.plot_odometry_colorbar_2D(c_values=Plotter1.time,ax=None,cbar_label="Time     _", cbar_unit='[s]')
# Plotter2.plot_odometry_2D(axes, label="NDT")
# axes[0].legend([, "NDT"])
plt.show()