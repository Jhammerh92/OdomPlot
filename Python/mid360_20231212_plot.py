from PlotClass import PlotOdom
import matplotlib.pyplot as plt
import os



data_set_1 = "20231212_mid360_01_buggy_run_data.csv"
dir_data = os.path.join("/home/slamnuc/Desktop/OdomPlot/data", data_set_1)
# data_set_2 = "lobby_01_NDT"

Plotter1 = PlotOdom(absolute_csv_path=dir_data)
# Plotter2 = PlotOdom(name=data_set_2)



# fig, axes = plt.subplots(1,2, num="Boxplots", sharey=True)
# Plotter1.plot_fitness_boxplot(axes[0])
# # Plotter2.plot_fitness_boxplot(axes[1])
# fig.set_figwidth(6)

fig,axes = Plotter1.plot_odometry_2D(label="P2Pl")
# Plotter2.plot_odometry_2D(axes, label="NDT")
# axes[0].legend([, "NDT"])
plt.show()