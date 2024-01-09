from PlotClass import PlotOdom
import matplotlib.pyplot as plt
from plot_ground_truth import InertialExplorerFileHandler
import os



data_set_1 = "20231212_mid360_01_buggy"
# dir_data = os.path.join("/home/slamnuc/Desktop/OdomPlot/data", data_set_1)
# data_set_2 = "lobby_01_NDT"

odom_handler = PlotOdom(data_path="/home/slamnuc/Desktop/OdomPlot/data", name=data_set_1)
odom_handler.zero_initial_heading()

start_time = odom_handler.get_start_time()
end_time = odom_handler.get_end_time()


odometry_poses = odom_handler.get_positions()


gt_handler = InertialExplorerFileHandler()
gt_handler.get_ground_truth_poses_from_file("/home/slamnuc/Desktop/OdomPlot/Python/Ground_truth_lidar_car_test_01_20231212.txt", start_time, end_time)
gt_handler.zero_initial_heading()

ground_truth_poses = gt_handler.get_zeroed_positions()


fig, ax = plt.subplots()
ax.plot(odometry_poses[:,0], odometry_poses[:,1], label="LiDAR Inertial Odometry Mid-360")
ax.plot(ground_truth_poses[:,0], ground_truth_poses[:,1], label="Ground Truth - TC GNSS-IMU SPAN")

ax.set_aspect('equal')
ax.axis('equal')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.legend()

xlim = ax.get_xlim()
xdiff = xlim[1] - xlim[0]
p = 0.01
ax.set_xlim([xlim[0] - xdiff*p, xlim[1]+xdiff*p])



# fig,axes = Plotter1.plot_odometry_2D(label="P2Pl")
# # Plotter2.plot_odometry_2D(axes, label="NDT")
# # axes[0].legend([, "NDT"])




plt.show()