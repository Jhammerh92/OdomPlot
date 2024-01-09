from PlotClass import PlotOdom
import matplotlib.pyplot as plt
from plot_ground_truth import InertialExplorerFileHandler, plot_confidence_ellipses
import os

"""Processed data"""
processed_lio_data = "20231114_rooftop"

"""Ground truth data"""
ground_truth_data = "/home/slamnuc/Desktop/OdomPlot/Python/ground_truth_data/ground_truth_roof_test02.txt"

"""Get poses"""
odom_handler = PlotOdom(data_path="/home/slamnuc/Desktop/OdomPlot/data", name=processed_lio_data)
start_time = odom_handler.get_start_time()
end_time = odom_handler.get_end_time()

gt_handler = InertialExplorerFileHandler()
gt_handler.get_ground_truth_poses_from_file(ground_truth_data, start_time, end_time)
gt_initial_heading = gt_handler.get_initial_heading(10)
ground_truth_poses = gt_handler.get_zeroed_positions()
ground_truth_stds = gt_handler.get_stds()

# odom_handler.zero_initial_heading(1.0)
odom_handler.rotate_to_heading(-1.27)
odometry_poses = odom_handler.get_positions()


fig, ax = plt.subplots()
ax.plot(odometry_poses[:,0], odometry_poses[:,1], label="LiDAR Inertial Odometry Mid360")
ax.plot(ground_truth_poses[:,0], ground_truth_poses[:,1], label="Ground Truth - TC GNSS-IMU SPAN")
plot_confidence_ellipses(ax, ground_truth_poses[:,:2], ground_truth_stds[:,:2])

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