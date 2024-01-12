from PlotClass import PlotOdom
import matplotlib.pyplot as plt
from plot_ground_truth import InertialExplorerFileHandler, plot_confidence_ellipses
import os

"""Processed data"""
processed_lio_data = "20231212_mid70"

# data_set_mid360 = "20231212_mid360'"
# data_set_hap = "20231212_HAP"
# data_set_test = "20240105_185100"'
# data_set_test = "20240105_200426"


"""Ground truth data"""
ground_truth_data = "./Python/ground_truth_data/ground_truth_lidar_car_test_01_20231212.txt"
# ground_truth_data = "ground_truth_data/ground_truth_lidar_car_test_01_20231212.txt"


"""Get poses"""
odom_handler = PlotOdom(data_path="./data", name=processed_lio_data)
start_time = odom_handler.get_start_time()
end_time = odom_handler.get_end_time()

gt_handler = InertialExplorerFileHandler()
gt_handler.get_ground_truth_poses_from_file(ground_truth_data, start_time, end_time)
gt_initial_heading = gt_handler.get_initial_heading()
ground_truth_poses = gt_handler.get_zeroed_positions()
ground_truth_stds = gt_handler.get_stds()
ground_truth_travelled_dist = gt_handler.get_travelled_dist()

odom_handler.rotate_to_heading(gt_initial_heading - 0.05)
odometry_poses = odom_handler.get_positions()
odometry_travelled_dist = odom_handler.get_travelled_dist()

"""Plot"""
fig, axes = plt.subplots(1,2)
ax = axes[0]
ax.plot(odometry_poses[:,0], odometry_poses[:,1], label="LiDAR Inertial Odometry")
ax.plot(ground_truth_poses[:,0], ground_truth_poses[:,1], label="Ground Truth - TC GNSS-IMU SPAN")
# plot_confidence_ellipses(ax, ground_truth_poses[:,:2], ground_truth_stds[:,:2])

ax.set_aspect('equal')
ax.axis('equal')
ax.set_xlabel('East [m]')
ax.set_ylabel('North [m]')
ax.legend()


ax = axes[1]
ax.plot(odometry_travelled_dist, odometry_poses[:,2], label="LiDAR Inertial Odometry")
ax.plot(ground_truth_travelled_dist, ground_truth_poses[:,2], label="LiDAR Inertial Odometry")


xlim = ax.get_xlim()
xdiff = xlim[1] - xlim[0]
p = 0.01
ax.set_xlim([xlim[0] - xdiff*p, xlim[1]+xdiff*p])



# fig,axes = Plotter1.plot_odometry_2D(label="P2Pl")
# # Plotter2.plot_odometry_2D(axes, label="NDT")
# # axes[0].legend([, "NDT"])




plt.show()