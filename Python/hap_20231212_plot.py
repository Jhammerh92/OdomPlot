from PlotClass import PlotOdom
# from numpy import where
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from plot_ground_truth import InertialExplorerFileHandler, plot_confidence_ellipses
import os

"""Processed data"""
processed_lio_data = "20231212_hap_01"


"""Ground truth data"""
ground_truth_data = "./Python/ground_truth_data/ground_truth_lidar_car_test_02_20231212.txt"
# ground_truth_data = "ground_truth_data/ground_truth_lidar_car_test_01_20231212.txt"


"""Get poses and correct for heading alignment"""
odom_handler = PlotOdom(data_path="./data", name=processed_lio_data)
odom_handler.load_ground_truth_poses_from_file(ground_truth_data)

odom_handler.plot_odometry_interactive(heading_corretion=0.02)

# gt_initial_heading = odom_handler.IEF.get_initial_heading()
# ground_truth_poses = odom_handler.IEF.get_zeroed_positions()
# ground_truth_travelled_dist = odom_handler.IEF.get_travelled_dist()
# # ground_truth_stds = gt_handler.get_stds()

# odom_handler.rotate_to_heading(gt_initial_heading - 0.02)
# odometry_poses = odom_handler.get_positions()
# odometry_travelled_dist = odom_handler.get_travelled_dist()

# data_deltas = np.max(ground_truth_poses, axis=0) - np.min(ground_truth_poses, axis=0)

# """Plots"""
# fig, axes = plt.subplots(2,1, gridspec_kw={'height_ratios': [5, 1]})
# ax = axes[0]

# ax.scatter(ground_truth_poses[:,0], ground_truth_poses[:,1],color="C1" ,marker='o', label="_", s=0)
# ax.plot(ground_truth_poses[:,0], ground_truth_poses[:,1],color="C1", label="Ground Truth - TC GNSS-IMU SPAN")

# odom_sc = ax.scatter(odometry_poses[:,0], odometry_poses[:,1], marker='o', color='C0', label="_", s=0)
# ax.plot(odometry_poses[:,0], odometry_poses[:,1], color='C0',label="LiDAR Inertial Odometry")

# xlim = ax.get_xlim()
# xdiff = xlim[1] - xlim[0]
# p = 0.01
# ax.set_xlim([xlim[0] - xdiff*p, xlim[1]+xdiff*p])

# red_point = ax.scatter(0, 0, color='r')
# red_point.set_visible(False)
# red_point_vline = ax.axvline(x=0)
# red_point_hline = ax.axhline(y=0)
# red_point_vline.set_visible(False)
# red_point_hline.set_visible(False)
# # plot_confidence_ellipses(ax, ground_truth_poses[:,:2], ground_truth_stds[:,:2])

# ax.set_aspect('equal')
# ax.axis('equal')
# ax.set_xlabel('East [m]')
# ax.set_ylabel('North [m]')
# ax.legend()


# ax_height = axes[1]
# ax_height.plot(odometry_travelled_dist, odometry_poses[:,2], label="LiDAR Inertial Odometry")
# height_sc = ax_height.scatter(odometry_travelled_dist, odometry_poses[:,2], marker='o', color='C0', label="_", s=0)

# ax_height.plot(ground_truth_travelled_dist, ground_truth_poses[:,2], label="LiDAR Inertial Odometry")

# height_vline = ax_height.axvline(x=0)
# height_vline.set_visible(False)

# ax_height.set_xlabel('Travelled distance [m]')
# ax_height.set_ylabel('Height [m]')



# fig_zoom, ax_zoom = plt.subplots(figsize=(4,4),)
# ax_zoom.scatter(odometry_poses[:,0], odometry_poses[:,1], marker='o', color='C0', label="_", s=2)
# ax_zoom.plot(odometry_poses[:,0], odometry_poses[:,1], color='C0',label="LiDAR Inertial Odometry")

# ax_zoom.scatter(ground_truth_poses[:,0], ground_truth_poses[:,1],color="C1" ,marker='o', label="_", s=2)
# ax_zoom.plot(ground_truth_poses[:,0], ground_truth_poses[:,1],color="C1", label="Ground Truth - TC GNSS-IMU SPAN")

# red_point_zoom = ax_zoom.scatter(0, 0, color='r')

# ax_zoom.set_aspect('equal')
# ax_zoom.legend()


# """Interactave plots callback on mouse hover"""

# def set_visable_all(bool):
#     red_point.set_visible(bool)
#     red_point_vline.set_visible(bool)
#     red_point_hline.set_visible(bool)
#     height_vline.set_visible(bool)

# def red_point_update(idx):
#     pos = odom_sc.get_offsets()[idx]
#     red_point.set_offsets(pos)
#     red_point_zoom.set_offsets(pos)
#     x,y = pos.data
#     red_point_vline.set_xdata(x)
#     red_point_hline.set_ydata(y)
#     zoom_dist = np.max(data_deltas[:2]) / 20.0 / 2.0
#     ax_zoom.set_xlim(x - zoom_dist, x + zoom_dist)
#     ax_zoom.set_ylim(y - zoom_dist, y + zoom_dist)

# def vline_update(idx):
#     height_vline.set_xdata(odometry_travelled_dist[idx])

# def get_closest_data_idx(x,y):
#     mouse_xy = np.array([x,y])
#     dist_to_mouse = np.linalg.norm(odometry_poses[:,:2] - mouse_xy, axis=1, ord=1)
#     idx = np.where(abs(dist_to_mouse) == min(abs(dist_to_mouse)))[0].item()
#     return idx

# def hover(event):
#     if event.inaxes == ax: # checks if the mouse/"event" is inside the desired plot axis
#         idx = get_closest_data_idx(event.xdata, event.ydata)
#         red_point_update(idx) # updates the plot annotations by the index found
#         vline_update(idx) # updates the plot annotations by the index found
#         set_visable_all(True)
#         fig.canvas.draw_idle() # redraws the plot when able to
#         fig_zoom.canvas.draw_idle()
#     elif event.inaxes == ax_height: # checks if the mouse/"event" is inside the desired plot axis
#         idx = np.where(abs(odometry_travelled_dist - event.xdata)==min(abs(odometry_travelled_dist - event.xdata)))[0].item()
#         red_point_update(idx) # updates the plot annotations by the index found
#         vline_update(idx) # updates the plot annotations by the index found
#         set_visable_all(True)
#         fig.canvas.draw_idle() # redraws the plot when able to
#         fig_zoom.canvas.draw_idle()
#     else: # else happens when mouse is not over any point in the plot
#         set_visable_all(False)
#         fig.canvas.draw_idle() # redraws the plot when able to
#         fig_zoom.canvas.draw_idle()
        

# fig.canvas.mpl_connect("motion_notify_event", hover) # checks the open plot window for mouse events and triggers the "hover" callback function on event









# ax2 = odom_handler.plot_ang_bias()
# ax2 = odom_handler.plot_fitness_boxplot()
# ax2 = odom_handler.plot_fitness()
# Plotter2.plot_odometry_2D(axes, label="NDT")
# # axes[0].legend([, "NDT"])




plt.show()