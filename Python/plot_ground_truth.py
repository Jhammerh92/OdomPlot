import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

from PlotClass import InertialExplorerFileHandler, rotate_to_nearest_x_axis, rotate_to_x_axis, plot_confidence_ellipses,set_axes_equal
from mpl_toolkits.mplot3d import Axes3D

def plot_odometry_colorbar_2D(positions, c_values, ax:plt.Axes=None, arg='', cmap='plasma', plane='xy',lw=3, cbar_label='', cbar_unit=''):
        # 3d plot of egomotion path - OBS. Z and Y axis are switched, but labelled correctly in plot
        c = positions[:len(c_values), :]
        if arg.lower() == "origo":
            x = [pt[0]-c[0][0] for pt in c]
            y = [-(pt[1]-c[0][1]) for pt in c]
            z = [pt[2]-c[0][2] for pt in c]
        elif arg.lower() == 'kitti':
            x = [pt[0] for pt in c]
            y = [-pt[1] for pt in c]
            z = [pt[2] for pt in c]
        else:
            x = [pt[0] for pt in c]
            y = [pt[1] for pt in c]
            z = [pt[2] for pt in c]
        c_values = np.asarray(c_values)

        if plane == 'xz':
            z_temp = z
            z = y
            y = z_temp
        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1, projection='3d')

        if ax is None:
            fig, ax = plt.subplots(1)

        # ax.plot3D(x, z, y, label='positon')
        # lnWidth = [40 for i in range(len(speed))]
        points = np.array([x, y]).T.reshape((-1, 1, 2))
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(c_values.min(), c_values.max())
        lc = LineCollection(segments, cmap=cmap, norm=norm, antialiaseds=True)
        # Set the values used for colormapping
        lc.set_array(c_values)
        lc.set_linewidth(lw)
        line = ax.add_collection(lc)
        # fig.colorbar(line, ax=ax)

        ax.scatter(x[0], y[0], s=50, color='g')
        ax.scatter(x[-1], y[-1], s=50, color='r')


        # cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        cbar = fig.colorbar(line, ax=ax,aspect=10)
        cbar.set_label(cbar_unit, rotation=270, labelpad=15)
        # cbar.ax.yaxis.set_label_position('right')
        cbar.ax.tick_params(direction="out",labelsize=8)
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.set_title(cbar_label, fontsize=8)


        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.axis('equal')

        # plt.show()
        return fig, ax

def plot_odometry_colorbar_3D(positions, c_values, ax=None, arg='origo', colorscheme='viridis'):
    # 3d plot of egomotion path - OBS. Z and Y axis are switched, but labelled correctly in plot
    c = positions[:len(c_values), :]
    if arg == "origo":
        x = [pt[0]-c[0][0] for pt in c]
        y = [-(pt[1]-c[0][1]) for pt in c]
        z = [pt[2]-c[0][2] for pt in c]
    else:
        x = [pt[0] for pt in c]
        y = [-pt[1] for pt in c]
        z = [pt[2] for pt in c]
    c_values = np.asarray(c_values)

    if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(x[0], ys=z[0], zs=y[0], s=50, c='g')
    ax.scatter(x[-1], ys=z[-1], zs=y[-1], s=50, c='r')

    # ax.plot(x, z, y, label='')

    points = np.array([x, z, y]).T.reshape((-1, 1, 3))
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(c_values.min(), c_values.max())
    lc = Line3DCollection(segments, cmap=colorscheme, norm=norm)
    # Set the values used for colormapping
    lc.set_array(c_values)
    lc.set_linewidth(2)
    line = ax.add_collection(lc)
    fig.colorbar(line, ax=ax)

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Z [m] - Depth')
    ax.set_zlabel('Y [m] - Height')

    set_axes_equal(ax)

if __name__ == "__main__":
    handler1 = InertialExplorerFileHandler()
    handler2 = InertialExplorerFileHandler()

    handler1.load_ground_truth_poses_from_file("./Python/ground_truth_data/ground_truth_lidar_car_test_01_20231212.txt")
    handler2.load_ground_truth_poses_from_file("./Python/ground_truth_data/ground_truth_lidar_car_test_02_20231212.txt")

    positions = np.r_[handler1.positions, handler2.positions]
    SDHeight = np.r_[handler1.SDHeight, handler2.SDHeight]

    # plot_odometry_colorbar_3D(positions, SDHeight)


    fig, ax = plt.subplots()
    ax.plot(SDHeight)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x, y, z, c='r', marker='o')
    ax.plot(positions[:,0], positions[:,1],positions[:,2])
    # set_axes_equal(ax)



    plt.show()