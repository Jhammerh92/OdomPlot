import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial.transform import Rotation
import os

# fitness = np.loadtxt('/home/slamnuc/Desktop/fitness_all_keyframes.csv', delimiter=",", dtype=float, usecols=[0])
# print(fitness)

def moving_average(a, n=3) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

class PlotOdom:
    def __init__(self, abs_csv_path:os.PathLike=None, name:str=None) -> None:
        if abs_csv_path is None:
            abs_csv_path = os.path.join("data",name + "_run_data.csv")
        assert os.path.isfile(abs_csv_path), f"path does not exist: {abs_csv_path}"

        self.path = abs_csv_path
        self.data = np.genfromtxt(abs_csv_path, dtype=float, delimiter=',', names=True, skip_header=31)


        self.headers = self.data.dtype.names
        for header in self.headers:
            setattr(self, header, self.data[header])
        # self.x = self.data['x']
        self.positions = np.c_[self.x, self.y, self.z]

        self.residual_norm = np.sqrt(np.square( self.residual_x) + np.square(self.residual_y) + np.square(self.residual_z))
        # self.residual_qnorm = np.sqrt(np.square(1- np.abs(self.residual_qw)) + np.square( self.residual_qx) + np.square(self.residual_qy) + np.square(self.residual_qz))
        self.residual_qnorm = np.rad2deg(np.sqrt( np.square( self.residual_qx) + np.square(self.residual_qy) + np.square(self.residual_qz)))
        
        self.fig = None
        self.axes= None

        plt.rcParams['figure.figsize'] = (9, 5)
        plt.rcParams['figure.constrained_layout.use'] = True
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.grid'] = True
        plt.rcParams['lines.linewidth'] = 1.0

        print(plt.rcParams)
        
    def plot_odometry(self):
        plt.figure()
        plt.plot(self.x, self.y)
        # plt.plot(self.x, self.fitness)
        # plt.plot( self.x - self.residual_x, self.y - self.residual_y)

    def plot_velocity(self, axes=None):
        if axes is None:
            fig, axes = plt.subplots(1,1, constrained_layout=True)
        axes.plot( self.time, self.vx,label="Velocity x")
        axes.plot( self.time, self.vy,label="Velocity y")
        axes.plot( self.time, self.vz,label="Velocity z")
        axes.legend()

        return axes

    def plot_observer_velocity(self, axes=None):
        if axes is None:
            fig, axes = plt.subplots(1,1, constrained_layout=True)
        axes.plot( self.time, self.obs_vx,label="Velocity x")
        axes.plot( self.time, self.obs_vy,label="Velocity y")
        axes.plot( self.time, self.obs_vz,label="Velocity z")
        axes.legend()

        return axes

    def plot_translation_cov(self):
        plt.figure()
        plt.plot( self.time, self.cov_x,label="Translation Covariance x")
        plt.plot( self.time, self.cov_y,label="Translation Covariance y")
        plt.plot( self.time, self.cov_z,label="Translation Covariance z")
        plt.legend()

    def _plot_acc_bias(self):
        self.axes.plot( self.time, self.bias_acc_x,label="Acc Bias x")
        self.axes.plot( self.time, self.bias_acc_y,label="Acc Bias y")
        self.axes.plot( self.time, self.bias_acc_z,label="Acc Bias z")
        self.axes.set_ylabel("$b_a$ $[m/s²]$")
        # self.axes.legend()

    def plot_acc_bias(self, fig=None):
        self._general_plot_handle(fig, self._plot_acc_bias, timewise=True)


    def _plot_ang_bias(self):
        self.axes.plot( self.time, self.bias_ang_x,label="Ang Bias x")
        self.axes.plot( self.time, self.bias_ang_y,label="Ang Bias y")
        self.axes.plot( self.time, self.bias_ang_z,label="Ang Bias z")
        self.axes.set_ylabel("$b_g$ $[rad/s]$")
        

    def plot_ang_bias(self, fig=None):
        self._general_plot_handle(fig, self._plot_ang_bias, timewise=True)

    def plot_fitness(self, fig=None):
        self._general_plot_handle(fig, self._plot_fitness, timewise=True)

    def _plot_fitness(self):
        n = 7
        self.axes.plot(self.time,  self.residual_norm*1e3, label="Linear Residual Norm", marker='.', ls='-')
        self.axes.plot(self.time,  self.residual_qnorm*1e2, label="Rotational Residual Norm", marker='.', ls='-')
        self.axes.plot(self.time,  self.fitness*1e3, label="Scan Match Fitness", marker='.',  ls='-')
        self.axes.set_prop_cycle(None)
        # self.axes.plot(self.time[:-(n-1)],  moving_average(self.residual_norm*1e3 , n), label="Linear Residual Norm", lw = 2.0)
        # self.axes.plot(self.time[:-(n-1)],  moving_average(self.residual_qnorm *1e2, n), label="Rotational Residual Norm")
        # self.axes.plot(self.time[:-(n-1)],  moving_average(self.fitness*1e3, n), label="Scan Match Fitness")
        # self.axes.set_xlabel("Time [s]")
        self.axes.set_ylabel("Residual/Fitness [mm] / [1/100°]")


    def plot_fitness_boxplot(self, fig=None):
        self._general_plot_handle(fig, self._plot_fitness_boxplot, timewise=False)

    def custom_boxplot(self, ax:plt.Axes, data, index=0, c='C0'):
        data_list = [data if i==index else np.array([np.nan]) for i in range(index+1)]

        ax.scatter(np.random.normal(index+1, 0.1, len(data)), data, marker='.', s=2.5, alpha=0.5, c=f"C{index}")
        ax.boxplot(data_list, showfliers = False, widths=(0.9), vert=True)

    def _plot_fitness_boxplot(self):
        data = [self.residual_norm*1e3, self.residual_qnorm *100, self.fitness*1e3]

        for i, d in enumerate(data):
            self.custom_boxplot(self.axes, d, i)

        self.axes.set_xticks([1,2,3],['P Norm', 'q Norm', 'Fitness'])
        self.axes.set_ylim([0, 75])
        # self.axes.set_xlim([0, 100])
        self.axes.grid(False)
        self.fig.set_figwidth(3)
        self.axes.set_ylabel("Residual/Fitness [mm] / [1/100°]")

    def _plot_timewise(self, plot_call):
        plot_call()
        self.axes.set_xlabel("Time [s]")

    def _general_plot_handle(self,fig:plt.Figure, plot_call, timewise=True):
        if fig is None:
            self.fig, self.axes = plt.subplots()
        else:
            self.fig = fig
            self.axes = fig.axes

        if timewise:
            self._plot_timewise( plot_call)
        else:
            plot_call()

        if isinstance(self.axes, list):
            for ax in self.axes:
                ax.legend()
        else:
            self.axes.legend()


    



    def plot_odometry_2D(self, axes=None, arg='', label='', **kwargs):
        # 2D plot of egomotion path
        c = self.positions
        if arg == "origo":
            x = [pt[0]-c[0][0] for pt in c]
            y = [pt[1]-c[0][1] for pt in c]
            z = [pt[2]-c[0][2] for pt in c]
        elif arg == "end":
            x = [pt[0]-c[-1][0] for pt in c]
            y = [pt[1]-c[-1][1] for pt in c]
            z = [pt[2]-c[-1][2] for pt in c]
        elif arg.lower() == "kitti":
            x = [pt[2] for pt in c]
            y = [-pt[0] for pt in c]
            z = [-pt[1] for pt in c]
        else:
            x = [pt[0] for pt in c]
            y = [pt[1] for pt in c]
            z = [pt[2] for pt in c]

        fig = None
        if axes is None:
            fig, axes = plt.subplots(2,1, constrained_layout=True)

        axes[0].scatter(x[0], y[0], s=50, color='g', marker='x') # start
        axes[0].scatter(x[-1], y[-1], s=50, color='r', marker='x') # end
        axes[0].plot(x, y, label=label, **kwargs)
        # axes[0].set_aspect('equal', adjustable='box')
        # axes[0].set_aspect('equal')
        axes[0].axis('equal')
        axes[0].set_xlabel('X [m]')
        axes[0].set_ylabel('Y [m]')
        axes[0].legend()

        axes[1].scatter(x[0], z[0], s=50, color='g', marker='x') # start
        axes[1].scatter(x[-1], z[-1], s=50, color='r', marker='x') # end
        axes[1].plot(x, z, **kwargs)
        axes[1].axis('equal')
        # axes[1].set_aspect('equal', adjustable='box')
        # axes[1].set_aspect('equal')
        axes[1].set_xlabel('X [m]')
        axes[1].set_ylabel('Z [m]')


        axes[0].set_title('XY-plane')
        axes[1].set_title('XZ-plane, elevation')

        if (fig is None):
            return axes
        else:
            return fig, axes


    def plot_odometry_colorbar_2D(odometry, c_values, fig, ax=None, arg='', cmap='summer_r', plane='xy'):
        # 3d plot of egomotion path - OBS. Z and Y axis are switched, but labelled correctly in plot
        c = odometry[:len(c_values), :]
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
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(c_values.min(), c_values.max())
        lc = LineCollection(segments, cmap=cmap, norm=norm, antialiaseds=True)
        # Set the values used for colormapping
        lc.set_array(c_values)
        lc.set_linewidth(7)
        line = ax.add_collection(lc)
        # fig.colorbar(line, ax=ax)

        ax.scatter(x[0], y[0], s=50, color='g')
        ax.scatter(x[-1], y[-1], s=50, color='r')


        # cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(line, ax=ax)

        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        # ax.set_zlabel('Y [m] - Height')    
        ax.axis('equal')
        # set_axes_equal(ax)

        # plt.show()


class PlotKalman(PlotOdom):
    def __init__(self, csv_path) -> None:
        self.path = csv_path
        self.data = np.genfromtxt(csv_path, dtype=float, delimiter=',', names=True, skip_header=0)
        self.headers = self.data.dtype.names
        for header in self.headers:
            setattr(self, header, self.data[header])
        # self.x = self.data['x']
        self.positions = np.c_[self.x, self.y, self.z]

    def plot_state_euler(self):
        plt.figure()
        plt.plot( self.time, np.rad2deg(self.ex),label="Roll")
        plt.plot( self.time, np.rad2deg(self.ey),label="Pitch")
        plt.plot( self.time, np.rad2deg(self.ez),label="Yaw")
        plt.xlabel("Time [s]")
        plt.ylabel("Angle [°]")
        plt.legend()


if __name__ == "__main__":
    dir_data = os.path.realpath("data")
    data_set = "frontyard"

    Plotter = PlotOdom(name=data_set)
    # PlotterKalman = PlotKalman(r"/home/slamnuc/temp_saved_odometry_data/kalman/kalman_data.csv")
    # print(Plotter.data)
    # Plotter.plot_odometry()
    # Plotter.plot_odometry_2D()
    # Plotter.plot_fitness()
    Plotter.plot_fitness_boxplot()
    # Plotter.plot_translation_cov()


    # Plotter.plot_acc_bias()
    # Plotter.plot_ang_bias()
    # PlotterKalman.plot_acc_bias()
    # PlotterKalman.plot_ang_bias()


    # fig, axes = plt.subplots(1,1, constrained_layout=True)
    # Plotter.plot_velocity(axes)
    # Plotter.plot_observer_velocity(axes)
    # PlotterKalman.plot_velocity(axes)
    plt.show()