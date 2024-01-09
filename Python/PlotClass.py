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

def capfirst(s:str):
    return s[:1].upper() + s[1:]

def rotate_to_x_axis(vector):
    # Normalize the input vector
    vector = np.array(vector)
    magnitude = np.linalg.norm(vector)
    if magnitude == 0:
        raise ValueError("Input vector has zero magnitude.")
    normalized_vector = vector / magnitude
    x_vector = np.array([[1, 0]]).T

    angle = np.arctan2(x_vector[1], x_vector[0]) - np.arctan2(normalized_vector[1], normalized_vector[0]).item()
    angle = (angle + np.pi) % (2 * np.pi) - np.pi

    # Create the 2D rotation matrix
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                [np.sin(angle), np.cos(angle),  0],
                                [0,             0,              1]])
    
    return rotation_matrix

def rotate_to_y_axis(vector):
    # Normalize the input vector
    vector = np.array(vector)
    magnitude = np.linalg.norm(vector)
    if magnitude == 0:
        raise ValueError("Input vector has zero magnitude.")
    normalized_vector = vector / magnitude

    # Calculate the angle of rotation
    # angle = np.arctan2(normalized_vector[0], normalized_vector[1])
    # angle = np.arccos(normalized_vector.dot(np.array([[0, 1]]).T)).item()

    y_vector = np.array([[0, 1]]).T

    angle = - np.arctan2(normalized_vector[0], normalized_vector[1]).item()
    # angle = (angle + np.pi) % (2 * np.pi) - np.pi

    # Create the 2D rotation matrix
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                [np.sin(angle), np.cos(angle),  0],
                                [0,             0,              1]])
    
    return rotation_matrix


def rotate_to_nearest_x_axis(vector):
    # Normalize the input vector
    vector = np.array(vector)
    magnitude = np.linalg.norm(vector)
    if magnitude == 0:
        raise ValueError("Input vector has zero magnitude.")
    normalized_vector = vector / magnitude

    # Calculate the angle of rotation to the nearest x-axis
    angle = np.arccos(normalized_vector.dot(np.array([[1, 0]]).T)).item()
    if angle > np.pi/2.0 : angle = angle - np.pi
    # angle_negative = -angle_positive if np.arcsin(normalized_vector[0].item()) >= 0 else angle_positive

    # Choose the nearest x-axis direction
    # angle = angle_negative if abs(angle_negative) < abs(angle_positive) else angle_positive

    # Create the 2D rotation matrix
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                [np.sin(angle), np.cos(angle),  0],
                                [0,             0,              1]])
    
    return rotation_matrix



class PlotOdom:
    def __init__(self, data_path:os.PathLike=None , name:str="", save_plots:bool=False) -> None:
        if data_path is None:
            if name is None or name == "":
                abs_csv_path = os.path.join("data","run_data.csv")
                name = "recent"
        else:
            abs_csv_path = os.path.join(data_path, name + "_run_data.csv")

        self.name = capfirst(name.replace('_', ' '))
        self.plot_name = name



        assert os.path.isfile(abs_csv_path), f"path does not exist: {abs_csv_path}"

        self.path = abs_csv_path
        self.data = np.genfromtxt(abs_csv_path, dtype=float, delimiter=',', names=True, skip_header=31)

        self.plot_dir = ""
        self.save_plots = save_plots
        self.all_opened_figs = []

        self.headers = self.data.dtype.names
        for header in self.headers:
            setattr(self, header, self.data[header])

        self.start_time = self.time[0]
        self.end_time = self.time[-1]


        # self.x = self.data['x']
        self.positions = np.c_[self.x, self.y, self.z]

        self.residual_norm = np.sqrt(np.square( self.residual_x) + np.square(self.residual_y) + np.square(self.residual_z))

        self.translation = np.sqrt(np.square(self.vx) + np.square(self.vy) + np.square(self.vz)) * 0.1

        self.residual_norm_normalised = self.residual_norm/self.translation
        # self.residual_qnorm = np.sqrt(np.square(1- np.abs(self.residual_qw)) + np.square( self.residual_qx) + np.square(self.residual_qy) + np.square(self.residual_qz))
        self.residual_qnorm = (np.sqrt( np.square( self.residual_qx) + np.square(self.residual_qy) + np.square(self.residual_qz)))
        
        self.fig = None
        self.axes= None

        plt.rcParams['figure.figsize'] = (9, 5)
        plt.rcParams['figure.constrained_layout.use'] = True
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.grid'] = True
        plt.rcParams['lines.linewidth'] = 1.0
        plt.rcParams['lines.color'] = "black"
        plt.rcParams['figure.dpi'] = 150

        # print(plt.rcParams)

    def get_positions(self):
        return self.positions

    def zero_initial_heading(self, heading_length=2.0):
        length = 0.0
        i = 0
        while length < heading_length:
            i += 1
            initial_xy_heading_vector = self.positions[i,:2] - self.positions[0,:2]
            length = np.linalg.norm(initial_xy_heading_vector)
    
        derotation_matrix = rotate_to_y_axis(initial_xy_heading_vector)

        self.rotate_positions(derotation_matrix)

    def rotate_to_heading(self, gt_initial_heading):
        self.zero_initial_heading()

        rotation_matrix = np.array([[np.cos(gt_initial_heading), -np.sin(gt_initial_heading), 0],
                                [np.sin(gt_initial_heading), np.cos(gt_initial_heading),  0],
                                [0,             0,              1]])
        
        self.rotate_positions(rotation_matrix)
    
    def rotate_positions(self, rotation_matrix, angle=None):
        self.positions = self.positions @ rotation_matrix 
        #TODO add x y and z is rewritten from the new positions

    def get_start_time(self):
        return self.start_time
    
    def get_end_time(self):
        return self.end_time

    def plot_odometry(self):
        plt.figure()
        plt.plot(self.x, self.y)
        # plt.plot(self.x, self.fitness)
        # plt.plot( self.x - self.residual_x, self.y - self.residual_y)

    def plot_velocity(self, ax=None, label="_fill"):
        self.label = label
        return self._general_plot_handle(self._plot_velocity, timewise=True, ax=ax)

    def _plot_velocity(self, axes=None):
        # if axes is None:
        #     fig, axes = plt.subplots(1,1, constrained_layout=True, num=" velocity")
        # axes.legend()
        num = "LOAM velocity"
        if not isinstance(self.axes, np.ndarray):
            self.fig, self.axes = plt.subplots(3,1, num=num, sharex=True)
        elif len(self.axes) != 3:
            self.fig, self.axes = plt.subplots(3,1, num=num, sharex=True)
        ylim = [np.inf,-np.inf ]
        for ax, d, a in zip(self.axes, [self.vx, self.vy, self.vz], ["x", "y", "z"]):
            ax.plot( self.time, d, label=self.label)
            ax.set_ylabel("$V$ $[m/s]$ - " + a)
            _ylim = ax.get_ylim()
            if _ylim[0] < ylim[0]:
                ylim[0] = _ylim[0]
            if _ylim[1] > ylim[1]:
                ylim[1] = _ylim[1]
        for ax in self.axes:
            ax.set_ylim(ylim)

        return self.axes


    def plot_observer_velocity(self, ax=None, label="_fill"):
        self.label = label
        return self._general_plot_handle(self._plot_observer_velocity, timewise=True, ax=ax)

    def _plot_observer_velocity(self, axes=None):
        num = "Observer velocity"
        if not isinstance(self.axes, np.ndarray):
            self.fig, self.axes = plt.subplots(3,1, num=num, sharex=True)
        elif len(self.axes) != 3:
            self.fig, self.axes = plt.subplots(3,1, num=num, sharex=True)

        for ax, d, a in zip(self.axes, [self.obs_vx, self.obs_vy, self.obs_vz], ["x", "y", "z"]):
            ax.plot( self.time, d, label=self.label)
            ax.set_ylabel("$V$ $[m/s]$ - " + a)

        return self.axes


    def plot_translation_cov(self, ax=None):
        return self._general_plot_handle(self._plot_translation_cov, timewise=True,ax=ax)

    def _plot_translation_cov(self):
        self.fig, self.axes = plt.subplots(num="Covariance")
        self.axes.plot( self.time, self.cov_x,label="x")
        self.axes.plot( self.time, self.cov_y,label="y")
        self.axes.plot( self.time, self.cov_z,label="z")
        self.axes.set_title("Translation Covariance")
        self.axes.set_ylabel("$Covariance$ $[m^2]$")
        self.axes.legend()

        self.all_opened_figs.append(self.fig)

    def _plot_acc_bias(self):
        num = "Acc bias"
        if not isinstance(self.axes, np.ndarray):
            self.fig, self.axes = plt.subplots(3,1, num=num,sharex=True)
        elif len(self.axes) != 3:
            self.fig, self.axes = plt.subplots(3,1, num=num,sharex=True)
        ylim = [np.inf,-np.inf ]
        for ax, d, a in zip(self.axes, [self.bias_acc_x, self.bias_acc_y, self.bias_acc_z], ["x", "y", "z"]):
            ax.plot( self.time, d, label="_")
            ax.set_ylabel("$b_a$ $[m/s²]$ - " + a)
            _ylim = ax.get_ylim()
            if _ylim[0] < ylim[0]:
                ylim[0] = _ylim[0]
            if _ylim[1] > ylim[1]:
                ylim[1] = _ylim[1]
        for ax in self.axes:
            ax.set_ylim(ylim)


        return self.axes

    def plot_acc_bias(self, ax=None):
        return self._general_plot_handle(self._plot_acc_bias, timewise=True,ax=ax)


    def _plot_ang_bias(self):
        num = "Gyro bias"
        if not isinstance(self.axes, np.ndarray):
            self.fig, self.axes = plt.subplots(3,1, num=num,sharex=True)
        elif len(self.axes) != 3:
            self.fig, self.axes = plt.subplots(3,1, num=num,sharex=True)

        ylim = [np.inf,-np.inf ]
        for ax, d, a in zip(self.axes, [self.bias_ang_x, self.bias_ang_y, self.bias_ang_z], ["x - Roll", "y - Pitch", "z - Yaw"]):
            ax.plot( self.time, d, label="_")
            ax.set_ylabel("$b_g$ $[rad/s]$ - " + a)
            _ylim = ax.get_ylim()
            if _ylim[0] < ylim[0]:
                ylim[0] = _ylim[0]
            if _ylim[1] > ylim[1]:
                ylim[1] = _ylim[1]
        for ax in self.axes:
            ax.set_ylim(ylim)

        return self.axes
        

    def plot_ang_bias(self, ax=None):
        return self._general_plot_handle( self._plot_ang_bias, timewise=True, ax=ax)

    def plot_fitness(self, ax=None):
        return self._general_plot_handle(self._plot_fitness, timewise=True,ax=ax)

    def _plot_fitness(self):
        n = 7
        alpha = 1
        if self.axes is None:
            self.fig, self.axes = plt.subplots(num="Fitness Metrics")
        # self.axes.plot(self.time,  self.residual_norm*1e3, label="Linear Residual Norm", marker='.', ls='-')
        self.axes.plot(self.time,  self.residual_norm_normalised,alpha=alpha, label="Linear Residual Norm Normalised", marker='.',markersize=5, ls='-', lw= 1)
        self.axes.plot(self.time,  np.rad2deg(self.residual_qnorm),alpha=alpha, label="Rotational Residual Norm"                , marker='.',markersize=5, ls='-', lw= 1)
        self.axes.plot(self.time,  self.fitness, alpha=alpha, label="Scan Match Fitness"                                        , marker='.',markersize=5, ls='-', lw= 1)
        self.axes.set_prop_cycle(None)
        # self.axes.plot(self.time[:-(n-1)],  moving_average(self.residual_norm*1e3 , n), label="Linear Residual Norm", lw = 2.0)
        # self.axes.plot(self.time[:-(n-1)],  moving_average(self.residual_qnorm *1e2, n), label="Rotational Residual Norm")
        # self.axes.plot(self.time[:-(n-1)],  moving_average(self.fitness*1e3, n), label="Scan Match Fitness")
        # self.axes.set_xlabel("Time [s]")
        self.axes.set_yscale('log')
        self.axes.set_ylabel("Residual/Fitness [m/m] / [°]/ [m]")
        self.axes.set_title("Timewise Log Fitness Metrics")


    def plot_fitness_boxplot(self, ax=None):
        return self._general_plot_handle(self._plot_fitness_boxplot, timewise=False, ax=ax)
        

    def custom_boxplot(self, ax:plt.Axes, data, index=0, c='C0'):
        data_list = [data if i==index else np.array([np.nan]) for i in range(index+1)]
        if ax is None:
            self.fig, self.axes = plt.subplots(num="Fitness Boxplots")
        self.axes.scatter(np.random.normal(index+1, 0.1, len(data)), data, marker='.', s=2.5, alpha=0.5, c=f"C{index}")
        self.axes.boxplot(data_list, showfliers = False, widths=(0.9), vert=True)
        return self.axes

    def _plot_fitness_boxplot(self):
        data = [self.residual_norm_normalised, np.rad2deg(self.residual_qnorm), self.fitness*1e1]

        for i, d in enumerate(data):
            self.custom_boxplot(self.axes, d, i)

        self.axes.set_xticks([1,2,3],['$||\Delta t||/t$ [m/m]', '$||q||$ [°]', 'Fitness [dm]'])
        self.axes.set_ylim([0, 2])
        # self.axes.set_xlim([0, 100])
        self.axes.grid(False)
        self.fig.set_figwidth(3)
        self.axes.set_ylabel("Residual/Fitness  [m/m] / [°] / [dm]")
        self.axes.set_title("Fitness Metrics")

    def _plot_timewise(self, plot_call):
        plot_call()
        try:
            self.axes.set_xlabel("Time [s]")
        except: 
            self.axes[-1].set_xlabel("Time [s]")

    # def _general_plot_handle(self, fig:plt.Figure=None, ax:plt.Axes, plot_call, timewise=True):
    def _general_plot_handle(self, plot_call, timewise=True, ax:plt.Axes=None):
        # if ax is None:
        #     self.fig, self.axes = plt.subplots()
            
        # else:
            # if not fig.axes:
        self.axes = ax
            # self.fig = fig

        if timewise:
            self._plot_timewise( plot_call)
        else:
            plot_call()

        if isinstance(self.axes, np.ndarray):
            for ax in self.axes:
                ax.legend()
                # ax.set_title(self.name)
        else:
            self.axes.get_title()
            self.axes.set_title(f"{self.axes.get_title()} - {self.name}")
            self.axes.legend()

        self.all_opened_figs.append(self.fig)

        return self.axes
    
    def save_figs(self, other_path=None):
        if not self.save_plots:
            return
        
        if other_path is None:
            self.plot_dir = os.path.join("plots", self.name)
        else:
            self.plot_dir = os.path.join(other_path, self.name)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        for fig_ in self.all_opened_figs:
            fig_.savefig(os.path.join(self.plot_dir, "{}.pdf".format(fig_.get_label() )) )
            # fig_.savefig(os.path.join(dir, "{}_{}.pdf".format(situs[situ], fig_.get_label() )) )

        print(f"Saved all figs of data set: '{self.name}'\nSaved to {self.plot_dir}")



    
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

        layout = (2,1)
        # if (max(x)-min(x)) < (max(y)-min(y)):
            # layout = (1,2)
        
        if axes is None:
            fig, axes = plt.subplots(*layout, constrained_layout=True, num=f"Odometry", sharex=False, sharey=False)

        axes[0].scatter(x[0], y[0], s=50, color='g', marker='x') # start
        axes[0].scatter(x[-1], y[-1], s=50, color='r', marker='x') # end
        axes[0].plot(x, y, label=label, **kwargs)
        # axes[0].set_aspect('equal', adjustable='box')
        axes[0].set_aspect('equal')
        # axes[0].axis('equal')
        # axes[0].set_xlabel('X [m]')
        axes[0].set_ylabel('Y [m]')
        axes[0].legend()

        ylim = axes[0].get_ylim()
        ydiff = ylim[1] - ylim[0]
        p = 0.25
        axes[0].set_ylim([ylim[0]- ydiff*p, ylim[1]+ydiff*p])

        axes[1].scatter(x[0], z[0], s=50, color='g', marker='x') # start
        axes[1].scatter(x[-1], z[-1], s=50, color='r', marker='x') # end
        axes[1].plot(x, z, **kwargs)
        # axes[1].axis('equal')
        axes[1].set_aspect('equal', adjustable='box')
        # axes[1].set_aspect('equal')
        axes[1].set_xlabel('X [m]')
        axes[1].set_ylabel('Z [m]')

        ylim = axes[1].get_ylim()
        ydiff = ylim[1] - ylim[0]
        p = 4
        axes[1].set_ylim([ylim[0]- ydiff*p, ylim[1]+ydiff*p])


        axes[0].set_title('XY-plane')
        axes[1].set_title('XZ-plane, elevation')

        self.all_opened_figs.append(fig)

        # if (fig is None):
        #     return axes
        # else:
        return fig, axes


    # def plot_odometry_2D_timewise(self, axes=None, arg='', label='', **kwargs):
    #     self._general_plot_handle(self._plot_odometry_2D_timewise, False, ax=axes, )

    def plot_odometry_2D_timewise(self, axes=None, arg='', label='', **kwargs):
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

        # fig = None
        if axes is None:
            fig, axes = plt.subplots(3,1, constrained_layout=True, num=f"Timewise Odometry", sharex=True)

        axes[0].plot(self.time, x, label=label, **kwargs)
        # axes[0].axis('equal')
        # axes[0].set_xlabel('Time [s]')
        axes[0].set_ylabel('X [m]')

        axes[1].plot(self.time, y, label=label, **kwargs)
        # axes[1].axis('equal')
        # axes[1].set_xlabel('Time [s]')
        axes[1].set_ylabel('Y [m]')

        axes[2].plot(self.time, z, label=label, **kwargs)
        # axes[2].axis('equal')
        axes[2].set_xlabel('Time [s]')
        axes[2].set_ylabel('Z [m]')

        # axes[0].set_title('X')
        # axes[1].set_title('Y')
        # axes[2].set_title('Z, elevation')

        self.all_opened_figs.append(fig)

        # if (fig is None):
        #     return axes
        # else:
        return fig, axes
        


    def plot_odometry_colorbar_2D(self, c_values, ax:plt.Axes=None, arg='', cmap='plasma', plane='xy',lw=3, cbar_label='', cbar_unit=''):
        # 3d plot of egomotion path - OBS. Z and Y axis are switched, but labelled correctly in plot
        c = self.positions[:len(c_values), :]
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
        lc.set_linewidth(lw)
        line = ax.add_collection(lc)
        # fig.colorbar(line, ax=ax)

        ax.scatter(x[0], y[0], s=50, color='g')
        ax.scatter(x[-1], y[-1], s=50, color='r')


        # cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        cbar = self.fig.colorbar(line, ax=ax,aspect=10)
        cbar.set_label(cbar_unit, rotation=270, labelpad=15)
        # cbar.ax.yaxis.set_label_position('right')
        cbar.ax.tick_params(direction="out",labelsize=8) 
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.set_title(cbar_label, fontsize=8)


        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        # ax.set_zlabel('Y [m] - Height')    
        # ax.axis('equal')
        # set_axes_equal(ax)

        # plt.show()
        return self.fig, ax


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
    data_set_1 = "lobby_01"
    data_set_2 = "lobby_01_NDT"

    Plotter1 = PlotOdom(name=data_set_1)
    Plotter2 = PlotOdom(name=data_set_2)
    # PlotterKalman = PlotKalman(r"/home/slamnuc/temp_saved_odometry_data/kalman/kalman_data.csv")
    # print(Plotter.data)
    # Plotter.plot_odometry()
    Plotter1.plot_odometry_2D()
    # Plotter.plot_fitness()
    # Plotter.plot_fitness_boxplot()
    # Plotter.plot_translation_cov()

    # fig, axes = plt.subplots(1,2, num="boxplots", sharey=True)
    # Plotter1.plot_fitness_boxplot(axes[0])
    # Plotter2.plot_fitness_boxplot(axes[1])



    # Plotter.plot_acc_bias()
    # Plotter.plot_ang_bias()
    # PlotterKalman.plot_acc_bias()
    # PlotterKalman.plot_ang_bias()

    # fig, axes = plt.subplots(1,1, constrained_layout=True)
    # Plotter.plot_velocity(axes)
    # Plotter.plot_observer_velocity(axes)
    # PlotterKalman.plot_velocity(axes)
    plt.show()