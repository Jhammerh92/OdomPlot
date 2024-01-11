import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from PlotClass import rotate_to_nearest_x_axis, rotate_to_x_axis

def plot_confidence_ellipses(ax, points, stds):
    """
    Draw ellipses with varying sizes for a set of points.

    Parameters:
    - points (array-like): Array of points in the form [[x1, y1], [x2, y2], ...].
    - sizes (array-like): Array of sizes for each ellipse.

    Returns:
    - None (displays the plot).
    """
    # fig, ax = plt.subplots()
    scale_factor = 2.0

    for i, (point, size) in enumerate(zip(points, stds)):
        ellipse = Ellipse(xy=point, width=size[0]*scale_factor, height=size[1]*scale_factor, edgecolor='none', facecolor='b', alpha=0.1)
        # ellipse = Ellipse(xy=point, width=scale_factor, height=scale_factor, edgecolor='none', facecolor='b', alpha=0.3)
        ax.add_patch(ellipse)

        # # Annotate each ellipse with its index
        # ax.annotate(str(i + 1), xy=point, color='r', ha='center', va='center', fontsize=8)


class InertialExplorerFileHandler():
    def __init__(self) -> None:
        pass
        

    def get_ground_truth_poses_from_file(self, filename, start_time=None, end_time=None):

        self.data = np.genfromtxt(filename, names=True, skip_header=23, skip_footer=4, dtype=np.float64)
        print(self.data.dtype.names)
        self.headers = self.data.dtype.names

        if not (start_time == None):
            start_time_idx = np.argmin( np.abs(self.data['UTCTime'][1:] - start_time))
        else: start_time_idx = 0
        if not (end_time == None):
            end_time_idx = np.argmin( np.abs( self.data['UTCTime'][1:] - end_time))
        else: end_time_idx = -1


        for header in self.headers:
            setattr(self, header, self.data[header][start_time_idx:end_time_idx])

        self.positions = np.c_[self.XLL, self.YLL, self.ZLL]
        self.positions -= self.positions[0,:]
        
        self.stds = np.c_[self.SDEast, self.SDNorth, self.SDHeight]

        diff = np.diff(self.positions, axis=0)
        self.travelled_dist = np.r_[0.0, np.cumsum(np.linalg.norm(diff, axis=1))]

        

    def get_zeroed_positions(self):
        return self.positions - self.positions[0,:]
    
    def get_travelled_dist(self):
        return self.travelled_dist
    
    def get_stds(self):
        return self.stds

    def zero_initial_heading(self, heading_length=1.0):
        # initial_xy_heading = -self.Heading[0]
        # # Create the 2D rotation matrix
        # derotation_matrix = np.array([[np.cos(initial_xy_heading), -np.sin(initial_xy_heading), 0],
        #                             [np.sin(initial_xy_heading), np.cos(initial_xy_heading),  0],
        #                             [0,             0,              1]])
    
        length = 0.0
        i = 0
        while length < heading_length:
            i += 1
            initial_xy_heading_vector = self.positions[i,:2] - self.positions[0,:2]
            length = np.linalg.norm(initial_xy_heading_vector)
    
        derotation_matrix = rotate_to_x_axis(initial_xy_heading_vector)

        self.positions = self.positions @ derotation_matrix

    def get_initial_heading(self, heading_at_length=2.0):
        length = 0.0
        i = 0
        while length < heading_at_length:
            i += 1
            initial_xy_heading_vector = self.positions[i,:2] - self.positions[0,:2]
            length = np.linalg.norm(initial_xy_heading_vector)
        return self.Heading[i]



if __name__ == "__main__":
    handler = InertialExplorerFileHandler()

    handler.get_ground_truth_poses_from_file("/home/slamnuc/Desktop/OdomPlot/Python/Ground_truth_lidar_car_test_01_20231212.txt")

    # positions = handler.get_positions()

    fig, ax = plt.subplots()
    ax.plot(handler.XLL, handler.YLL)
    ax.set_aspect('equal')

    plt.show()