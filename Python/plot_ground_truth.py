import numpy as np
import matplotlib.pyplot as plt
from PlotClass import rotate_to_nearest_x_axis, rotate_to_x_axis


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

        

    def get_zeroed_positions(self):
        return self.positions - self.positions[0,:]

    def zero_initial_heading(self):
        # initial_xy_heading = -self.Heading[0]
        # # Create the 2D rotation matrix
        # derotation_matrix = np.array([[np.cos(initial_xy_heading), -np.sin(initial_xy_heading), 0],
        #                             [np.sin(initial_xy_heading), np.cos(initial_xy_heading),  0],
        #                             [0,             0,              1]])
    
        length = 0.0
        i = 0
        while length < 1.0:
            i += 1
            initial_xy_heading_vector = self.positions[i,:2] - self.positions[0,:2]
            length = np.linalg.norm(initial_xy_heading_vector)
    
        derotation_matrix = rotate_to_nearest_x_axis(initial_xy_heading_vector)

        self.positions = self.positions @ derotation_matrix





if __name__ == "__main__":
    handler = InertialExplorerFileHandler()

    handler.get_ground_truth_poses_from_file("/home/slamnuc/Desktop/OdomPlot/Python/Ground_truth_lidar_car_test_01_20231212.txt")

    # positions = handler.get_positions()

    fig, ax = plt.subplots()
    ax.plot(handler.XLL, handler.YLL)

    plt.show()