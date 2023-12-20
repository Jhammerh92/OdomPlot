import numpy as np


class InertialExplorerFileHandler():
    def __init__(self) -> None:
        pass
        

    def get_ground_truth_poses_from_file(self, filename, heading_correction=0.0):

        self.data = np.genfromtxt(filename, names=True, skip_header=23, skip_footer=4, dtype=np.float64)
        print(self.data.dtype.names)
        self.headers = self.data.dtype.names
        for header in self.headers:
            setattr(self, header, self.data[header][:1])



if __name__ == "__main__":
    handler = InertialExplorerFileHandler()

    handler.get_ground_truth_poses_from_file("/home/slamnuc/Desktop/OdomPlot/Python/Ground_truth_lidar_car_test_01_20231212.txt")

    handler.data