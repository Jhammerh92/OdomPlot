import numpy as np
from PlotClass import PlotOdom
from filterpy.common import Q_discrete_white_noise
import matplotlib.pyplot as plt



class KalmanHeight():
    def __init__(self) -> None:
        self.dt = 0.1
        self.X = np.array([[0.0,0.0,0.0]]) # height, clime rate , clime rate bias
        self.F = np.array([[1.0, self.dt, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0]])

class KalmanFilterHeight():
    def __init__(self, dimx , dimz, spatial_dim, process_std, dt, measure_variance, measured_variables=[0], input_var=None, create_Q=True):
        self.dt = dt
        self.t = 0
        self.process_std = process_std
        self.spatial_dim = spatial_dim
        self.dimx = dimx # number of state tracked, ie. pos, vel , acc -> 3, pos, vel, bias, -> 3, pos, vel, acc_bias, euler_angles, ang_vel_bias -> 7
        self.dimz = dimz #
        self.x = np.zeros((self.dimx,1)) # pos, vel, acc, (jerk) in 3D 
        self.x_log = []
        self.x_post = self.x
        if not isinstance( measured_variables, list):
            measured_variables = [v for v in [measured_variables]]
        self.measured_variables = measured_variables
        if not isinstance( measure_variance, list):
            measure_variance = [v for v in [measure_variance]]
        self.measure_variance = measure_variance

        self.F = np.array([[1.0, self.dt, -self.dt],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0]])

        # with np.printoptions(precision=4, suppress=True, formatter={'float': '{:0.5}'.format}, linewidth=200):
            # print(self.F)
        self.R =[np.eye(self.dimz) * var for var in self.measure_variance]

        # self.H = np.c_[np.eye(3), np.zeros((3,3)), np.eye(3), np.zeros((3,3))]
        self.H = []
        for index in self.measured_variables:
            h = np.zeros((self.dimz, self.dimx))
            h[:,spatial_dim*index : spatial_dim*index+self.dimz] = np.eye(self.dimz)
            self.H.append(h)
        # self.H[:3,6:9] =  np.eye(3)
        # self.H[3:, 9:] = np.eye(3)
        # print(self.H)

        # self.Q = np.eye(self.dimx) * 1.0

        self.P = np.eye(self.dimx) * self.process_std**2
        self.P_post = self.P
        # self.update_P()

        # self.K = np.tile(np.eye(self.dimz), (self.dimz,1)) * 0.0
        # self.update_K()
        # print(self.K)
        # self.K = np.zeros((self.dimx,self.dimz))

        if input_var is None:
            self.B = None
        else:
            d = np.zeros((self.dimx,1))
            d[input_var] = 1
            self.B = self.F @ d
        # self.H_u = self.H
        # self.K_u = None
        # self.R_u = np.eye(self.dimz)

        if create_Q:
            self.Q = Q_discrete_white_noise(int(self.dimx/self.spatial_dim), self.dt, var=self.process_std**2, block_size=self.spatial_dim, order_by_dim=False)


        self.z = None
        self.y = np.zeros((3,1))
        self.y_prior = self.y
        self.Y = np.zeros((3,1))

    def predict(self, t, input=None):
        # predict
        # self.x = self.x_post
        # self.P = self.P_post
        # if not (dt is None):
        #     self.update_F(dt)

        dt = t - self.t

        self.update_F(dt)

        self.update_state(input)
        self.update_P()
        
        self.t = t

        # if not (dt is None):
        #     self.update_F(self.dt)

    def set_init_t(self, t):
        self.t = t

    # def predict_with_measurement(self, u, dt = None):
    #     # predict
    #     # self.x = self.x_post
    #     # self.P = self.P_post
    #     if not (dt is None):
    #         self.update_F(dt)

    #     self.update_state_with_measurement(u)
    #     self.update_P()
        
    #     if not (dt is None):
    #         self.update_F(self.dt)




    def update(self, input=None, input_index=[], H=None, R=None):
        # if not (z is None):

        ret_input = False
        if isinstance(input_index, int):
           ret_input = True

        if not(H is None):
            self.temp_H = self.H
            self.H = H
        if not(R is None):
            # self.temp_R = self.R
            # self.R = R
            _R = R
        else:
            _R = self.R[input_index]

        self.z = input.reshape((self.dimz,1))
            
        # this is not happening?
        if ret_input:
            self.update_K(_R, self.H[input_index])
            self.post_state(self.H[input_index])
            self.post_P(_R, self.H[input_index])
        else:
            self.update_K(self.R, self.H)
            self.post_state(self.H)
            self.post_P(self.R, self.H)

    
        # if not (z is None):

        # else:
            # self.x = self.x_post
            # self.P = self.P_post


        # if not(H is None):
        #     self.H = self.temp_H
        # if not(R is None):
        #     self.R = self.temp_R



    def post_state(self, H):
        self.y_prior = self.y
        if self.z is None:
            self.z = np.zeros((self.dimz,1))
        
        self.y = (self.z - H @ self.x)
        self.x = self.x + self.K @ self.y
        # epsilon = self.R[0,0]
        # self.Y = self.y_prior**2/(self.y**2 + np.tile(epsilon,(1,3)))
        # self.Y = (self.y**2 + np.tile(epsilon,(1,3)))
        self.x_post = self.x

        self.z = None

    def post_P(self, R, H):
        # self.P = (np.eye(self.dimx) - self.K @ self.H ) @ self.P
        self.P = (np.eye(self.dimx) - self.K @ H ) @ self.P
        # self.P = (np.eye(self.dimx) - self.K @ H ) @ self.P @(np.eye(self.dimx) - self.K @ H ).T + self.K@R@self.K.T
        self.P_post = self.P
        # print(np.max(self.P_post, axis=0))

    def update_state(self, input=None):
        if input is None:
            self._update_state()
        else:
            self._update_state_with_input(input)

        self._log_state()

    def _update_state(self): # pure prediction
        self.x = self.F @ self.x
        return

    # def __update_state_with_input(self, u): #  prediction with measurement
    #     PHT = self.P @ self.H_u.T
    #     self.K_u = PHT @ np.linalg.inv( self.H_u @ PHT + self.R_u )
    #     y = (u.reshape((self.dimz, -1))  - self.H_u @ self.x)
    #     u = self.H_u @ (self.x + self.K_u @ y)
    #     self.x = self.F @ self.x + self.B @ u
    #     eye_u = np.diag(np.sum(self.H_u,0)>0)
    #     P = (np.eye(self.dimx) - eye_u@self.K_u @ self.H_u ) @ self.P
    #     self.P = P

    def _update_state_with_input(self, u): #  prediction with measurement
        # PHT = self.P @ self.H_u.T
        # self.K_u = PHT @ np.linalg.inv( self.H_u @ PHT + self.R_u )
        # y = (u.reshape((self.dimz, -1))  - self.H_u @ self.x)
        # u = self.H_u @ (self.x + self.K_u @ y)
        self.x = self.F @ self.x + self.B @ u
        # eye_u = np.diag([0,0,0,0,0,0,1,1,1])
        # P = (np.eye(self.dimx) - eye_u@self.K_u @ self.H_u ) @ self.P
        # self.P = P

    def update_F(self, dt=None):

        self.F = np.array([[1.0, dt, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0]])

        # if dt is None:
        #     dt = self.dt



        # self.F = np.zeros((self.dimx,self.dimx ))
        # for n in range(int(self.dimx/self.spatial_dim)):
        #     self.F += np.roll( np.eye(self.dimx) * 1/factorial(n)*dt**(n), (n)*self.spatial_dim, axis=1)
        #     # self.F = np.triu(np.eye(self.dimx) + np.roll(np.eye(self.dimx)*dt, self.dimz, axis=1) + np.roll(np.eye(self.dimx)*1/2*dt**2, self.dimz*2, axis=1))
        # self.F =' np.triu(self.F) # get the upper triangle matrix

    def update_P(self):
        # print(self.P)
        self.P = self.F @ self.P @ self.F.T + self.Q
        return



    def update_K(self, R, H):
        # self.PHT = self.P @ self.H.T
        PHT = self.P @ H.T
        self.K = PHT @ np.linalg.inv( H @ PHT + R )
        # self.K = self.PHT @ (np.linalg.inv( self.H @ self.PHT + self.R ) + np.eye(self.dimz)*self.Y)**0.5
        # print(np.max(self.K, axis=0))
        return

    def get_state(self):
        return self.x.copy().flatten()

    def get_kalman_gain(self):
        return np.max(self.H @ self.K, axis=0)

    def get_uncertainty(self):
        return np.sqrt(np.max(self.H @ self.P, axis=0))
    
    def _log_state(self,in_body_frame=False):
        x_log_copy = self.x.copy()
        # if in_body_frame:
        #     acc = x_log_copy[self.dimx_linear - 3* self.with_bias - 3:self.dimx_linear - 3* self.with_bias ]
        #     x_log_copy[self.dimx_linear - 3* self.with_bias - 3:self.dimx_linear - 3* self.with_bias ] = self.orientation_quat.inverse.rotate(acc).reshape(acc.shape)
        self.x_log.append(x_log_copy.flatten())

    def _print_matrix(self, matrix_name):
        with np.printoptions(precision=4, suppress=True, formatter={'float': '{:0.5}'.format}, linewidth=200): 
            print(matrix_name ,getattr(self, matrix_name), sep="\n")

    def print_matrices(self):
        for key in self.__dict__:
            if type(getattr(self, key)) == np.ndarray:
                self._print_matrix(key)

    def get_state_log(self):
        return np.asarray(self.x_log)



if __name__ == "__main__":

    """Processed data"""
    processed_lio_data = "20231212_mid70"


    """Ground truth data"""
    ground_truth_data = "./Python/ground_truth_data/ground_truth_lidar_car_test_01_20231212.txt"
    # ground_truth_data = "ground_truth_data/ground_truth_lidar_car_test_01_20231212.txt"


    odom_handler = PlotOdom(data_path="./data", name=processed_lio_data)
    odom_handler.load_ground_truth_poses_from_file(ground_truth_data)
    
    loam_height = odom_handler.get_positions()[:,2]
    loam_clime_rate = np.r_[0.0,np.diff(loam_height)]
    loam_time = odom_handler.get_time()

    loam_error_climb = np.ones(len(loam_time)) * 0.002
    loam_error_height = np.logspace(0.00001, 10.0, len(loam_time))**2

    loam_input_climb = np.c_[loam_time, loam_clime_rate, np.tile(1, len(loam_time)), loam_error_climb]
    loam_input_height = np.c_[loam_time, loam_height, np.tile(0, len(loam_time)), loam_error_height]

    gt_height = odom_handler.IEF.get_positions()[:,2]
    gt_height = gt_height - gt_height[0]
    gt_time = odom_handler.IEF.get_time()

    # gt_error = (0.5 + 2*gt_height)**2
    gt_error = np.ones(len(gt_time))*(20)**2

    gt_input = np.c_[gt_time, gt_height, np.tile(0, len(gt_time)), gt_error]

    kf_inputs = np.r_[loam_input_climb, loam_input_height, gt_input]
    kf_inputs = kf_inputs[np.argsort(kf_inputs[:,0])]



    KF = KalmanFilterHeight(3, 1, 1, 0.5, 0.1, [100.0, .0002], [0, 1] )
    KF.print_matrices()

    KF.set_init_t(kf_inputs[0,0])

    for input in kf_inputs:
        print(input)
        KF.predict(input[0])
        KF.update(input=input[1], input_index=int(input[2]) , R= np.array([[input[3]]]))




    kf_state_log = KF.get_state_log()




    fig, ax = plt.subplots()
    ax.plot(loam_time, loam_height)
    ax.plot(gt_time, gt_height)
    ax.plot(kf_inputs[:,0], kf_state_log[:,0], label='Kalman Height')

    fig, ax = plt.subplots()
    ax.plot(kf_inputs[:,0], kf_state_log[:,0], label='Kalman Height')
    ax.plot(kf_inputs[:,0], kf_state_log[:,1], label='Kalman Climb rate')
    ax.plot(kf_inputs[:,0], kf_state_log[:,2], label='Kalman Climb Rate Bias')

    fig.legend()

    plt.show()


