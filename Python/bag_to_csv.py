# from bagpy import bagreader


# b = bagreader('home/bagfiles/lidar_imu_12122022_190120/lidar_imu_12122022_190120.db3')
# bmesg = b.message_by_topic('/odom')

from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
import numpy as np
import matplotlib.pyplot as plt
import os.path

lidar_timestamps = []
imu_timestamps = []
imu_time_ref = []

bagfiles_folder ='/home/slamnuc/bagfiles/'
# recording = 'lidar_imu_12122022_190120' # with all topic recorded
recording = 'lidar_imu_14122022_161254' # with only essential topics recorded

# create reader instance and open for reading
with Reader(os.path.join(bagfiles_folder, recording) ) as reader:
    # topic and msgtype information is available on .connections list
    for connection in reader.connections:
        print(connection.topic, connection.msgtype)

    # iterate over messages
    for connection, timestamp, rawdata in reader.messages():
        if connection.topic == '/livox/lidar':
            msg = deserialize_cdr(rawdata, connection.msgtype)
            stamp = msg.header.stamp
            lidar_timestamps.append(stamp.sec + stamp.nanosec/1e9)
            # print(stamp.sec, stamp.nanosec)
        if connection.topic == '/imu/data_raw' or connection.topic == '/imu/data':
            msg = deserialize_cdr(rawdata, connection.msgtype)
            stamp = msg.header.stamp
            imu_timestamps.append(stamp.sec + stamp.nanosec/1e9)
        if connection.topic == '/imu/time_ref':
            msg = deserialize_cdr(rawdata, connection.msgtype)
            # print(msg)
            stamp = msg.time_ref
            imu_time_ref.append(stamp.sec + stamp.nanosec/1e9)
            
            # print(stamp.sec, stamp.nanosec)

lidar_timestamps = np.array(lidar_timestamps)
imu_timestamps = np.array(imu_timestamps)
imu_time_ref = np.array(imu_time_ref)

lidar_range = np.linspace(0,1,len(lidar_timestamps))
imu_range_stamp = np.linspace(0,1,len(imu_timestamps))
imu_range_ref = np.linspace(0,1,len(imu_time_ref))

plt.figure()
plt.scatter(lidar_range, lidar_timestamps - lidar_timestamps[0], s=0.8, alpha=1, label="Lidar")
plt.scatter(imu_range_stamp, imu_timestamps - imu_timestamps[0], s=0.8, alpha=0.8, label="IMU")
plt.scatter(imu_range_ref, imu_time_ref - imu_time_ref[0], s=0.8, alpha=0.8, label="IMU time_ref")
plt.ylabel("Relateive timestamp [s]")
plt.xlabel("Relative recording length [0,1]")
plt.title('Lidar to IMU timestamp comparison')
plt.legend()

plt.figure()
plt.scatter(lidar_range, lidar_timestamps, s=0.8, alpha=1, label="Lidar")
plt.scatter(imu_range_stamp, imu_timestamps, s=0.8, alpha=0.8, label="IMU")
# plt.scatter(imu_range_ref, imu_time_ref, s=0.5, alpha=0.5, label="IMU time_ref")
plt.ylabel("Absolute timestamp [UTC]")
plt.xlabel("Relative recording length [0,1]")
plt.title('Lidar to IMU timestamp comparison')
plt.legend()

# TODO plot difference between timestamp, user interpolater to get pseudo-timestamp in the intervals..

plt.figure()
plt.scatter(lidar_range[:-1], np.diff(lidar_timestamps),s=0.8, alpha=1, label="Lidar")
# plt.scatter(imu_range[:-1], np.diff(imu_timestamps),s=0.2, alpha=0.5, label="IMU")
# plt.hlines(1/400, 0,1, ls='--', lw=1, alpha = 0.5)
plt.ylabel("$d/dt$ [s]")
plt.xlabel("Relative recording length [0,1]")
plt.title('Lidar to IMU delta t comparison')
plt.legend()

plt.figure()
# plt.scatter(lidar_range[:-1], np.diff(lidar_timestamps),s=0.2, alpha=1, label="Lidar")
plt.scatter(imu_range_stamp[:-1], np.diff(imu_timestamps),s=0.2, alpha=0.8, label="IMU")
plt.scatter(imu_range_ref[:-1], np.diff(imu_time_ref),s=0.2, alpha=0.8, label="IMU time_ref")
plt.hlines(1/100, 0,1, ls='--', lw=1, alpha = 0.8)
plt.ylabel("$d/dt$ [s]")
plt.xlabel("Relative recording length [0,1]")
plt.title('Lidar to IMU delta t comparison')
plt.legend()


plt.show()

