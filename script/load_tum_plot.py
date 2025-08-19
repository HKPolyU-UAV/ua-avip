import numpy as np
import tf
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

def read_tum_file(file_path):
    timestamps = []
    positions = []
    orientations = []
    with open(file_path, 'r') as file:
        for line in file:
            data = line.strip().split()
            timestamps.append(float(data[0]))
            positions.append([float(data[1]), float(data[2]), float(data[3])])
            orientations.append([float(data[4]), float(data[5]), float(data[6]), float(data[7])])
    return np.array(timestamps), np.array(positions), np.array(orientations)

# File paths
fgo_file = '/home/xyb/ua_avip/fgo_02_adap_2.tum'
ground_truth_file = '/home/xyb/ua_avip/ground_truth_02_one_4.tum'

# Read the TUM files
fgo_timestamps, fgo_positions, fgo_orientations = read_tum_file(fgo_file)
ground_truth_timestamps, ground_truth_positions, ground_truth_orientations = read_tum_file(ground_truth_file)

# Comparison of other methods #
svin2_file = 'svin_2025_01_06_14_26_29.tum'
vins_file = 'vins_vio_tum.tum'
avip_file = 'avip.tum'

svin2_timestamps, svin2_positions, svin2_orientations = read_tum_file(svin2_file)
vins_timestamps, vins_positions, vins_orientations = read_tum_file(vins_file)
avip_timestamps, avip_positions, avip_orientations = read_tum_file(avip_file)

# Convert RGB to normalized values
color = [
    (233/255, 196/255, 107/255),
    (243/255, 162/255, 97/255),
    (230/255, 111/255, 81/255),
    (38/255, 70/255, 83/255),
    (42/255, 157/255, 142/255)
]

# Rotate the trajectories by 180 degrees around the X-axis
ground_truth_positions[:, 1] = -ground_truth_positions[:, 1]
vins_positions[:, 1] = -vins_positions[:, 1]
svin2_positions[:, 1] = -svin2_positions[:, 1]
avip_positions[:, 1] = -avip_positions[:, 1]
fgo_positions[:, 1] = -fgo_positions[:, 1]
avip_positions[:, 1] = -avip_positions[:, 1]

# Plot the trajectories (XY)
plt.figure(figsize=(10, 15))
plt.plot(ground_truth_positions[:, 0], ground_truth_positions[:, 1], label='Ground Truth', color='grey', linestyle='--')
plt.plot(vins_positions[:, 0], vins_positions[:, 1], label='VINS', color=color[4])
plt.plot(svin2_positions[:, 0], svin2_positions[:, 1], label='SVIN2', color=color[1])
plt.plot(avip_positions[:, 0], avip_positions[:, 1], label='AVIP', color=color[0])
plt.plot(fgo_positions[:, 0], fgo_positions[:, 1], label='UA-AVIP', color=color[2])

# Start and end points for each trajectory corresponding to their colors
plt.scatter(ground_truth_positions[0, 0], ground_truth_positions[0, 1], color='grey', marker= 'o', label='Start')
plt.scatter(ground_truth_positions[-1, 0], ground_truth_positions[-1, 1], color='black', marker = 's', label='End')
plt.scatter(fgo_positions[0, 0], fgo_positions[0, 1], color=color[2], marker= 'o')
plt.scatter(fgo_positions[-1, 0], fgo_positions[-1, 1], color=color[2], marker = 's')
plt.scatter(avip_positions[0, 0], avip_positions[0, 1], color=color[0], marker= 'o')
plt.scatter(avip_positions[-1, 0], avip_positions[-1, 1], color=color[0], marker = 's')
plt.scatter(svin2_positions[0, 0], svin2_positions[0, 1], color=color[1], marker= 'o')
plt.scatter(svin2_positions[-1, 0], svin2_positions[-1, 1], color=color[1], marker = 's')
plt.scatter(vins_positions[0, 0], vins_positions[0, 1], color=color[4], marker= 'o')
plt.scatter(vins_positions[-1, 0], vins_positions[-1, 1], color=color[4], marker = 's')
# Set equal spacing for X and Y axis grids
x_ticks = np.arange(np.floor(min(ground_truth_positions[:, 0])), np.ceil(max(ground_truth_positions[:, 0])), 0.5)
y_ticks = np.arange(np.floor(min(ground_truth_positions[:, 1])), np.ceil(max(ground_truth_positions[:, 1])), 0.5)
plt.xticks(x_ticks)
plt.yticks(y_ticks)
plt.xlim(-1.5, 2.3)
plt.ylim(-0.5, 2)
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize=14)
plt.grid()
plt.show()

# Plot the XYZ positions

fig, axs = plt.subplots(3, 1, figsize=(10, 15))
axs[0].plot(ground_truth_timestamps, ground_truth_positions[:, 0], label='Ground Truth', color='grey', linestyle='--')
axs[0].plot(vins_timestamps, vins_positions[:, 0], label='VINS', color=color[4])
axs[0].plot(svin2_timestamps, svin2_positions[:, 0], label='SVIN2', color=color[1])
axs[0].plot(avip_timestamps, avip_positions[:, 0], label='AVIP', color=color[0])
axs[0].plot(fgo_timestamps, fgo_positions[:, 0], label='UA-AVIP', color=color[2])
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('X (m)')
axs[0].grid()

axs[1].plot(ground_truth_timestamps, ground_truth_positions[:, 1], label='Ground Truth', color='grey', linestyle='--')
axs[1].plot(vins_timestamps, vins_positions[:, 1], label='VINS', color=color[4])
axs[1].plot(svin2_timestamps, svin2_positions[:, 1], label='SVIN2', color=color[1])
axs[1].plot(avip_timestamps, avip_positions[:, 1], label='AVIP', color=color[0])
axs[1].plot(fgo_timestamps, fgo_positions[:, 1], label='UA-AVIP', color=color[2])


axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Y (m)')
axs[1].grid()

axs[2].plot(ground_truth_timestamps, ground_truth_positions[:, 2], label='Ground Truth', color='grey', linestyle='--')
axs[2].plot(vins_timestamps, vins_positions[:, 2], label='VINS', color=color[4])
axs[2].plot(svin2_timestamps, svin2_positions[:, 2], label='SVIN2', color=color[1])
axs[2].plot(avip_timestamps, avip_positions[:, 2], label='AVIP', color=color[0])
axs[2].plot(fgo_timestamps, fgo_positions[:, 2], label='UA-AVIP', color=color[2])
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('Z (m)')
axs[2].grid()

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 3.67), ncol=7, fontsize=14)

plt.show()

# Plot the RPY angles

def quaternion_to_euler(quaternions):
    eulers = []
    for q in quaternions:
        euler = np.zeros(3)
        euler[0] = np.arctan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1]**2 + q[2]**2))
        euler[1] = np.arcsin(2 * (q[0] * q[2] - q[3] * q[1]))
        euler[2] = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2]**2 + q[3]**2))
        eulers.append(euler)
    return np.array(eulers)

ground_truth_eulers = np.array([tf.transformations.euler_from_quaternion(orientation) for orientation in ground_truth_orientations])
fgo_eulers = np.array([tf.transformations.euler_from_quaternion(orientation) for orientation in fgo_orientations])
avip_eulers = np.array([tf.transformations.euler_from_quaternion(orientation) for orientation in avip_orientations])
svin2_eulers = np.array([tf.transformations.euler_from_quaternion(orientation) for orientation in svin2_orientations])
vins_eulers = np.array([tf.transformations.euler_from_quaternion(orientation) for orientation in vins_orientations])
avip_eulers = np.array([tf.transformations.euler_from_quaternion(orientation) for orientation in avip_orientations])

# radian to degree
ground_truth_eulers = ground_truth_eulers * 180 / np.pi
fgo_eulers = fgo_eulers * 180 / np.pi
avip_eulers = avip_eulers * 180 / np.pi
svin2_eulers = svin2_eulers * 180 / np.pi
vins_eulers = vins_orientations * 180 / np.pi
avip_eulers = avip_eulers * 180 / np.pi

fig, axs = plt.subplots(3, 1, figsize=(10, 15))
axs[0].plot(ground_truth_timestamps, ground_truth_eulers[:, 0], label='Ground Truth', color='grey', linestyle='--')
axs[0].plot(vins_timestamps, vins_orientations[:, 0] * 180 / np.pi, label='VINS', color=color[4])
axs[0].plot(svin2_timestamps, svin2_eulers[:, 0], label='SVIN2', color=color[1])
axs[0].plot(avip_timestamps, avip_eulers[:, 0], label='AVIP', color=color[0])
axs[0].plot(filtered_odometry_timestamps, filtered_odometry_eulers[:, 0], label='UA-AVIP', color=color[2])
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Roll (degree)')
axs[0].grid()

axs[1].plot(ground_truth_timestamps, ground_truth_eulers[:, 1], label='Ground Truth', color='grey', linestyle='--')
axs[1].plot(vins_timestamps, vins_eulers[:, 1], label='VINS', color=color[4])
axs[1].plot(svin2_timestamps, svin2_eulers[:, 1], label='SVIN2', color=color[1])
axs[1].plot(avip_timestamps, avip_eulers[:, 1], label='AVIP', color=color[0])
axs[1].plot(fgo_timestamps, fgo_eulers[:, 1], label='UA-AVIP', color=color[2])
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Pitch (degree)')
axs[1].grid()

axs[2].plot(ground_truth_timestamps, ground_truth_eulers[:, 2], label='Ground Truth', color='grey', linestyle='--')
axs[2].plot(vins_timestamps, vins_orientations[:, 2] * 180 / np.pi, label='VINS', color=color[4])
axs[2].plot(svin2_timestamps, svin2_eulers[:, 2], label='SVIN2', color=color[1])
axs[2].plot(avip_timestamps, avip_eulers[:, 2], label='AVIP', color=color[0])
axs[2].plot(fgo_timestamps, fgo_eulers[:, 2], label='UA-AVIP', color=color[2])


axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('Yaw (degree)')
axs[2].grid()

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 3.67), ncol=7, fontsize=14)

plt.show()
