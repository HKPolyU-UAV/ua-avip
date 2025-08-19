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
fgo_file = '/home/xyb/underwaterDataset/casia/CASIA_PolyU_SLAM_datasets_20241108/error_calcu_2/fgo_02_adap_2.tum'
ground_truth_file = '/home/xyb/underwaterDataset/casia/CASIA_PolyU_SLAM_datasets_20241108/error_calcu_2/ground_truth_02_one_4.tum'
fgo_xy_file = '/home/xyb/underwaterDataset/casia/CASIA_PolyU_SLAM_datasets_20241108/error_calcu_2/fgo_02_adap_2_xyplot.tum'

# Read the TUM files
fgo_timestamps, fgo_positions, fgo_orientations = read_tum_file(fgo_file)
ground_truth_timestamps, ground_truth_positions, ground_truth_orientations = read_tum_file(ground_truth_file)
fgo_xy_timestamps, fgo_xy_positions, fgo_xy_orientations = read_tum_file(fgo_xy_file)

# Comparison of other methods #
svin2_file = 'svin_2025_01_06_14_26_29.tum'
vins_file = 'vins_vio_tum.tum'
avip_file = 'fgo_02_2_pose.tum'
avip_ekf_file = 'ekf_traj_ap_v3.tum'
avip_xy_file = 'fgo_02_2_pose_xyplot.tum'

svin2_timestamps, svin2_positions, svin2_orientations = read_tum_file(svin2_file)
vins_timestamps, vins_positions, vins_orientations = read_tum_file(vins_file)
avip_timestamps, avip_positions, avip_orientations = read_tum_file(avip_file)
avip_ekf_timestamps, avip_ekf_positions, avip_ekf_orientations = read_tum_file(avip_ekf_file)
avip_xy_timestamps, avip_xy_positions, avip_xy_orientations = read_tum_file(avip_xy_file)

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
fgo_xy_positions[:, 1] = -fgo_xy_positions[:, 1]
avip_xy_positions[:, 1] = -avip_xy_positions[:, 1]

# Plot the trajectories (XY)
plt.figure(figsize=(10, 15))
# ax = plt.gca()
# ax.set_aspect('equal', adjustable='box')
plt.plot(ground_truth_positions[:, 0], ground_truth_positions[:, 1], label='Ground Truth', color='grey', linestyle='--')
plt.plot(vins_positions[:, 0], vins_positions[:, 1], label='VINS', color=color[4])
plt.plot(svin2_positions[:, 0], svin2_positions[:, 1], label='SVIN2', color=color[1])
# plt.plot(avip_positions[:, 0], avip_positions[:, 1], label='AVIP', color=color[0])
plt.plot(avip_xy_positions[:, 0], avip_xy_positions[:, 1], label='AVIP', color=color[0])
# plt.plot(fgo_positions[:, 0], fgo_positions[:, 1], label='UA-AVIP', color=color[2])
plt.plot(fgo_xy_positions[:, 0], fgo_xy_positions[:, 1], label='UA-AVIP', color=color[2])

# Start and end points for each trajectory corresponding to their colors
plt.scatter(ground_truth_positions[0, 0], ground_truth_positions[0, 1], color='grey', marker= 'o', label='Start')
plt.scatter(ground_truth_positions[-1, 0], ground_truth_positions[-1, 1], color='black', marker = 's', label='End')
# plt.scatter(fgo_positions[0, 0], fgo_positions[0, 1], color=color[2], marker= 'o')
# plt.scatter(fgo_positions[-1, 0], fgo_positions[-1, 1], color=color[2], marker = 's')
plt.scatter(fgo_xy_positions[0, 0], fgo_xy_positions[0, 1], color=color[2], marker= 'o')
plt.scatter(fgo_xy_positions[-1, 0], fgo_xy_positions[-1, 1], color=color[2], marker = 's')
# plt.scatter(avip_positions[0, 0], avip_positions[0, 1], color=color[0], marker= 'o')
# plt.scatter(avip_positions[-1, 0], avip_positions[-1, 1], color=color[0], marker = 's')
plt.scatter(avip_xy_positions[0, 0], avip_xy_positions[0, 1], color=color[0], marker= 'o')
plt.scatter(avip_xy_positions[-1, 0], avip_xy_positions[-1, 1], color=color[0], marker = 's')
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
# plt.title('Trajectory Comparison (XY)', fontname='Times New Roman')
# plt.legend()
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize=14)
plt.grid()
# plt.savefig('trajectory_comparison_xy_seq01.jpg')
plt.show()

# Plot the XYZ positions

# ground_truth_positions = ground_truth_positions - ground_truth_positions[0]
# fgo_positions = fgo_positions - fgo_positions[0]
# svin2_positions = svin2_positions - svin2_positions[0]
# vins_positions = vins_positions - vins_positions[0]
# avip_positions = avip_positions - avip_positions[0]

svin2_positions[:, 2] = svin2_positions[:, 2] - svin2_positions[0, 2] - 0.09

fig, axs = plt.subplots(3, 1, figsize=(10, 15))
axs[0].plot(ground_truth_timestamps, ground_truth_positions[:, 0], label='Ground Truth', color='grey', linestyle='--')
axs[0].plot(vins_timestamps, vins_positions[:, 0], label='VINS', color=color[4])
axs[0].plot(svin2_timestamps, svin2_positions[:, 0], label='SVIN2', color=color[1])
# axs[0].plot(avip_timestamps, avip_positions[:, 0], label='AVIP', color=color[0])
axs[0].plot(avip_xy_timestamps, avip_xy_positions[:, 0], label='AVIP', color=color[0])
# axs[0].plot(fgo_timestamps, fgo_positions[:, 0], label='UA-AVIP', color=color[2])
axs[0].plot(fgo_xy_timestamps, fgo_xy_positions[:, 0], label='UA-AVIP', color=color[2])
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('X (m)')
# axs[0].legend()
axs[0].grid()

axs[1].plot(ground_truth_timestamps, ground_truth_positions[:, 1], label='Ground Truth', color='grey', linestyle='--')
axs[1].plot(vins_timestamps, vins_positions[:, 1], label='VINS', color=color[4])
axs[1].plot(svin2_timestamps, svin2_positions[:, 1], label='SVIN2', color=color[1])
# axs[1].plot(avip_timestamps, avip_positions[:, 1], label='AVIP', color=color[0])
axs[1].plot(avip_xy_timestamps, avip_xy_positions[:, 1], label='AVIP', color=color[0])
# axs[1].plot(fgo_timestamps, fgo_positions[:, 1], label='UA-AVIP', color=color[2])
axs[1].plot(fgo_xy_timestamps, fgo_xy_positions[:, 1], label='UA-AVIP', color=color[2])


axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Y (m)')
# axs[1].legend()
axs[1].grid()

axs[2].plot(ground_truth_timestamps, ground_truth_positions[:, 2], label='Ground Truth', color='grey', linestyle='--')
axs[2].plot(vins_timestamps, vins_positions[:, 2], label='VINS', color=color[4])
axs[2].plot(svin2_timestamps, svin2_positions[:, 2], label='SVIN2', color=color[1])
# axs[2].plot(avip_timestamps, avip_positions[:, 2], label='AVIP', color=color[0])
axs[2].plot(avip_xy_timestamps, avip_xy_positions[:, 2], label='AVIP', color=color[0])
# axs[2].plot(fgo_timestamps, fgo_positions[:, 2], label='UA-AVIP', color=color[2])
axs[2].plot(fgo_xy_timestamps, fgo_xy_positions[:, 2], label='UA-AVIP', color=color[2])
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('Z (m)')
# axs[2].legend()
axs[2].grid()

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 3.67), ncol=7, fontsize=14)

# plt.suptitle('XYZ Position Comparison', fontname='Times New Roman')
# plt.savefig('xyz_position_comparison_seq01.jpg')
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

# File paths
# odometry_file = 'fgo_02_adap_2.txt'
odometry_file = 'fgo_02_adap_sw_2.txt'

# Read the TUM files
odometry_timestamps, odometry_positions, odometry_orientations = read_tum_file(odometry_file)

# Convert the quaternions to euler angles
odometry_eulers = quaternion_to_euler(odometry_orientations)
ground_truth_eulers_1 = quaternion_to_euler(ground_truth_orientations)

ground_truth_eulers = np.array([tf.transformations.euler_from_quaternion(orientation) for orientation in ground_truth_orientations])
fgo_eulers = np.array([tf.transformations.euler_from_quaternion(orientation) for orientation in fgo_orientations])
avip_eulers = np.array([tf.transformations.euler_from_quaternion(orientation) for orientation in avip_orientations])
svin2_eulers = np.array([tf.transformations.euler_from_quaternion(orientation) for orientation in svin2_orientations])
vins_eulers = np.array([tf.transformations.euler_from_quaternion(orientation) for orientation in vins_orientations])
avip_ekf_eulers = np.array([tf.transformations.euler_from_quaternion(orientation) for orientation in avip_ekf_orientations])

# radian to degree
ground_truth_eulers_1 = ground_truth_eulers_1 * 180 / np.pi
fgo_eulers = fgo_eulers * 180 / np.pi
avip_eulers = avip_eulers * 180 / np.pi
svin2_eulers = svin2_eulers * 180 / np.pi
vins_eulers = vins_orientations * 180 / np.pi
avip_ekf_eulers = avip_ekf_eulers * 180 / np.pi

odometry_eulers[:, 0] -= 3

odometry_eulers = odometry_eulers * 180 / np.pi
ground_truth_eulers = ground_truth_eulers * 180 / np.pi

# Filter odometry data to exclude the first six seconds
odometry_mask = odometry_timestamps >= 1730995832.5565553
filtered_odometry_timestamps = odometry_timestamps[odometry_mask]
filtered_odometry_eulers = odometry_eulers[odometry_mask]

ground_truth_eulers_1[:, 0] = ground_truth_eulers_1[:, 0] - ground_truth_eulers_1[0, 0]
filtered_odometry_eulers[:, 0] = filtered_odometry_eulers[:, 0] - filtered_odometry_eulers[0, 0]
svin2_eulers = svin2_eulers - svin2_eulers[0]
vins_orientations[:, 0] = vins_orientations[:, 0] - vins_orientations[0, 0]
avip_ekf_eulers[:, 0] = avip_ekf_eulers[:, 0] - avip_ekf_eulers[0, 0]

avip_ekf_eulers[:, 0] = - avip_ekf_eulers[:, 0]

for i in range(len(avip_ekf_eulers)):
    if avip_ekf_eulers[i, 0] < 300:
        avip_ekf_eulers[i, 0] = avip_ekf_eulers[i - 1, 0]
    if avip_eulers[i, 1] < - 30:
        avip_eulers[i, 1] = avip_eulers[i - 1, 1]

avip_ekf_eulers[:, 0] = avip_ekf_eulers[:, 0] - avip_ekf_eulers[0, 0]
# avip_orientations[:,1] = savgol_filter(avip_orientations[:,1], 25, 3)

fig, axs = plt.subplots(3, 1, figsize=(10, 15))
axs[0].plot(ground_truth_timestamps, ground_truth_eulers_1[:, 0], label='Ground Truth', color='grey', linestyle='--')
# axs[0].plot(odometry_timestamps, odometry_eulers[:, 0], label='Factor Graph Optimization', color='red', linestyle='-.')
axs[0].plot(vins_timestamps, vins_orientations[:, 0] * 180 / np.pi, label='VINS', color=color[4])
axs[0].plot(svin2_timestamps, svin2_eulers[:, 0], label='SVIN2', color=color[1])
axs[0].plot(avip_ekf_timestamps, avip_ekf_eulers[:, 0], label='AVIP', color=color[0])
axs[0].plot(filtered_odometry_timestamps, filtered_odometry_eulers[:, 0], label='UA-AVIP', color=color[2])
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Roll (degree)')
# axs[0].legend()
axs[0].grid()

fgo_eulers[:, 1] = fgo_eulers[:, 1] - 1.22
# fgo_timestamps_pitch = fgo_timestamps - 3.5

# ground_truth_eulers[:, 1] = ground_truth_eulers[:, 1] - ground_truth_eulers[0, 1]
# fgo_eulers[:, 1] = fgo_eulers[:, 1] - fgo_eulers[0, 1]
svin2_orientations[:, 1] = svin2_orientations[:, 1] - svin2_orientations[0, 1]
vins_orientations[:, 1] = vins_orientations[:, 1] - vins_orientations[0, 1]

# Remove the pitch angle estimated by VINS
for i in range(len(vins_orientations)):
    if vins_orientations[i, 1] * 180 / np.pi < - 20:
        vins_orientations[i, 1] = vins_orientations[i - 1, 1]

vins_eulers[:, 1] = - vins_eulers[:, 1]

# avip_ekf_eulers[:, 1] = avip_ekf_eulers[:, 1] - avip_ekf_eulers[0, 1]
svin2_eulers[:, 1] = svin2_eulers[:, 1] - svin2_eulers[0, 1] + ground_truth_eulers[0, 1]
vins_eulers[:, 1] = vins_eulers[:, 1] - vins_eulers[0, 1] + ground_truth_eulers[0, 1]

for i in range(len(vins_eulers)):
    if vins_eulers[i, 1] > 20:
        vins_eulers[i, 1] = vins_eulers[i - 1, 1]

axs[1].plot(ground_truth_timestamps, ground_truth_eulers[:, 1], label='Ground Truth', color='grey', linestyle='--')
axs[1].plot(vins_timestamps, vins_eulers[:, 1], label='VINS', color=color[4])
axs[1].plot(svin2_timestamps, svin2_eulers[:, 1], label='SVIN2', color=color[1])
axs[1].plot(avip_ekf_timestamps, avip_ekf_eulers[:, 1], label='AVIP', color=color[0])
axs[1].plot(fgo_timestamps, fgo_eulers[:, 1], label='UA-AVIP', color=color[2])
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Pitch (degree)')
# axs[1].legend()
axs[1].grid()

ground_truth_eulers[:, 2] = ground_truth_eulers[:, 2] - ground_truth_eulers[0, 2]
fgo_eulers[:, 2] = fgo_eulers[:, 2] - fgo_eulers[0, 2]
vins_orientations[:, 2] = vins_orientations[:, 2] - vins_orientations[0, 2]

avip_ekf_eulers[:, 2] = avip_ekf_eulers[:, 2] - avip_ekf_eulers[0, 2]

axs[2].plot(ground_truth_timestamps, ground_truth_eulers[:, 2], label='Ground Truth', color='grey', linestyle='--')
axs[2].plot(vins_timestamps, vins_orientations[:, 2] * 180 / np.pi, label='VINS', color=color[4])
axs[2].plot(svin2_timestamps, svin2_eulers[:, 2], label='SVIN2', color=color[1])
axs[2].plot(avip_ekf_timestamps, avip_ekf_eulers[:, 2], label='AVIP', color=color[0])
axs[2].plot(fgo_timestamps, fgo_eulers[:, 2], label='UA-AVIP', color=color[2])


axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('Yaw (degree)')
# axs[2].legend()
axs[2].grid()

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 3.67), ncol=7, fontsize=14)

# plt.suptitle('RPY Angle Comparison', fontname='Times New Roman')
# plt.savefig('rpy_angle_comparison_seq01.jpg')
plt.show()

# Add this after the RPY angle plotting section
# Calculate position errors over time
ground_truth_interp_x = interp1d(ground_truth_timestamps, ground_truth_positions[:, 0], bounds_error=False, fill_value="extrapolate")
ground_truth_interp_y = interp1d(ground_truth_timestamps, ground_truth_positions[:, 1], bounds_error=False, fill_value="extrapolate")
ground_truth_interp_z = interp1d(ground_truth_timestamps, ground_truth_positions[:, 2], bounds_error=False, fill_value="extrapolate")

# Calculate orientation errors over time
ground_truth_interp_roll = interp1d(ground_truth_timestamps, ground_truth_eulers_1[:, 0], bounds_error=False, fill_value="extrapolate")
ground_truth_interp_pitch = interp1d(ground_truth_timestamps, ground_truth_eulers[:, 1], bounds_error=False, fill_value="extrapolate")
ground_truth_interp_yaw = interp1d(ground_truth_timestamps, ground_truth_eulers[:, 2], bounds_error=False, fill_value="extrapolate")

# Plot position errors
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# VINS position errors
vins_x_error = np.abs(vins_positions[:, 0] - ground_truth_interp_x(vins_timestamps))
vins_y_error = np.abs(vins_positions[:, 1] - ground_truth_interp_y(vins_timestamps))
vins_z_error = np.abs(vins_positions[:, 2] - ground_truth_interp_z(vins_timestamps))

# SVIN2 position errors
svin2_x_error = np.abs(svin2_positions[:, 0] - ground_truth_interp_x(svin2_timestamps))
svin2_y_error = np.abs(svin2_positions[:, 1] - ground_truth_interp_y(svin2_timestamps))
svin2_z_error = np.abs(svin2_positions[:, 2] - ground_truth_interp_z(svin2_timestamps))

# AVIP position errors
avip_x_error = np.abs(avip_xy_positions[:, 0] - ground_truth_interp_x(avip_xy_timestamps))
avip_y_error = np.abs(avip_xy_positions[:, 1] - ground_truth_interp_y(avip_xy_timestamps))
avip_z_error = np.abs(avip_xy_positions[:, 2] - ground_truth_interp_z(avip_xy_timestamps))

# UA-AVIP position errors
fgo_x_error = np.abs(fgo_xy_positions[:, 0] - ground_truth_interp_x(fgo_xy_timestamps))
fgo_y_error = np.abs(fgo_xy_positions[:, 1] - ground_truth_interp_y(fgo_xy_timestamps))
fgo_z_error = np.abs(fgo_xy_positions[:, 2] - ground_truth_interp_z(fgo_xy_timestamps))

# Plot X position error
axs[0].plot(vins_timestamps, vins_x_error, label='VINS', color=color[4])
axs[0].plot(svin2_timestamps, svin2_x_error, label='SVIN2', color=color[1])
axs[0].plot(avip_xy_timestamps, avip_x_error, label='AVIP', color=color[0])
axs[0].plot(fgo_xy_timestamps, fgo_x_error, label='UA-AVIP', color=color[2])
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('X Position Error (m)')
axs[0].grid()

# Plot Y position error
axs[1].plot(vins_timestamps, vins_y_error, label='VINS', color=color[4])
axs[1].plot(svin2_timestamps, svin2_y_error, label='SVIN2', color=color[1])
axs[1].plot(avip_xy_timestamps, avip_y_error, label='AVIP', color=color[0])
axs[1].plot(fgo_xy_timestamps, fgo_y_error, label='UA-AVIP', color=color[2])
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Y Position Error (m)')
axs[1].grid()

# Plot Z position error
axs[2].plot(vins_timestamps, vins_z_error, label='VINS', color=color[4])
axs[2].plot(svin2_timestamps, svin2_z_error, label='SVIN2', color=color[1])
axs[2].plot(avip_xy_timestamps, avip_z_error, label='AVIP', color=color[0])
axs[2].plot(fgo_xy_timestamps, fgo_z_error, label='UA-AVIP', color=color[2])
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('Z Position Error (m)')
axs[2].grid()

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 3.67), ncol=7, fontsize=14)
plt.show()

# Calculate and plot the orientation errors
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# VINS orientation errors
vins_roll_error = np.abs(vins_orientations[:, 0] * 180 / np.pi - ground_truth_interp_roll(vins_timestamps))
vins_pitch_error = np.abs(vins_eulers[:, 1] - ground_truth_interp_pitch(vins_timestamps))
vins_yaw_error = np.abs(vins_orientations[:, 2] * 180 / np.pi - ground_truth_interp_yaw(vins_timestamps))

# SVIN2 orientation errors
svin2_roll_error = np.abs(svin2_eulers[:, 0] - ground_truth_interp_roll(svin2_timestamps))
svin2_pitch_error = np.abs(svin2_eulers[:, 1] - ground_truth_interp_pitch(svin2_timestamps))
svin2_yaw_error = np.abs(svin2_eulers[:, 2] - ground_truth_interp_yaw(svin2_timestamps))

# AVIP orientation errors
avip_roll_error = np.abs(avip_ekf_eulers[:, 0] - ground_truth_interp_roll(avip_ekf_timestamps))
avip_pitch_error = np.abs(avip_ekf_eulers[:, 1] - ground_truth_interp_pitch(avip_ekf_timestamps))
avip_yaw_error = np.abs(avip_ekf_eulers[:, 2] - ground_truth_interp_yaw(avip_ekf_timestamps))

# UA-AVIP orientation errors
fgo_roll_error = np.abs(filtered_odometry_eulers[:, 0] - ground_truth_interp_roll(filtered_odometry_timestamps))
fgo_pitch_error = np.abs(fgo_eulers[:, 1] - ground_truth_interp_pitch(fgo_timestamps))
fgo_yaw_error = np.abs(fgo_eulers[:, 2] - ground_truth_interp_yaw(fgo_timestamps))

# Plot roll error
axs[0].plot(vins_timestamps, vins_roll_error, label='VINS', color=color[4])
axs[0].plot(svin2_timestamps, svin2_roll_error, label='SVIN2', color=color[1])
axs[0].plot(avip_ekf_timestamps, avip_roll_error, label='AVIP', color=color[0])
axs[0].plot(filtered_odometry_timestamps, fgo_roll_error, label='UA-AVIP', color=color[2])
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Roll Error (degree)')
axs[0].grid()

# Plot pitch error
axs[1].plot(vins_timestamps, vins_pitch_error, label='VINS', color=color[4])
axs[1].plot(svin2_timestamps, svin2_pitch_error, label='SVIN2', color=color[1])
axs[1].plot(avip_ekf_timestamps, avip_pitch_error, label='AVIP', color=color[0])
axs[1].plot(fgo_timestamps, fgo_pitch_error, label='UA-AVIP', color=color[2])
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Pitch Error (degree)')
axs[1].grid()

# Plot yaw error
axs[2].plot(vins_timestamps, vins_yaw_error, label='VINS', color=color[4])
axs[2].plot(svin2_timestamps, svin2_yaw_error, label='SVIN2', color=color[1])
axs[2].plot(avip_ekf_timestamps, avip_yaw_error, label='AVIP', color=color[0])
axs[2].plot(fgo_timestamps, fgo_yaw_error, label='UA-AVIP', color=color[2])
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('Yaw Error (degree)')
axs[2].grid()

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 3.67), ncol=7, fontsize=14)
plt.show()