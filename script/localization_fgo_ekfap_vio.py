import gtsam
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Slerp, Rotation as R
from scipy.stats import chi2
import tf

# Load VIO data
from vio_traj import variance_x as vio_var_x, variance_y as vio_var_y, variance_z as vio_var_z
vins_vio_data = np.loadtxt('vins_vio_tum.txt')
vio_timestamps = vins_vio_data[:, 0]
vio_x = vins_vio_data[:, 1]
vio_y = vins_vio_data[:, 2]
vio_z = vins_vio_data[:, 3]
vio_qx = vins_vio_data[:, 4]
vio_qy = vins_vio_data[:, 5]
vio_qz = vins_vio_data[:, 6]
vio_qw = vins_vio_data[:, 7]

vio_x = vio_x - vio_x[0]
vio_y = vio_y - vio_y[0]
vio_z = vio_z - vio_z[0]

# Load EKF-based acoustic/pressure data from a txt file for TUM format
ekf_data = np.loadtxt('ekf_traj_ap_v3_adjust.txt')
ekf_timestamps = ekf_data[:, 0]
ekf_x = ekf_data[:, 1]
ekf_y = ekf_data[:, 2]
ekf_z = ekf_data[:, 3]
ekf_qx = ekf_data[:, 4]
ekf_qy = ekf_data[:, 5]
ekf_qz = ekf_data[:, 6]
ekf_qw = ekf_data[:, 7]

ekf_x = ekf_x - ekf_x[0]
ekf_y = ekf_y - ekf_y[0]
ekf_z = ekf_z - ekf_z[0]

# Interpolate the EKF data to match the VIO timestamps
ekf_x_interp = interp1d(ekf_timestamps, ekf_x, kind='linear', fill_value='extrapolate')
ekf_y_interp = interp1d(ekf_timestamps, ekf_y, kind='linear', fill_value='extrapolate')
ekf_z_interp = interp1d(ekf_timestamps, ekf_z, kind='linear', fill_value='extrapolate')

ekf_qx_interp = interp1d(ekf_timestamps, ekf_qx, kind='linear', fill_value='extrapolate')
ekf_qy_interp = interp1d(ekf_timestamps, ekf_qy, kind='linear', fill_value='extrapolate')
ekf_qz_interp = interp1d(ekf_timestamps, ekf_qz, kind='linear', fill_value='extrapolate')
ekf_qw_interp = interp1d(ekf_timestamps, ekf_qw, kind='linear', fill_value='extrapolate')

ekf_x_adjusted = ekf_x_interp(vio_timestamps)
ekf_y_adjusted = ekf_y_interp(vio_timestamps)
ekf_z_adjusted = ekf_z_interp(vio_timestamps)

ekf_qx_adjusted = ekf_qx_interp(vio_timestamps)
ekf_qy_adjusted = ekf_qy_interp(vio_timestamps)
ekf_qz_adjusted = ekf_qz_interp(vio_timestamps)
ekf_qw_adjusted = ekf_qw_interp(vio_timestamps)

# Initialize factor graph and initial estimate
graph = gtsam.NonlinearFactorGraph()
initial_estimate = gtsam.Values()

# Function to calculate noise based on chi-square distribution
def calculate_noise(variance, degrees_of_freedom, confidence_level):
    chi_square_value = chi2.ppf(confidence_level, degrees_of_freedom)
    noise_std = np.sqrt(variance * chi_square_value / degrees_of_freedom)
    return gtsam.noiseModel.Diagonal.Sigmas(np.concatenate([noise_std, [0.1, 0.1, 0.1]]))  # Adding small rotation noise

# Set degrees of freedom
degrees_of_freedom = 6  # For 6D measurements (x, y, z, roll, pitch, yaw)

# Calculate confidence levels based on desired confidence intervals
confidence_level_ekf = 0.95  # Example confidence level for EKF
confidence_level_vio = 0.85  # Example confidence level for VIO

# Define variances for EKF and VIO measurements
# Calculate EKF variances based on the EKF data
ekf_var_x = np.var(ekf_x_adjusted)
ekf_var_y = np.var(ekf_y_adjusted)
ekf_var_z = np.var(ekf_z_adjusted)
ekf_var_pos = np.array([ekf_var_x, ekf_var_y, ekf_var_z])
ekf_var_rot = np.array([0.01, 0.01, 0.01])  # Example variances for EKF rotation

vio_var_pos = np.array([vio_var_x, vio_var_y, vio_var_z]) 
vio_var_rot = np.array([0.01, 0.01, 0.01])  # Example variances for VIO rotation

# Calculate noise models
ekf_noise_model_pos = calculate_noise(ekf_var_pos, degrees_of_freedom, confidence_level_ekf)
ekf_noise_model_rot = calculate_noise(ekf_var_rot, degrees_of_freedom, confidence_level_ekf)
vio_noise_model_pos = calculate_noise(vio_var_pos, degrees_of_freedom, confidence_level_vio)
vio_noise_model_rot = calculate_noise(vio_var_rot, degrees_of_freedom, confidence_level_vio)

# Add prior factor
initial_pose = gtsam.Pose3(gtsam.Rot3(ekf_qw_adjusted[0], ekf_qx_adjusted[0], ekf_qy_adjusted[0], ekf_qz_adjusted[0]),
                           gtsam.Point3(ekf_x_adjusted[0], ekf_y_adjusted[0], ekf_z_adjusted[0]))
prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.concatenate([ekf_var_pos, ekf_var_rot]))
graph.add(gtsam.PriorFactorPose3(0, initial_pose, prior_noise))
initial_estimate.insert(0, initial_pose)

# Add factors for EKF-based odometry
for i in range(1, len(vio_timestamps)):
    # Calculate adaptive weights
    S_ekf_pos = ekf_var_pos
    S_vio_pos = vio_var_pos
    w_ekf_pos = S_vio_pos / (S_ekf_pos + S_vio_pos)
    w_vio_pos = S_ekf_pos / (S_ekf_pos + S_vio_pos)
    
    S_ekf_rot = ekf_var_rot
    S_vio_rot = vio_var_rot
    w_ekf_rot = S_vio_rot / (S_ekf_rot + S_vio_rot)
    w_vio_rot = S_ekf_rot / (S_ekf_rot + S_vio_rot)

    # EKF position and orientation
    ekf_pose = gtsam.Pose3(gtsam.Rot3(ekf_qw_adjusted[i], ekf_qx_adjusted[i], ekf_qy_adjusted[i], ekf_qz_adjusted[i]),
                           gtsam.Point3(ekf_x_adjusted[i], ekf_y_adjusted[i], ekf_z_adjusted[i]))
    # graph.add(gtsam.PriorFactorPose3(i, ekf_pose, ekf_noise_model_pos))
    ekf_noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.concatenate([w_ekf_pos * ekf_var_pos, w_ekf_rot * ekf_var_rot]))
    graph.add(gtsam.PriorFactorPose3(i, ekf_pose, ekf_noise_model))
    initial_estimate.insert(i, ekf_pose)

    # VIO position and orientation
    vio_pose = gtsam.Pose3(gtsam.Rot3(),
                           gtsam.Point3(vio_x[i], vio_y[i], vio_z[i]))
    vio_noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.concatenate([w_vio_pos * vio_var_pos, w_vio_rot * vio_var_rot]))
    graph.add(gtsam.BetweenFactorPose3(i-1, i, vio_pose, vio_noise_model_pos))

# Optimize the graph
optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
result = optimizer.optimize()

# Save the optimized poses
optimized_poses = []
for i in range(len(vio_timestamps)):
    pose = result.atPose3(i)
    optimized_poses.append(pose)

# Plot the optimized trajectory
import matplotlib.pyplot as plt

# Extract x, y, z coordinates
x_opt = [pose.x() for pose in optimized_poses]
y_opt = [pose.y() for pose in optimized_poses]
z_opt = [pose.z() for pose in optimized_poses]

plt.plot(x_opt, y_opt)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Optimized Trajectory')
plt.show()

# Save the optimized poses to a file
# with open('fgo_02_adap_3.txt', 'w') as f:
#     for i, pose in enumerate(optimized_poses):
#         timestamp = vio_timestamps[i]
#         tx, ty, tz = pose.x(), pose.y(), pose.z()
#         q = pose.rotation().toQuaternion()
#         qx, qy, qz, qw = q.x(), q.y(), q.z(), q.w()
#         f.write(f"{timestamp} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")
