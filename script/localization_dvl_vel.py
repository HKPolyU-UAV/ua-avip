import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dvl data from csv
df = pd.read_csv('dvl_vel_2.csv')

# load the imu data from csv providing orientation
df_imu = pd.read_csv('imu_2.csv')

# Extract relevant columns from IMU data
timestamps_imu = df_imu['field.header.stamp'].values
orientation = df_imu[['field.orientation.x', 'field.orientation.y', 'field.orientation.z', 'field.orientation.w']].values

# Helper function to convert quaternion to rotation matrix
def quaternion_to_rotation_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)]
    ])

# Synchronize the DVL and IMU data based on timestamps
def synchronize_orientation(time_values_dvl, time_values_imu, orientations):
    synchronized_orientations = []
    for t_dvl in time_values_dvl:
        closest_idx = np.argmin(np.abs(time_values_imu - t_dvl))
        synchronized_orientations.append(orientations[closest_idx])
    return np.array(synchronized_orientations)

# Synchronize orientations
time_values_dvl = df['field.header.stamp'].values
synchronized_orientations = synchronize_orientation(time_values_dvl, timestamps_imu, orientation)

# Initialize variables
positions_x = [0]
positions_y = [0]
positions_z = [0]
prev_time = df.iloc[0]['field.header.stamp']
syn_orient_quat = []
syn_orient_quat.append(synchronized_orientations[0])

# Dead reckoning estimation
for index, row in df.iterrows():
    # time = row['%time']
    time = row['field.header.stamp']
    dt = time - prev_time if index > 0 else 0
    dt = dt / 1e9  # Convert nanoseconds to seconds
    vel_x = row['field.linear_acceleration.x']
    vel_y = row['field.linear_acceleration.y']
    vel_z = row['field.linear_acceleration.z']

    # Get the corresponding orientation
    orientation_quat = synchronized_orientations[index]
    rot_matrix = quaternion_to_rotation_matrix(orientation_quat)
    syn_orient_quat.append(orientation_quat)

    # Rotate the DVL velocity to the global frame
    vel_dvl = np.array([vel_x, vel_y, vel_z])
    vel_global = rot_matrix.dot(vel_dvl)
    
    # Update positions
    velocity_x = positions_x[-1] + vel_global[0] * dt
    velocity_y = positions_y[-1] + vel_global[1] * dt
    velocity_z = positions_z[-1] + vel_global[2] * dt

    positions_x.append(velocity_x)
    positions_y.append(velocity_y)
    positions_z.append(velocity_z)
    
    prev_time = time

    # print("vel_x: ", vel_x)
    # print("vel_y: ", vel_y)
    # print("dt: ", dt)

positions_x = np.array(positions_x)
positions_y = np.array(positions_y)
positions_z = np.array(positions_z)

# Coordinate transformation
# positions_x = - positions_x
# positions_y = - positions_y

print("First position:", positions_x[0], positions_y[0])

syn_orient_quat = np.array(syn_orient_quat)
print("Length of syn_orient_quat: ", len(syn_orient_quat))
print("First orientation:", syn_orient_quat[0])

# Calculate the variance of DVL measurements
variance_x = np.var(df['field.linear_acceleration.x'])
variance_y = np.var(df['field.linear_acceleration.y'])
variance_z = np.var(df['field.linear_acceleration.z'])

print("Variance of DVL velocity measurements in X direction:", variance_x)
print("Variance of DVL velocity measurements in Y direction:", variance_y)
print("Variance of DVL velocity measurements in Z direction:", variance_z)

# Save results to TXT in TUM format (time tx ty tz qx qy qz qw)
# with open('dvl_pos_vel.txt', 'w') as f:
#     for i in range(len(time_values)):
#         timestamp = time_values[i] / 1e9  # Convert from nanoseconds to seconds
#         tx = positions_x[i+1]
#         ty = positions_y[i+1]
#         tz = positions_z[i+1]
#         qx = synchronized_orientations[i][0]
#         qy = synchronized_orientations[i][1]
#         qz = synchronized_orientations[i][2]
#         qw = synchronized_orientations[i][3]
#         f.write(f"{timestamp} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")
