import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from csv
df = pd.read_csv('dvl_pos_2.csv')

# time_values_dvl_pos = np.array(df['field.header.stamp'])  # Convert time values to numpy array

# Extract the positions estimated by DVL
positions_x = df['field.orientation.x'].values
positions_y = df['field.orientation.y'].values
positions_z = df['field.orientation.z'].values

# Extract the roll, pitch, and yaw angles
roll = df['field.angular_velocity.x'].values * np.pi / 180  # Convert to radians
pitch = df['field.angular_velocity.y'].values * np.pi / 180  # Convert to radians
yaw = df['field.angular_velocity.z'].values * np.pi / 180  # Convert to radians

# roll minus 180 degrees
roll = roll - np.pi
# yaw minus 90 degrees
yaw = yaw - np.pi / 2

# Debugging: check the values of roll
# print("Roll values:", roll)
if np.any(roll > 0):
    print("*********************************Roll values:", roll)

# Convert euler angles to quaternion
def euler_to_quaternion(roll, pitch, yaw):
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    # qw = cy * cp * cr + sy * sp * sr
    # qx = cy * cp * sr - sy * sp * cr
    # qy = sy * cp * sr + cy * sp * cr
    # qz = sy * cp * cr - cy * sp * sr

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return qw, qx, qy, qz

# # Transform the positions to the world coordinate system
# for i in range(1, len(positions_x)):
#     # Calculate the change in position
#     dx = np.cos(yaw[i]) * positions_x[i] - np.sin(yaw[i]) * positions_y[i]
#     dy = np.sin(yaw[i]) * positions_x[i] + np.cos(yaw[i]) * positions_y[i]

#     # Update the positions
#     positions_x[i] = positions_x[i-1] + dx
#     positions_y[i] = positions_y[i-1] + dy


positions_x = - positions_x + positions_x[0]
positions_y = - positions_y + positions_y[0]

print("First position:", positions_x[0], positions_y[0])
print("Length of positions:", len(positions_x))
# print("Length of time values:", len(time_values_dvl_pos))

# Plotting the trajectory
from localization_dvl_vel_02 import positions_x as dvl_vel_positions_x, positions_y as dvl_vel_positions_y

plt.figure(figsize=(10, 6))
time_values = np.array(df['field.header.stamp'])  # Convert time values to numpy array
plt.plot(positions_x, positions_y, label='DVL Pos')


plt.plot(dvl_vel_positions_x, dvl_vel_positions_y, label='DVL Vel')

# Start and end points
plt.plot(positions_x[0], positions_y[0], 'go', label='Start')
plt.plot(positions_x[-1], positions_y[-1], 'ro', label='End')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Vehicle Trajectory Estimated by DVL')
plt.legend()

plt.show()


# Calculate the variance of DVL measurements
variance_x = np.var(df['field.orientation.x'])
variance_y = np.var(df['field.orientation.y'])
variance_z = np.var(df['field.orientation.z'])

print("Variance of DVL position measurements in X direction:", variance_x)
print("Variance of DVL position measurements in Y direction:", variance_y)
print("Variance of DVL position measurements in Z direction:", variance_z)

# Save results to TXT in TUM format (time tx ty tz qx qy qz qw)
with open('dvl_pos_dr_2_pose_2.txt', 'w') as f:
    for i in range(len(time_values)):
        timestamp = time_values[i] / 1e9  # Convert from nanoseconds to seconds
        tx = positions_x[i]
        ty = positions_y[i]
        tz = positions_z[i]
        # roll[i] -= np.pi
        # if roll[i] > -1.74532925:
        #     print("***************************Unexpected roll value:", roll[i])
        print("Roll value:", roll[i])
        qx = euler_to_quaternion(roll[i], pitch[i], yaw[i])[1]
        qy = euler_to_quaternion(roll[i], pitch[i], yaw[i])[2]
        qz = euler_to_quaternion(roll[i], pitch[i], yaw[i])[3]
        qw = euler_to_quaternion(roll[i], pitch[i], yaw[i])[0]
        f.write(f"{timestamp} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")