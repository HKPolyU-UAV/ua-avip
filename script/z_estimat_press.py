import pandas as pd
import numpy as np

# Load the fluid pressure data from the CSV file
df = pd.read_csv('press.csv')

# Extract the pressure values
press = df['field.fluid_pressure'].values

timestamps = df['field.header.stamp'].values

# Calculate the z coordinates over time based on the pressure values
z = press / 100  # cm to m

# for i in range(700, 800):
#     print(f'{timestamps[i]}: {z[i]}')

for i in range(0, len(z)):
    if abs(z[i]) > 1:
        # print("Find outlier at index: ", i)
        z[i] = z[i-1]
    if abs(z[i]-z[i-1]) > 0.06:
            z[i] = z[i-1]

# Plot the pressure values over time
import matplotlib.pyplot as plt
# plt.plot(timestamps, press)
# plt.xlabel('Time')
# plt.ylabel('Pressure')
# plt.title('Pressure over time')
# plt.show()

# Plot the z coordinates over time
plt.plot(timestamps, z)
plt.xlabel('Time')
plt.ylabel('Z coordinate')
plt.title('Z coordinate over time')
plt.show()

# Calculate the variance of pressure sensor measurements
variance_press = np.var(z)
print("Variance of pressure sensor measurements:", variance_press)
