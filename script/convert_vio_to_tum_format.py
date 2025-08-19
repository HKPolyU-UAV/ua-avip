import pandas as pd

# Load the data from the CSV file
df = pd.read_csv('/home/xyb/ua_avip/vins_vio.csv')

# Extract the relevant columns
timestamps = df['timestamp'].values
px = df['px'].values
py = df['py'].values
pz = df['pz'].values
qw = df['qw'].values
qx = df['qx'].values
qy = df['qy'].values
qz = df['qz'].values

# Create the TUM format text file
with open('vins_vio_tum.txt', 'w') as f:
    for i in range(len(timestamps)):
        # timestamp = timestamps[i] / 1e9  # Convert from nanoseconds to seconds
        timestamp = timestamps[i]
        tx = px[i]
        ty = py[i]
        tz = pz[i]
        qx_val = qx[i]
        qy_val = qy[i]
        qz_val = qz[i]
        qw_val = qw[i]
        f.write(f"{timestamp} {tx} {ty} {tz} {qx_val} {qy_val} {qz_val} {qw_val}\n")
