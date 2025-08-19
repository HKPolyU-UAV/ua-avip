import csv
import numpy as np

input_file = '/home/xyb/ua_avip/ground_truth_02.csv'
output_file = '/home/xyb/ua-avip/gt_displacement_02.csv'

def calculate_displacement(pos1, pos2):
    return [pos2[i] - pos1[i] for i in range(3)]

with open(input_file, mode='r', newline='') as infile, open(output_file, mode='w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    header = next(reader)
    writer.writerow(['timestamp', 'displacement_x', 'displacement_y', 'displacement_z'])
    
    prev_row = next(reader)
    prev_pos = [float(prev_row[1]), float(prev_row[2]), float(prev_row[3])]
    accumulated_displacement = [0.0, 0.0, 0.0]
    frame_count = 0
    
    for row in reader:
        timestamp = row[0]
        current_pos = [float(row[1]), float(row[2]), float(row[3])]
        displacement = calculate_displacement(prev_pos, current_pos)
        
        # Accumulate displacement
        accumulated_displacement = [accumulated_displacement[i] + displacement[i] for i in range(3)]
        frame_count += 1
        
        # Output displacement every 1 second (approximately 17 frames)
        if frame_count >= 17:
            writer.writerow([timestamp] + accumulated_displacement)
            accumulated_displacement = [0.0, 0.0, 0.0]
            frame_count = 0
        
        prev_pos = current_pos
