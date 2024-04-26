import json
import numpy as np
import os


#Open a JSONL file and average the x, y, and z by accelaraiton
def normalizeAccel():
    # Open the JSONL file
    with open('data/2018/02/day=27/hour=17/00.jsonl', 'r') as file:
        lines = file.readlines()

    # Process each line
    for i, line in enumerate(lines):
        data = json.loads(line)

        # Normalize each array
        for axis in ['x', 'y', 'z']:
            arr = np.array(data[axis])
            avg = np.average(arr)
            data[axis] = (arr / avg).tolist()

        # Write the updated JSON object back to the file
        lines[i] = json.dumps(data)

    # Write the updated lines back to the file
    with open('data/2018/02/day=27/hour=17/05.jsonl', 'w') as file:
        file.write('\n'.join(lines))

# Function to process a JSONL file
def greaterAccel(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    for line in lines:
        data = json.loads(line)
        if any(x > 5 for x in data['x']):
            maxAccelInFile(filename)
            return
def findAccel():
    # Get a list of all files in the directory
    # Process each JSONL file
    high_accels = []
    highest_accel = 0
    highest_file = ""
    for dirpath, dirnames, filenames in os.walk('data'):
        # Process each JSONL file
        for filename in filenames:
            if filename.endswith('.jsonl'):
               maxInFile = maxAccelInFile(os.path.join(dirpath, filename))
               if maxInFile>highest_accel and maxInFile!=201.342:
                   highest_accel = maxInFile
                   highest_file = os.path.join(dirpath, filename)
               high_accels.append(maxInFile)
               
    return highest_accel, highest_file

def maxAccelInFile(filename: str):
    # Open the JSONL file
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Initialize the maximum value
    max_value = float('-inf')

    # Process each line
    for line in lines:
        data = json.loads(line)

        # Update the maximum value
        for axis in ['x', 'y', 'z']:
            max_value = max(max_value, max(data[axis]))
    return max_value

#Takes in file at pathname, takes sqrt of of squared x y and z accelarations at each timestep to get total ground accelaration
def addTotalAccelerationInfo(pathname:str):
    # Get a list of all files in the directory
    for dirpath, dirnames, filenames in os.walk(pathname):
        # Process each JSONL file
        for filename in filenames:
            if filename.endswith('.jsonl'):
                filepath = os.path.join(dirpath, filename)
                with open(filepath, 'r') as file:
                    lines = file.readlines()
                # Process each line
                for i, line in enumerate(lines):
                    data = json.loads(line)
                    # Calculate total acceleration
                    total_acceleration = np.sqrt((np.square(data['x'])) + (np.square(data['y'])) + (np.square(data['z']))).tolist()
                    # Add total acceleration info to the JSON object
                    data['total_acceleration'] = total_acceleration
                    # Write the updated JSON object back to the file
                    lines[i] = json.dumps(data)
                # Write the updated lines back to the file
                with open(filepath, 'w') as file:
                    file.write('\n'.join(lines))
#Take in a path, walk through all jsonl files in said path, and for each line if said line's 'total_accelaration'
# list of values contains a value greater than accel, add that to a new jsonl file in the root directory
def dataWithAccel(path:str, accel:float):
    # Get a list of all files in the directory
    for dirpath, dirnames, filenames in os.walk(path):
        # Process each JSONL file
        for filename in filenames:
            if filename.endswith('.jsonl'):
                filepath = os.path.join(dirpath, filename)
                with open(filepath, 'r') as file:
                    lines = file.readlines()
                # Process each line
                for line in lines:
                    data = json.loads(line)
                    # Check if 'total_acceleration' contains a value greater than 5
                    if any(x > accel for x in data['total_acceleration']):
                        # Write the line to a new JSONL file in the root directory
                        with open(f"over{str(accel)}.jsonl", 'a') as output_file:
                            output_file.write(line)

def deleteWithinFive(path:str):
    time_difs = []
    if path.endswith('.jsonl'):
        with open(path, 'r') as file:
            lines = file.readlines()
        # Get reference time value
        prev_time = json.loads(lines[0])['cloud_t']
        # Process each line
        for line in lines[1:]:
            data = json.loads(line)
            time = data['cloud_t']
            # Check if 'cloud_t' value is within 5 minutes (300 seconds)
            time_difs.append(time-prev_time)
            if (time - prev_time > 300):
                # Write the line to a new JSONL file in the root directory
                with open(f"{str(path)}_delete_within_5.jsonl", 'a') as output_file:
                    output_file.write(line)
            prev_time = time
    print(time_difs)


# dataWithAccel("data", 1.7)
deleteWithinFive("over1.7.jsonl")
