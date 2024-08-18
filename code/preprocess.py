import json
import numpy as np
import os
import math
import argparse
import time
import datetime
import shutil
from random import shuffle

#SEE THAT THIS HAS BEEN CHANGED FROM THE ONE IN PREPROCESS: take in a single accelaration value, 
# get a single richter value
'''
NOTE: May want to change the ratio computation to more closely fit the log function 
'''
def accel_to_rich_one(accel):
    g = accel / 980.665
    mercalli_split = [.000464, .00175, .00297, .0276, .062, .115, .215, .401, .747, 1.39]
    ratios = g / next((mval for mval in mercalli_split if g < mval), mercalli_split[-1])
    mercalli_id = np.digitize(g, mercalli_split) + 1
    mercalli_richter = {1:1, 2:3, 3:3.5, 4:4, 5:4.5, 6:5, 7:5.5, 8:6, 9:6.5, 10:7, 11:7.5, 12:8}
    richter_val = mercalli_richter[mercalli_id]
    richter_val += ratios
    return richter_val

#Take a jsonl file, and for each line create data of the following format:
#data[1]=richter - avg, data[2]=accelaration matrix where the first row is the x accel, second row the y, 
#and third row the z, and where we are subtracting the average of all accelaration values
#  data[0] = log(t_i-t_{i-1})-(log_avg interval time)
def jsonl_to_data(filename, start_time, end_time):
    times = []
    richters = []
    accels = []

    with open(f"{filename}", 'r') as file:
        lines = file.readlines()

    # Process first line
    line0 = lines[0]
    json_data = json.loads(line0)
    # Append time, richter, and acceleration
    t = json_data['cloud_t']-start_time
    times.append(t)
    richter = accel_to_rich_one(np.array(json_data["total_acceleration"]).max())
    richters.append(richter)
    accel_matrix = np.array([json_data['x'], json_data['y'], json_data['z']])
    accels.append(accel_matrix.flatten())

    # Process the rest of the lines in pairs
    for i in range(1, len(lines)):
        line2 = lines[i]
        line1 = lines[i - 1]
        json_data_2 = json.loads(line2)
        json_data_1= json.loads(line1)
        # Append time, richter, and acceleration
        t = json_data_2['cloud_t']-json_data_1['cloud_t']
        times.append(t)
        richter = accel_to_rich_one(np.array(json_data_2["total_acceleration"]).max())
        richters.append(richter)
        accel_matrix = np.array([json_data_2['x'], json_data_2['y'], json_data_2['z']])
        accels.append(accel_matrix.flatten())
    times.append(end_time - json_data_2['cloud_t'])

    # Get average values after, and subtract them from relevant values
    richter_avg = np.average(richters)
    average_accel = np.mean(accels)

    richters = [r - richter_avg for r in richters]
    accels = [a - average_accel for a in accels]

    return times, richters, accels

#takes in a JSONL filename WIHTOUT suffix, sorts by "cloud_t" value
def sort_by_time(filename):
    with open(f"{filename}.jsonl", 'r+') as file:
        lines = file.readlines()
    sorted_lines = sorted(lines, key=lambda line: json.loads(line)['cloud_t'])
    with open(f"{filename}.jsonl", 'w') as file:
        file.writelines(sorted_lines)

#filepath does not include suffix, MODIFIES THE FILE
def delete_within_x(filepath:str, num_secs:int):
    with open(f"{filepath}.jsonl", 'r+') as file:
        lines = file.readlines()
    # Process each line
    prev_time = float('-inf')
    for line in lines:
        data = json.loads(line)
        curr_time = data['cloud_t']
        # Check if curr_time and prev_time are separated by num_mins minutes
        if curr_time - prev_time > (num_secs):
            # Write the line to a new JSONL file in the root directory
            with open(f"delete_within_{str(num_secs)}.jsonl", 'a') as output_file:
                output_file.write(line)
        prev_time = curr_time
    # Delete the original file
    os.remove(f"{filepath}.jsonl")
    # Rename the second file to be the first file
    os.rename(f"delete_within_{str(num_secs)}.jsonl", f"{filepath}.jsonl")

def add_total_and_select(path:str, output:str, accel:float): 
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
                    total_acceleration = np.sqrt((np.square(data['x'])) + (np.square(data['y'])) + (np.square(data['z']))).tolist()
                    # Add total acceleration info to the JSON object
                    data['total_acceleration'] = total_acceleration
                    # Check if 'total_acceleration' contains a value greater than 5
                    if any(x > accel for x in data['total_acceleration']):
                        # Write the updated JSON object back to the file
                        line = json.dumps(data)
                        # Write the line to a new JSONL file in the root directory
                        with open(f"{output}.jsonl", 'a') as output_file:
                            output_file.write(f"{line}\n")

# Takes in a path and preprocesses the data, returns the data
# IMPORTANT: the return format of this function is 3 lists
def full_preprocess(path:str, output:str, accel:float, start_time: int, end_time: int):
    add_total_and_select(path, output, accel)
    sort_by_time(output)
    delete_within_x(output, 100)

def get_year_unix_times(year:int) -> tuple:
    '''
    Get the Unix timestamps for the start and end of a given year

    Args: Year: int representing the year
    Returns: Tuple of Unix timestamps (start, end)
    '''
    # Start of the year (January 1st, 00:00:00)
    start_time = datetime.datetime(year, 1, 1, 0, 0, 0)
    # End of the year (December 31st, 23:59:59)
    end_time = datetime.datetime(year, 12, 31, 23, 59, 59)

    # Converting datetime objects to Unix timestamps
    start_unix = int(time.mktime(start_time.timetuple()))
    end_unix = int(time.mktime(end_time.timetuple()))

    return (start_unix, end_unix+1)

def shuffle_files(base_dir) -> None:
    '''
    This function shuffles the files in the training, testing, and validation folders

    args: base_dir: The base directory containing the training, testing, and validation folders
    '''
    # Paths to the subfolders
    training_path = os.path.join(base_dir, 'training')
    testing_path = os.path.join(base_dir, 'testing')
    validation_path = os.path.join(base_dir, 'validation')

    # Gather all files and their current paths
    all_files = []
    for path in [training_path, testing_path, validation_path]:
        for file in os.listdir(path):
            full_path = os.path.join(path, file)
            all_files.append((full_path, path))
    
    # Shuffle the list of all files
    shuffle(all_files)

    # Calculate how many files to place back into each folder
    num_train = len(os.listdir(training_path))
    num_test = len(os.listdir(testing_path))
    num_val = len(os.listdir(validation_path))

    # Function to redistribute files
    def redistribute_files(files, folder, count):
        for i in range(count):
            file_path, _ = files.pop(0)
            new_path = os.path.join(folder, os.path.basename(file_path))
            shutil.move(file_path, new_path)
    
    # Redistribute the files back into the original folders
    redistribute_files(all_files, training_path, num_train)
    redistribute_files(all_files, testing_path, num_test)
    redistribute_files(all_files, validation_path, num_val)

def main():
    parser = argparse.ArgumentParser(description='Preprocess data')
    parser.add_argument('path', type=str, help='Path to the directory containing JSONL files')
    parser.add_argument('output', type=str, help='Output file name')
    args = parser.parse_args()
    start_time, end_time = get_year_unix_times(2018)
    accel = 1.7
    full_preprocess(args.path, args.output, accel, start_time, end_time)

if __name__ == '__main__':
    main()
