I am writing a model to help with earthquake predictions. This model is going to be based off of different data than traditional models, as this model will be taking in acceleration data in the X, Y, and Z directions. This data will be coupled with other information, such as time and location and compiled in a JSON file. This file needs to be parsed to extract and format the data correctly for the model to process the information and update its weights. 
My research assistant has taken the liberty to write the functions needed to preprocess the data into an acceptable format. I need you to annotate and explain the code to me to allow me to understand this code and present it to our research group. The file contains functions that convert the data into a working format and refactor them to be the appropriate type and shape for the model to input. 
Your task is to write docstrings for each of the functions in the class, capturing their intended purpose, the parameters it take in, and its expected output. You then need to annotate the `add_total_and_select` and `jsonl_to_data` functions with inline comments that explain the logic within the code. These comments should explain how the lines of code achieve their intended purpose and should use a formal tone as I am presenting this to a group of researchers. Finally, write a written explanation regarding how the input data should be structured, how the preprocess file extracts from the input data, and the shape and format of the output data.
```python
import json
import numpy as np
import os
import math
import argparse
import time
import datetime
import shutil
from random import shuffle

def accel_to_rich_one(accel):
    g = accel / 980.665
    mercalli_split = [.000464, .00175, .00297, .0276, .062, .115, .215, .401, .747, 1.39]
    ratios = g / next((mval for mval in mercalli_split if g < mval), mercalli_split[-1])
    mercalli_id = np.digitize(g, mercalli_split) + 1
    mercalli_richter = {1:1, 2:3, 3:3.5, 4:4, 5:4.5, 6:5, 7:5.5, 8:6, 9:6.5, 10:7, 11:7.5, 12:8}
    richter_val = mercalli_richter[mercalli_id]
    richter_val += ratios
    return richter_val

def jsonl_to_data(filename, start_time, end_time):
    times = []
    richters = []
    accels = []

    with open(f"{filename}", 'r') as file:
        lines = file.readlines()

    line0 = lines[0]
    json_data = json.loads(line0)
    t = json_data['cloud_t']-start_time
    times.append(t)
    richter = accel_to_rich_one(np.array(json_data["total_acceleration"]).max())
    richters.append(richter)
    accel_matrix = np.array([json_data['x'], json_data['y'], json_data['z']])
    accels.append(accel_matrix.flatten())

    for i in range(1, len(lines)):
        line2 = lines[i]
        line1 = lines[i - 1]
        json_data_2 = json.loads(line2)
        json_data_1= json.loads(line1)
        t = json_data_2['cloud_t']-json_data_1['cloud_t']
        times.append(t)
        richter = accel_to_rich_one(np.array(json_data_2["total_acceleration"]).max())
        richters.append(richter)
        accel_matrix = np.array([json_data_2['x'], json_data_2['y'], json_data_2['z']])
        accels.append(accel_matrix.flatten())
    times.append(end_time - json_data_2['cloud_t'])

    richter_avg = np.average(richters)
    average_accel = np.mean(accels)

    richters = [r - richter_avg for r in richters]
    accels = [a - average_accel for a in accels]

    return times, richters, accels

def sort_by_time(filename):
    with open(f"{filename}.jsonl", 'r+') as file:
        lines = file.readlines()
    sorted_lines = sorted(lines, key=lambda line: json.loads(line)['cloud_t'])
    with open(f"{filename}.jsonl", 'w') as file:
        file.writelines(sorted_lines)

def delete_within_x(filepath:str, num_secs:int):
    with open(f"{filepath}.jsonl", 'r+') as file:
        lines = file.readlines()
    prev_time = float('-inf')
    for line in lines:
        data = json.loads(line)
        curr_time = data['cloud_t']
        if curr_time - prev_time > (num_secs):
            with open(f"delete_within_{str(num_secs)}.jsonl", 'a') as output_file:
                output_file.write(line)
        prev_time = curr_time
    os.remove(f"{filepath}.jsonl")
    os.rename(f"delete_within_{str(num_secs)}.jsonl", f"{filepath}.jsonl")

def add_total_and_select(path:str, output:str, accel:float): 
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('.jsonl'):
                filepath = os.path.join(dirpath, filename)
                with open(filepath, 'r') as file:
                    lines = file.readlines()
                for line in lines:
                    data = json.loads(line)
                    total_acceleration = np.sqrt((np.square(data['x'])) + (np.square(data['y'])) + (np.square(data['z']))).tolist()
                    data['total_acceleration'] = total_acceleration
                    if any(x > accel for x in data['total_acceleration']):
                        line = json.dumps(data)
                        with open(f"{output}.jsonl", 'a') as output_file:
                            output_file.write(f"{line}\n")

def full_preprocess(path:str, output:str, accel:float, start_time: int, end_time: int):
    add_total_and_select(path, output, accel)
    sort_by_time(output)
    delete_within_x(output, 100)
```