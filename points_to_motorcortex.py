import numpy as np
import os
from typing import Union
import csv
from scipy.optimize import minimize, lsq_linear
from random import random
from pyswarms import single
import argparse
import json
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import motorcortex
import math
import time
from robot_control.robot_command import RobotCommand


FIELDNAMES_OPTIONS = {
    "random" : ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'px_r', 'py_r', 'pz_r', 'rx_r', 'ry_r', 'rz_r'],
    "circles": ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'px_r', 'py_r', 'pz_r', 'joint'],
}

class MoveRobotService():
    def __init__(self):
        parameter_tree = motorcortex.ParameterTree()
        self.motorcortex_types = motorcortex.MessageTypes()
        license_file = "mcx.cert.crt"

        try:
            self.req, self.sub = motorcortex.connect('wss://192.168.2.100:5568:5567', self.motorcortex_types, parameter_tree,
                                                     certificate=license_file, timeout_ms=1000, login="admin", password="vectioneer")
                  
        except Exception as e:
            print(f"Failed to establish connection: {e}")
            return
        self.robot = RobotCommand(self.req, self.motorcortex_types)
        self.joint_subscription = self.sub.subscribe(
                ['root/ManipulatorControl/fkActualToolCoords/jointPositions'], 'group1', 1)

        if self.robot.engage():
            print('Robot is at Engage')
        else:
            print('Failed to set robot to Engage')
            return
        self.robot.reset()

    def move_robot(self, angles):
        print(angles)
        self.robot.moveToPoint(angles, 0.25, 1.0)
        joint_params = self.joint_subscription.read()
        joint_pos_value = joint_params[0].value
        return joint_pos_value
    

def write_dataset(dataset: np.ndarray, filename: str, fieldnames: list[str]):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in dataset:
                writer.writerow({name: row[index] for index, name in enumerate(fieldnames)})
        
def read_dataset(filename: str, fieldnames: list[str]) -> Union[np.ndarray, np.ndarray]:
    dataset = np.array([], dtype='float').reshape(0, len(fieldnames))
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dataset = np.concatenate((dataset, np.array([float(row[field]) for field in fieldnames]).reshape(1, -1)), axis=0)
    return dataset

def gather_data(dataset, robot: MoveRobotService):
    for index, row in enumerate(dataset):
        real_angles = np.array(robot.move_robot(row[:6].tolist()))
        print(real_angles)
        dataset[index, :6] = real_angles
        input()

def experiment(robot, dataset_file, fieldnames):
    dataset = read_dataset(dataset_file, fieldnames)
    gather_data(dataset, robot)
    write_dataset(dataset, dataset_file, fieldnames)

def main(args):
    robot = MoveRobotService()
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)

    experiment(robot, config["base_circles_dataset_file"], FIELDNAMES_OPTIONS["circles"])
    experiment(robot, config["tool_circles_dataset_file"], FIELDNAMES_OPTIONS["circles"])
    experiment(robot, config["dataset_file"], FIELDNAMES_OPTIONS["random"])
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Name of .json configuration file. Default: ar_5.json", default="ar_5.json")
    args = parser.parse_args()
    main(args)
