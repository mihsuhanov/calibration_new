import numpy as np

from math import cos, sin, pi, sqrt, atan2
from typing import Union
import random
import csv
from scipy.optimize import minimize
import json
import time
from multipledispatch import dispatch

NOMINAL_BASE = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype="float")

NOMINAL_TOOL = np.array([[1, 0,  0,  0   ],
                         [0, 0, -1, -0.26],
                         [0, 1,  0,  1.84],
                         [0, 0,  0,  1   ]], dtype="float")

NOMINAL_AXES = np.array([[0, 0, 1, 0,  0,     0.19],
                         [0, 1, 0, 0,  0,     0.19],
                         [0, 1, 0, 0,  0,     0.99],
                         [0, 1, 0, 0,  0,     1.71],
                         [0, 0, 1, 0, -0.191, 1.71],
                         [0, 1, 0, 0, -0.191, 1.84]],dtype='float')

REAL_TOOL = np.array([[1, 0,  0,  0   ],
                      [0, 0, -1, -0.26],
                      [0, 1,  0,  1.84],
                      [0, 0,  0,  1   ]], dtype="float")

REAL_AXES = np.array([[0, 0, 1, 0,  0,     0.19],
                      [0, 1, 0, 0,  0,     0.19],
                      [0, 1, 0, 0,  0,     0.99],
                      [0, 1, 0, 0,  0,     1.71],
                      [0, 0, 1, 0, -0.191, 1.71],
                      [0, 1, 0, 0, -0.191, 1.84]],dtype='float')

SENSOR_OFFSETS = np.array([-0.01, 0.01, -0.01, -0.01, 0.01, 0.01],dtype='float')

RADIUS = 0.03


class ZeroReferenceModel:
    def __init__(self):
        self.angles = np.zeros(6)


        self.linear_params = []
        self.angular_params = []
    


    def sphere_model(self):
        # for 
        pass








        
    def calibration_jacobian(self):
        pass


    def generate_angles(self, inc_rx, inc_ry, inc_rz, initial_pose, initial_angles, samples):
        self.angles = [initial_angles]
        pose = initial_pose

        for i in range(samples):
            pose = pose @ z_rot(inc_rz) @ y_rot(inc_rx) @ x_rot(inc_ry)
            self.angles.append(self.numerical_ik(pose, self.angles[i - 1]))

        self.real_start_pose = full_transform(self.angles[0], 'real')


    def numerical_ik(self, target_pose, initial_guess):
        def cost(x, target_pose):
            pose = full_transform(x, 'nominal')
            res = np.linalg.norm(pose - target_pose)
            return res

        result = minimize(cost, x0=initial_guess, args=target_pose, method='BFGS', options={"gtol":1e-17})
        print(result.fun)
        return result.x



    
def y_rot(angle: float) -> np.ndarray:
    mat = np.array([[cos(angle), 0, sin(angle), 0],
                    [0, 1, 0, 0],
                    [-sin(angle), 0, cos(angle), 0],
                    [0, 0, 0, 1]],dtype='float')
    return mat

def z_rot(angle: float) -> np.ndarray:
    mat = np.array([[cos(angle), -sin(angle), 0, 0],
                    [sin(angle), cos(angle), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]],dtype='float')
    return mat

def x_rot(angle: float) -> np.ndarray:
    mat = np.array([[1, 0, 0, 0],
                    [0, cos(angle), -sin(angle), 0],
                    [0, sin(angle), cos(angle), 0],
                    [0, 0, 0, 1]],dtype='float')
    
    return mat

def trans(vector: np.ndarray) -> np.ndarray:
    mat = np.array([[1, 0, 0, vector[0]],
                    [0, 1, 0, vector[1]],
                    [0, 0, 1, vector[2]],
                    [0, 0, 0, 1]],dtype='float')

    return mat

def arbitrary_axis_rot(axis: np.ndarray, angle: float) -> np.ndarray:
    nu = 1 - cos(angle)
    x, y, z = axis
    mat = np.array([[cos(angle)+nu*x**2,  nu*x*y-sin(angle)*z, nu*x*z+sin(angle)*y],
                    [nu*y*x+sin(angle)*z, cos(angle)+nu*y**2,  nu*y*z-sin(angle)*x],
                    [nu*z*x-sin(angle)*y, nu*z*y+sin(angle)*x, cos(angle)+nu*z**2]], dtype='float')
    return mat

@dispatch(np.ndarray, np.ndarray, (float, int))
def screw_transform(axis: np.ndarray, point: np.ndarray, angle: Union[float, int]) -> np.ndarray:
    rot = arbitrary_axis_rot(axis, angle)
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = (np.eye(3) - rot) @ point
    return transform

@dispatch(np.ndarray, (float, int))
def screw_transform(axis: np.ndarray, angle: Union[float, int]) -> np.ndarray:
    return screw_transform(axis[0:3], axis[3:6],  angle)

def full_transform(angles: Union[np.ndarray, list], type: str) -> np.ndarray:
    tf = NOMINAL_BASE
    if type == 'real':
        angles = angles + SENSOR_OFFSETS
        axes = REAL_AXES
        tool = REAL_TOOL
    elif type == 'nominal':
        axes = NOMINAL_AXES
        tool = NOMINAL_TOOL
    else:
        raise ValueError("type must be 'nominal' or 'real'")
    
    for i in range(len(angles)):
        tf = tf @ screw_transform(axes[i], angles[i])
    return tf @ tool
    

def read_dataset(file_name: str) -> Union[np.ndarray, np.ndarray]:
    js = []
    points = []
    points_fk = []
    with open(file_name, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            js.append([float(row['q1']), float(row['q2']), float(
                row['q3']), float(row['q4']), float(row['q5']), float(row['q6'])])
            points.append(
                [float(row['px'])*1000, float(row['py'])*1000, float(row['pz'])*1000])
            try:
                points_fk.append([float(row['fk_x'])*1000, float(
                    row['fk_y'])*1000, float(row['fk_z'])*1000])
            except:
                pass




def main():
    model = ZeroReferenceModel()
    target_pose = np.array([[1, 0,  0,  0.5],
                            [0, 0, -1,  0],
                            [0, 1,  0,  1.6],
                            [0, 0,  0,  1   ]], dtype="float")
    
    first_result = model.numerical_ik(target_pose, [1,0,1,0,0,0])
    print(first_result)
    print(full_transform(first_result, 'nominal'))


    # model.generate_angles(0.1, 0.1, 0.1, target_pose, first_result, 60)
    # print(model.angles)




if __name__ == "__main__":
    main()


