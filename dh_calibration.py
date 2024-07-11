import numpy as np
from math import cos, sin, pi, sqrt, atan2, asin, log10
from typing import Union
import csv
from scipy.optimize import minimize, lsq_linear
from random import random
from pyswarms import single
import argparse
import json

np.set_printoptions(suppress=True, threshold=np.inf, precision=5)

RADIUS = 0.025
FIELDNAMES_OPTIONS = {
    "random" : ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'px_r', 'py_r', 'pz_r', 'rx_r', 'rx_y', 'rx_z'],
    # "sphere" : ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'px_n', 'py_n', 'pz_n'],
    # "runtime_sphere" : ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'px_n', 'py_n', 'pz_n'],
    # "random_base" : ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'px_r', 'py_r', 'pz_r', 'rx_r', 'rx_y', 'rx_z'],
    "circles_base": ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'px_r', 'py_r', 'pz_r', 'joint'],
}

TASK_SCALE = np.diag([1, 1, 1, 0, 0, 0])


def cross(a:np.ndarray,b:np.ndarray)->np.ndarray:
    return np.cross(a,b)

class HayatiModel:
    def __init__(self, config):
        self.dataset_type = config['calibration_method']
        self.optimization_method = config['optimization_method']
        self.fieldnames = FIELDNAMES_OPTIONS[self.dataset_type]
        self.file = config['dataset_file']
        self.results_file = config["results_file"]
         # DH params: [a, alpha, d/beta, theta_offset, parallel_axis]. Angle beta is used instead of d if axis is nearly parallel to the previous
        self.nominal_dh = config['nominal_dh']
        self.nominal_base_params = config['nominal_base_params']
        self.nominal_tool_params = config['nominal_tool_params']
        
        self.estimated_dh = self.nominal_dh.copy()
        self.estimated_base_params = self.nominal_base_params.copy()
        self.estimated_tool_params = self.nominal_tool_params.copy()
        
        self.real_dh = config['real_dh']
        self.real_base_params = config['real_base_params']
        self.real_tool_params = config['real_tool_params']

        if self.dataset_type == "runtime_sphere":
            self.angle_dist = config["angle_distribution_interval"]
            self.linear_dist = config["linear_distribution_interval"]
            self.base_dist = config["base_distribution_interval"]
            self.tool_dist = config["tool_distribution_interval"]
        else:
            self.angle_dist = self.linear_dist = self.base_dist = self.tool_dist = 0
        
        self.exclude_base_tool = config["exclude_base_tool"]
        self.initial_rotation_matrix = config["initial_rotation_matrix"]
        self.sphere_points = config["sphere_points"]
        self.joint_limits = config["joint_limits_actual"]
        self.joint_limits_circle = config["joint_limits_circle"]
        self.bypass_sphere_models = config["bypass_sphere_models"]
        self.samples_number = config["samples_number"]

        if self.dataset_type == "random" or self.dataset_type == "random_base":
            self.measurable_params_mask = np.array([0, 1, 2, 3, 4, 5], dtype='int')
        else:
            self.measurable_params_mask = np.array([0, 1, 2], dtype='int')

        self.identifiability_mask = np.ones(36, dtype='int')
        self.koef = 0.001
        self.lm_koef = 0.01
        self.prev_norm = 0
        self.num_point = 0

    def y_rot(self, angle: Union[int, float]) -> np.ndarray:
        mat = np.array([[cos(angle), 0, sin(angle), 0],
                        [0, 1, 0, 0],
                        [-sin(angle), 0, cos(angle), 0],
                        [0, 0, 0, 1]],dtype='float')
        return mat

    def init_metrics(self):
        self.cond = None
        self.norm = None
        self.max_x_error = 0
        self.max_y_error = 0
        self.max_z_error = 0
        self.max_dist_error = 0
        self.min_x_error = 999999
        self.min_y_error = 999999
        self.min_z_error = 999999
        self.min_dist_error = 999999

    def calculate_metrics(self, err):
        dist_error = np.linalg.norm(err[:3])
        if abs(err[0]) > self.max_x_error:
            self.max_x_error = abs(err[0])
        if abs(err[1]) > self.max_y_error:
            self.max_y_error = abs(err[1])
        if abs(err[2]) > self.max_z_error:
            self.max_z_error = abs(err[2])
        if dist_error > self.max_dist_error:
            self.max_dist_error = dist_error

        if abs(err[0]) < self.min_x_error:
            self.min_x_error = abs(err[0])
        if abs(err[1]) < self.min_y_error:
            self.min_y_error = abs(err[1])
        if abs(err[2]) < self.min_z_error:
            self.min_z_error = abs(err[2])
        if dist_error < self.min_dist_error:
            self.min_dist_error = dist_error
        
    def display_metrics(self):
        print(f"Norm: {self.norm}\n" +
              f"Cond: {self.cond}\n" +
              f"Lambda: {self.lm_koef}\n" +
              f"Identifiables: {self.identifiability_mask} ({self.identifiability_mask.sum()})\n" +
              f"Max x error: {self.max_x_error}\n" +
              f"Max y error: {self.max_y_error}\n" +
              f"Max z error: {self.max_z_error}\n" +
              f"Max dist error: {self.max_dist_error}\n" +
              f"Min x error: {self.min_x_error}\n" +
              f"Min y error: {self.min_y_error}\n" +
              f"Min z error: {self.min_z_error}\n" +
              f"Min dist error: {self.min_dist_error}\n" +
              f"Base: {self.estimated_base_params}\n" +
              f"Tool: {self.estimated_tool_params}\n")
        print("DH: ", *self.estimated_dh, sep='\n',)

    def test_estimated_params(self, samples=300):
        print(f"\nTesting estimated params on {samples} random samples\n")
        self.init_metrics()
        for _ in range(samples):
            angle_set = (np.random.rand(6) - 0.5) * 2 * pi
            estimated_position = self.fk(angle_set, 'estimated')[:3, 3]
            real_position = self.fk(angle_set, 'real')[:3, 3]
            self.calculate_metrics(real_position - estimated_position)
        self.display_metrics()
        input()
              
    def write_results(self, filename='results.json'):
        tfs = self.get_transforms([0, 0, 0, 0, 0, 0], self.estimated_dh)
        mcx_params = []
        offsets = []
        for index, tf in enumerate(tfs):
            offset = tf[:3, 3]
            rotation = self.extract_zyx_euler(tf)

            if index == 0:
                mcx_params.append([self.estimated_dh[0][0], 0, self.estimated_dh[0][2], 0, 0, pi/2-self.estimated_dh[0][1]])
            elif index == 1:
                mcx_params.append([0, 0, self.estimated_dh[1][0], self.estimated_dh[1][1], -self.estimated_dh[1][2], 0])
            elif index == 2:
                mcx_params.append([0, 0, self.estimated_dh[2][0], self.estimated_dh[2][1], -self.estimated_dh[2][2], 0])
            elif index == 3:
                mcx_params.append([-self.estimated_dh[3][0], self.estimated_dh[3][2], 0, 0, 0, -pi/2-self.estimated_dh[3][1]])
            elif index == 4:
                mcx_params.append([self.estimated_dh[4][0], 0, self.estimated_dh[4][2], 0, 0, -pi/2+self.estimated_dh[4][1]])
            elif index == 5:
                mcx_params.append([-self.estimated_dh[5][0], self.estimated_dh[5][2], 0, 0, 0, -self.estimated_dh[5][1]])
            offsets.append(rotation[2] - self.nominal_dh[index][3])

        res_dict = {"estimated_dh": self.estimated_dh, "estimated_base_params": self.estimated_base_params,
                    "estimated_tool_params": self.estimated_tool_params, "mcx_params": mcx_params, "offsets": offsets}
        with open(filename, 'w') as file:
            json.dump(res_dict, file)

    def init_params(self):
        vec = self.pack_param_vec(self.estimated_base_params, self.estimated_dh, self.estimated_tool_params)
        if self.exclude_base_tool:
            base_dist = 0
            tool_dist = 0
        else:
            base_dist = self.base_dist
            tool_dist = self.tool_dist
        inc = np.array([(random() - 0.5)*base_dist for _ in range(6)], dtype='float')
        for row in self.nominal_dh:
            if row[-1]:
                inc = np.concatenate((inc, np.array([(random() - 0.5)*self.linear_dist,
                                                     (random() - 0.5)*self.angle_dist,
                                                     (random() - 0.5)*self.linear_dist,
                                                     (random() - 0.5)*self.angle_dist], dtype='float')))
            else: 
                inc = np.concatenate((inc, np.array([(random() - 0.5)*self.linear_dist,
                                                     (random() - 0.5)*self.angle_dist,
                                                     (random() - 0.5)*self.angle_dist,
                                                     (random() - 0.5)*self.angle_dist], dtype='float')))
        inc = np.concatenate((inc, np.array([(random() - 0.5)*tool_dist for _ in range(6)], dtype='float')))
        self.estimated_base_params, self.estimated_dh, self.estimated_tool_params = self.unpack_param_vec(vec + inc)

        return vec + inc

    def z_rot(self, angle: Union[int, float]) -> np.ndarray:
        mat = np.array([[cos(angle), -sin(angle), 0, 0],
                        [sin(angle), cos(angle), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]],dtype='float')
        return mat

    def x_rot(self, angle: Union[int, float]) -> np.ndarray:
        mat = np.array([[1, 0, 0, 0],
                        [0, cos(angle), -sin(angle), 0],
                        [0, sin(angle), cos(angle), 0],
                        [0, 0, 0, 1]],dtype='float')
    
        return mat
    
    def trans(self, vector: np.ndarray) -> np.ndarray:
        mat = np.array([[1, 0, 0, vector[0]],
                        [0, 1, 0, vector[1]],
                        [0, 0, 1, vector[2]],
                        [0, 0, 0, 1]],dtype='float')
        return mat
    
    def euler_trans(self, euler, trans) -> np.ndarray:
        rx = self.x_rot(euler(0))
        ry = self.y_rot(euler(1))
        rz = self.z_rot(euler(2))

        R = rz @ ry @ rx
        R = R.tolist()
        R[0].append(trans[0])
        R[1].append(trans[1])
        R[2].append(trans[2])
        R.append([0,0,0,1])
        return np.array(R)
    
    def extract_zyx_euler(self, mat: np.ndarray) -> np.ndarray:
        sy = -mat[2, 0]
        cy = sqrt(1 - sy**2)
        return np.array([atan2(mat[2, 1]/cy, mat[2, 2]/cy) , asin(sy), atan2(mat[1, 0]/cy, mat[0, 0]/cy)], dtype='float')
    
    def dh_trans(self, params: Union[np.ndarray, list], angle: Union[int, float]) -> np.ndarray:
        a, alpha, d, theta_offtet, _ = params
        sa = sin(alpha)
        ca = cos(alpha)
        sq = sin(theta_offtet + angle)
        cq = cos(theta_offtet + angle)
        mat = np.array([[cq, -ca*sq,  sa*sq, a*cq],
                        [sq,  ca*cq, -sa*cq, a*sq],
                        [ 0,     sa,     ca,    d],
                        [ 0,      0,      0,    1]], dtype='float')
        return mat
    
    def hayati_trans(self, params: Union[np.ndarray, list], angle: Union[int, float]) -> np.ndarray:
        a, alpha, beta, theta_offtet, _ = params
        sa = sin(alpha)
        ca = cos(alpha)
        sb = sin(beta)
        cb = cos(beta)
        sq = sin(theta_offtet + angle)
        cq = cos(theta_offtet + angle)
        mat = np.array([[-sa*sb*sq+cb*cq, -ca*sq,  sa*cb*sq+sb*cq, a*cq],
                        [ sa*sb*cq+cb*sq,  ca*cq, -sa*cb*cq+sb*sq, a*sq],
                        [         -ca*sb,     sa,           ca*cb,    0],
                        [              0,      0,               0,    1]], dtype='float')
        return mat

    def skew(self, vector: Union[list, np.ndarray]) -> np.ndarray:
        return np.array([[0, -vector[2], vector[1]],
                        [vector[2], 0, -vector[0]],
                        [-vector[1], vector[0], 0]], dtype='float')

    
    def get_transforms(self, angles: Union[np.ndarray, list], params: list) -> list:
        tfs = []
        for index, unit in enumerate(params):
            if self.nominal_dh[index][-1] == 0:
                tfs.append(self.dh_trans(unit, angles[index]))
            elif self.nominal_dh[index][-1] == 1:
                tfs.append(self.hayati_trans(unit, angles[index]))
        return tfs
    
    def get_base_tool_tf(self, base_params, tool_params):
        main_tf = self.trans(base_params[:3]) @ self.z_rot(base_params[3]) @ self.y_rot(base_params[4]) @ self.x_rot(base_params[5])
        tool = self.trans(tool_params[:3]) @ self.z_rot(tool_params[3]) @ self.y_rot(tool_params[4]) @ self.x_rot(tool_params[5])
        return main_tf, tool

    def fk(self, angles: Union[np.ndarray, list], type: str) -> np.ndarray:
        if type == 'estimated':
            params = self.estimated_dh
            tool_params = self.estimated_tool_params
            base_params = self.estimated_base_params
        elif type == 'nominal':
            params = self.nominal_dh
            tool_params = self.nominal_tool_params
            base_params = self.nominal_base_params
        elif type == 'real':
            params = self.real_dh
            tool_params = self.real_tool_params
            base_params = self.real_base_params
        else:
            raise ValueError("type must be 'nominal', 'real' or 'estimated'")
        main_tf, tool = self.get_base_tool_tf(base_params, tool_params)

        tfs = self.get_transforms(angles, params)
        for tf in tfs:
            main_tf = main_tf @ tf
        return main_tf @ tool

    def numerical_ik(self, target_pose: np.ndarray, initial_guess: Union[np.ndarray, list], type: str) -> np.ndarray:
        def cost(x, target_pose):
            pose = self.fk(x, type)
            res = np.linalg.norm(pose - target_pose)
            return res

        result = minimize(cost, x0=initial_guess, args=target_pose, method='BFGS', options={"gtol":1e-17})
        return result.x, result.fun
    
    def generate_circles_dataset(self, samples):
        dataset = np.array([], dtype='float').reshape(0, len(self.fieldnames))
        inc1 = (self.joint_limits_circle[0] * 2)/samples
        inc2 = (self.joint_limits_circle[1] * 2)/samples

        angles = np.array([-self.joint_limits_circle[0], 0, 0, 0, 0, 0], dtype='float')
        for _ in range(samples):
            real_position = self.fk(angles, 'real')[:3, 3]
            dataset = np.concatenate((dataset, np.concatenate((angles, real_position, [1]), axis=0).reshape(1, len(self.fieldnames))), axis=0)
            angles[0] += inc1

        angles = np.array([0, -self.joint_limits_circle[1], 0, 0, 0, 0], dtype='float')
        for _ in range(samples):
            real_position = self.fk(angles, 'real')[:3, 3]
            dataset = np.concatenate((dataset, np.concatenate((angles, real_position, [2]), axis=0).reshape(1, len(self.fieldnames))), axis=0)
            angles[1] += inc2

        return dataset
    
    # Required for sphere calibration

    # def generate_angles(self, initial_pose, initial_angles, samples):
    #     angles = [initial_angles]
    #     pose = initial_pose
    #     for i in range(samples):
    #         pose = initial_pose @ self.z_rot(random()*pi) @ self.y_rot(random()*pi) @ self.x_rot(random()*pi)
    #         angle_set, val = self.numerical_ik(pose, angles[i - 1], 'estimated')
    #         if val < 1.0e-5:
    #             angles.append(angle_set)

    #     return angles

    # def generate_sphere_dataset(self, samples=300):
    #     self.init_params()
    #     dataset = np.array([], dtype='float').reshape(0, len(self.fieldnames))
    #     for point in self.sphere_points[self.num_point : self.num_point + 1]:
    #         pose = np.eye(4, 4)
    #         pose[0:3, 0:3] = self.initial_rotation_matrix
    #         pose[:3, 3] = np.array(point)

    #         iterations = 0
    #         val = 100
    #         while val > 1.0e-5 and iterations < 1000:
    #             angle_set, val = self.numerical_ik(pose, (np.random.rand(6) - 0.5) * 2 * pi, 'estimated')
    #             iterations += 1
    #         if iterations == 1000:
    #             print(f"Unable to find good solution for point {point}")
    #             break
    #         angles = self.generate_angles(pose, angle_set, samples)

    #         for angle_set in angles:
    #             dataset = np.concatenate((dataset, np.concatenate((angle_set, point)).reshape(1, len(self.fieldnames))), axis=0)

    #     return dataset

    

    # def direct_sphere_model(self, reference_position: np.ndarray, angles: np.ndarray) -> np.ndarray:
    #     x0, y0, z0 = reference_position
    #     xc, yc, zc = self.fk(angles, 'real')[:3, 3] - reference_position

    #     if self.bypass_sphere_models:
    #         return np.array([xc, yc, zc], dtype='float')

    #     b = -2*xc
    #     c = xc**2 + (y0 - yc)**2 + (z0 - zc)**2 - RADIUS**2
    #     x_det = (-b - sqrt(b**2 - 4*c))/2

    #     b = -2*yc
    #     c = yc**2 + (x0 - xc)**2 + (z0 - zc)**2 - RADIUS**2
    #     y_det = (-b - sqrt(b**2 - 4*c))/2

    #     b = -2*zc
    #     c = zc**2 + (x0 - xc)**2 + (y0 - yc)**2 - RADIUS**2
    #     z_det = (-b - sqrt(b**2 - 4*c))/2

    #     return -(np.array([x_det, y_det, z_det], dtype='float') + RADIUS)
    
    # def inverse_sphere_model(self, detector_readings: np.ndarray) -> np.ndarray:
    #     if self.bypass_sphere_models:
    #         return detector_readings

    #     xd, yd, zd = -detector_readings - RADIUS

    #     b1 = 0.5 * (xd**2 - zd**2) / xd
    #     a1 = zd / xd
    #     b2 = 0.5 * (yd**2 - zd**2) / yd
    #     a2 = zd / yd

    #     a = (a1**2 + a2**2 + 1)
    #     b = 2*(a1 * b1 + a2 * b2 - zd)
    #     c = b1**2 + b2**2 + zd**2 - RADIUS**2

    #     z = 0.5 * (-b + sqrt(b**2 - 4*a*c)) / a
    #     x = a1 * z + b1
    #     y = a2 * z + b2

    #     return np.array([x, y, z], dtype='float')
    
    def generate_dataset(self):
        if self.dataset_type == "random":
            self.optimal_random_dataset(self.samples_number)

        elif self.dataset_type == "circles_base":
            self.write_dataset(self.generate_circles_dataset(self.samples_number))


        elif self.dataset_type == "random_base":
            print("Dataset generation is not supported. Use 'random' type")
        else:
            self.write_dataset(self.generate_sphere_dataset(self.samples_number))

    def optimal_random_dataset(self, samples):
        best_value = 10**10
        prev_best_value = 10**11
        dataset = self.generate_random_dataset(samples)

        while prev_best_value - best_value > 0.00001:
            dataset, val = self.conf_plus(dataset)
            print(val)
            dataset, val = self.conf_minus(dataset)
            print(val)
            prev_best_value = best_value
            best_value = val
        self.write_dataset(dataset)

    def conf_plus(self, dataset):
        def cost(angle_set):
            real_pose = self.fk(angle_set, 'real')
            real_position = real_pose[:3, 3]
            real_orientation = self.extract_zyx_euler(real_pose)
            string = np.concatenate((angle_set, real_position, real_orientation))
            new_dataset = np.concatenate((dataset, string.reshape(1, -1)), axis=0)
            jac, _ = self.full_jac(new_dataset)
            jac, _ = self.remove_redundant_params(jac)
            cond = np.linalg.cond(jac)
            params_number = self.identifiability_mask.sum()
            return cond + 100/params_number
        result = minimize(cost, x0=[(random() - 0.5) * fact for fact in self.joint_limits], method='Nelder-Mead',
                          bounds=list(zip(-np.array(self.joint_limits), np.array(self.joint_limits))), options={"maxiter": 200})
        
        real_pose = self.fk(result.x, 'real')
        real_position = real_pose[:3, 3]
        real_orientation = self.extract_zyx_euler(real_pose)
        string = np.concatenate((result.x, real_position, real_orientation))
        new_dataset = np.concatenate((dataset, string.reshape(1, -1)), axis=0)

        return new_dataset, result.fun
    
    def conf_minus(self, dataset):
        index = 0
        min_cost = 100000
        for row in range(dataset.shape[0]):
            new_dataset = np.delete(dataset, row, axis=0)
            jac, _ = self.full_jac(new_dataset)
            jac, _ = self.remove_redundant_params(jac)
            cond = np.linalg.cond(jac)
            params_number = self.identifiability_mask.sum()
            cost = cond + 100/params_number
            if cost < min_cost:
                min_cost = cost
                index = row
        return np.delete(dataset, index, axis=0), min_cost

    def generate_random_dataset(self, samples):
        dataset = np.zeros((samples, len(self.fieldnames)))
        for i in range(samples):
            angle_set = np.array([(random() - 0.5) * fact for fact in self.joint_limits], dtype='float')
            real_pose = self.fk(angle_set, 'real')
            real_position = real_pose[:3, 3]
            real_orientation = self.extract_zyx_euler(real_pose)
            dataset[i] = np.concatenate((angle_set, real_position, real_orientation))
        return dataset

    def write_dataset(self, dataset: np.ndarray):
        with open(self.file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()
            for row in dataset:
                writer.writerow({name: row[index] for index, name in enumerate(self.fieldnames)})
        
    def read_dataset(self, file_name: str) -> Union[np.ndarray, np.ndarray]:
        dataset = np.array([], dtype='float').reshape(0, len(self.fieldnames))
        with open(file_name, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                dataset = np.concatenate((dataset, np.array([float(row[field]) for field in self.fieldnames]).reshape(1, len(self.fieldnames))), axis=0)
        return dataset
     
    def calibration_jacobian(self, angles: Union[np.ndarray, list], base: Union[np.ndarray, list],
                             dh: Union[np.ndarray, list], tool: Union[np.ndarray, list]) -> np.ndarray:
        tfs = self.get_transforms(angles, dh)
        main_tf, tool = self.get_base_tool_tf(base, tool)

        np_estimated_dh = np.array(dh)
        offsets = np_estimated_dh[:, 3]
        parallel_mask = np_estimated_dh[:, 4]

        frames_cnt = len(tfs) + 2

        from_base_trans = [main_tf.copy()] * frames_cnt
        from_tool_trans = [tool.copy()] * frames_cnt

        for i in range(len(tfs)):
            from_base_trans[i + 1] = from_base_trans[i] @ tfs[i]
            from_tool_trans[frames_cnt - 2 - i] = tfs[len(tfs) - 1 - i] @ from_tool_trans[frames_cnt - 1 - i]

        from_base_trans[-1] = from_base_trans[-2] @ tool
        from_tool_trans[0] = main_tf @ from_tool_trans[1]

        # Jac structure: [[tx_b, ty_b, tz_b, rx_b, ry_b, rz_b], [d_a, d_alpha, d_d/d_beta, d_theta_off] - len(tfs) times, [tx_t, ty_t, tz_t, rx_t, ry_t, rz_t]]
        # [tx_b, ty_b, tz_b, rx_b, ry_b, rz_b] always equals eye(6, 6)
        # [tx_t, ty_t, tz_t, rx_t, ry_t, rz_t] = / R  0 \ ,  R - rotation matrix of tool relative to base
        #                                        \ 0  R /
        # Vertical order: [tx, ty, tz, rx, ry, rz]
        
        jac = np.eye(6, len(tfs)*4 + 12)
        last_mat = np.zeros((6, 6))
        last_mat[:3, :3] = from_base_trans[-1][:3, :3]
        last_mat[3:6, 3:6] = from_base_trans[-1][:3, :3]
        jac[:, (6+(frames_cnt-2)*4):(12+(frames_cnt-2)*4)] = last_mat

        for i in range(1, len(tfs) + 1):
            jac_sub = np.zeros((6, 4))
            p_vec = from_base_trans[i - 1][0:3, 0:3] @ from_tool_trans[i][0:3, 3]
            p_vec_plus = from_base_trans[i][0:3, 0:3] @ from_tool_trans[i + 1][0:3, 3]
            xi_rot = (from_base_trans[i - 1][0:3, 0:3] @ self.z_rot(offsets[i - 1] + angles[i - 1])[0:3, 0:3])[0:3, 0]
            z_vec = from_base_trans[i - 1][0:3, 2]
            y_vec_plus = from_base_trans[i][0:3, 1]
            # a
            jac_sub[:3, 0] = xi_rot
            # alpha
            jac_sub[:3, 1] = cross(xi_rot, p_vec_plus)
            jac_sub[3:6, 1] = xi_rot
            # theta
            jac_sub[:3, 3] = cross(z_vec, p_vec)
            jac_sub[3:6, 3] = z_vec
            if parallel_mask[i - 1] == 0:
                # d
                jac_sub[:3, 2] = z_vec
            else:
                # beta
                jac_sub[:3, 2] = cross(y_vec_plus, p_vec_plus)
                jac_sub[3:6, 2] = y_vec_plus

            jac[:, (6+(i-1)*4):(10+(i-1)*4)] = jac_sub

        return jac[self.measurable_params_mask, :]

    def scaling_matrix(self, jac: np.ndarray) -> np.ndarray:
        diag = np.linalg.norm(jac, axis=0)
        diag[diag < 10**(-8)] = 1
        return np.linalg.inv(np.diag(diag))

    def remove_redundant_params(self, jac: np.ndarray) -> np.ndarray:
        self.identifiability_mask = np.ones(36, dtype='int')
        scale = self.scaling_matrix(jac)
        jac = jac @ scale
        idx = np.argwhere(np.all(jac[..., :] == 0, axis=0))

        if self.exclude_base_tool:
            idx = np.array(list(set([0, 1, 2, 3, 4, 5, -1, -2, -3, -4, -5, -6]).union(set(idx.flatten()))), dtype='int').reshape(-1, 1)
        
        jac = np.delete(jac, idx, axis=1)
        for index in idx.flatten():
            self.identifiability_mask[index] = 0

        if self.dataset_type == "random_base":
            self.identifiability_mask[:6] = 1
            self.identifiability_mask[6:] = 0

        cond = np.linalg.cond(jac)
        while cond > 100:
            dim = jac.shape[1]
            _, s, v = np.linalg.svd(jac)
            ind = np.argmax(np.abs(v[dim - 1, :]))
            self.identifiability_mask[np.cumsum(self.identifiability_mask) == ind + 1] = 0
            jac = np.delete(jac, ind, axis=1)
            cond = s[0]/s[dim - 1]
        self.cond = cond
        return jac, scale
    
    def pack_param_vec(self, base_i, dh_i, tool_i):
        base = np.array(base_i)
        tool = np.array(tool_i)
        dh = np.delete(np.array(dh_i), 4, 1)
        return np.concatenate((base, dh.flatten(), tool))

    def unpack_param_vec(self, vec):
        base = vec[:6].tolist()
        dh = np.reshape(vec[6:-6], (6, int((vec.size - 12) / 6 )))
        dh = np.concatenate((dh, np.array(self.nominal_dh)[:, 4].reshape(6, 1)), axis=1).tolist()
        tool = vec[-6:].tolist()
        return base, dh, tool

    def update_params(self, inc, scale):
        vec = self.pack_param_vec(self.estimated_base_params, self.estimated_dh, self.estimated_tool_params)
        mapped_inc = inc[np.cumsum(self.identifiability_mask) - 1]
        mapped_inc[self.identifiability_mask == 0] = 0
        mapped_inc = np.linalg.inv(scale) @ mapped_inc
        vec += mapped_inc * self.koef
        self.estimated_base_params, self.estimated_dh, self.estimated_tool_params = self.unpack_param_vec(vec)

    def full_jac(self, dataset):
        self.init_metrics()

        # Traditional calibration, works in most cases

        if self.dataset_type == 'random':
            jac = np.array([], dtype='float').reshape(0, 36)
            error_vec = np.array([], dtype='float')
            for row in dataset:
                jac = np.concatenate((jac, TASK_SCALE @ self.calibration_jacobian(row[:6], self.estimated_base_params,
                                                                                  self.estimated_dh, self.estimated_tool_params)), axis=0)
                estimated_pose = self.fk(row[:6], 'estimated')
                estimated_coordinates = np.concatenate((estimated_pose[:3, 3], self.extract_zyx_euler(estimated_pose)))
                real_coordinates = row[6:]
                cur_err = real_coordinates[self.measurable_params_mask] - estimated_coordinates[self.measurable_params_mask]
                self.calculate_metrics(cur_err)                              
                error_vec = np.concatenate((error_vec, TASK_SCALE @ cur_err), axis=0)
            return jac, error_vec
        
        # Questionable performance, but in theory might work. Very tedious and impractical due to lengthy measurement process
                
        elif self.dataset_type == 'sphere':
            jac = np.array([], dtype='float').reshape(0, 36)
            error_vec = np.array([], dtype='float')
            for row in dataset:
                jac = np.concatenate((jac,self.calibration_jacobian(row[:6], self.estimated_base_params,
                                                                    self.estimated_dh, self.estimated_tool_params)), axis=0)
                cur_err = self.inverse_sphere_model(self.direct_sphere_model(row[6:], row[:6]))
                self.calculate_metrics(cur_err)
                error_vec = np.concatenate((error_vec, cur_err))

            return jac, error_vec
        
        # Same as above, but does not work at all
        
        elif self.dataset_type == 'runtime_sphere':
            error_vec = np.array([], dtype='float')
            for row in dataset:
                cur_err = self.inverse_sphere_model(self.direct_sphere_model(row[6:], row[:6]))
                self.calculate_metrics(cur_err)
                error_vec = np.concatenate((error_vec, cur_err))
            return error_vec

    def end_of_cycle_action(self):
        val = input()
        try:
            new_val = float(val)
        except ValueError:
            if val == 'exit':
                self.write_results(self.results_file)
                return True
            if val == 'test':
                self.test_estimated_params()
            if val == "point":
                self.num_point = int(input())
        else:
            self.koef = new_val
        return False
    
    def optimize(self):
        if (self.dataset_type == "random" or self.dataset_type == "sphere" or self.dataset_type == "random_base") and self.optimization_method == "gauss_newton":
            self.gauss_newton_ls()
        elif (self.dataset_type == "random" or self.dataset_type == "sphere" or self.dataset_type == "random_base") and self.optimization_method == "levenberg_marquardt":
            self.levenberg_marquardt()
        elif self.dataset_type == "runtime_sphere" and self.optimization_method == "pso":
            self.pso_runtime_method()
        elif self.dataset_type == "runtime_sphere" and self.optimization_method == "nelder_mead":
            self.runtime_optimize()
        else:
            print("Unknown method")


    # Performs poorly, but in theory better than nothing
            
    # def optimize_base(self):
    #     dataset = self.read_dataset(self.file)
    #     def cost(x):
    #         self.init_metrics()
    #         error_vec = np.array([], dtype='float')
    #         self.estimated_base_params = x
    #         for row in dataset:
    #             estimated_pose = self.fk(row[:6], 'estimated')
    #             estimated_coordinates = np.concatenate((estimated_pose[:3, 3], self.extract_zyx_euler(estimated_pose)))
    #             real_coordinates = row[6:]
    #             cur_err = real_coordinates[self.measurable_params_mask] - estimated_coordinates[self.measurable_params_mask]
    #             self.calculate_metrics(cur_err)                              
    #             error_vec = np.concatenate((error_vec, cur_err), axis=0)
    #         self.norm = np.linalg.norm(error_vec)
    #         self.display_metrics()
    #         return self.norm
    #     result = minimize(cost, x0=np.array(self.nominal_base_params), method='Nelder-Mead')
    #     print(result)
    #     return result.x, result.fun


    def gauss_newton_ls(self):
        dataset = self.read_dataset(self.file)
        self.init_params()

        while True:
            if self.dataset_type == 'sphere':
                dataset = self.generate_sphere_dataset(100)

            jac, error_vec = self.full_jac(dataset)
            self.norm = np.linalg.norm(error_vec)
            new_jac, scale = self.remove_redundant_params(jac)
            # inc = np.linalg.pinv(new_jac) @ error_vec
            solution = lsq_linear(new_jac, error_vec)
            self.update_params(solution.x, scale)
            self.display_metrics()
            if self.end_of_cycle_action():
                break

    def levenberg_marquardt(self):
        dataset = self.read_dataset(self.file)
        self.init_params()

        cost = 100000
        while cost > 0.001:
            if self.dataset_type == 'sphere':
                dataset = self.generate_sphere_dataset(100)
            jac, error_vec = self.full_jac(dataset)

            self.norm = np.linalg.norm(error_vec)
            if self.norm > self.prev_norm:
                self.lm_koef *= 10
            else: self.lm_koef /= 10

            self.prev_norm = self.norm
            
            new_jac, scale = self.remove_redundant_params(jac)
            inc = np.linalg.inv((new_jac.T @ new_jac + self.lm_koef * np.eye(new_jac.shape[1], new_jac.shape[1]))) @ new_jac.T @ error_vec
            # print(inc)
            self.update_params(inc, scale)
            self.display_metrics()
            if self.end_of_cycle_action():
                break
    

    # Doesn't work at all, but leave it here in case someone wants to try it or improve it

    # def set_bounds(self):
    #     upper_bound = np.array([], dtype='float')
    #     lower_bound = np.array([], dtype='float')
    #     for row in self.nominal_dh:
    #         if row[-1]:
    #             upper_bound = np.concatenate((upper_bound, np.array([row[0] + self.linear_dist, row[1] + self.angle_dist,
    #                                                                        row[2] + self.linear_dist, row[3] + self.angle_dist,], dtype='float')))
    #             lower_bound = np.concatenate((lower_bound, np.array([row[0] - self.linear_dist, row[1] - self.angle_dist,
    #                                                                        row[2] - self.linear_dist, row[3] - self.angle_dist,], dtype='float')))
    #         else:
    #             upper_bound = np.concatenate((upper_bound, np.array([row[0] + self.linear_dist, row[1] + self.angle_dist,
    #                                                                        row[2] + self.angle_dist, row[3] + self.angle_dist,], dtype='float')))
    #             lower_bound = np.concatenate((lower_bound, np.array([row[0] - self.linear_dist, row[1] - self.angle_dist,
    #                                                                        row[2] - self.angle_dist, row[3] - self.angle_dist,], dtype='float')))
                
    #     if not self.exclude_base_tool:
    #         upper_bound = np.concatenate((upper_bound, np.array([val + self.tool_dist for val in self.nominal_tool_params], dtype='float')))
    #         lower_bound = np.concatenate((lower_bound, np.array([val - self.tool_dist for val in self.nominal_tool_params], dtype='float')))
        
    #     return (upper_bound, lower_bound)

    # def pso_runtime_method(self):
    #     options = {'c1': 0.5, 'c2': 0.3, 'w':0.9, 'k': 3, 'p': 2}
    #     bounds = self.set_bounds()
    #     optimizer = single.local_best.LocalBestPSO(n_particles=20, dimensions=bounds[0].size, options=options, bounds=bounds)
    #     optimizer.optimize(self.pso_cost, 200)

    # def pso_cost(self, args):
    #     res = np.zeros(args.shape[0])
    #     for index, particle in enumerate(args):
    #         res[index] = self.runtime_cost(particle)
    #     return res
    
    # def runtime_cost(self, params):
    #     params = np.concatenate((np.zeros(6), params))
    #     if self.exclude_base_tool:
    #         params = np.concatenate((params, np.zeros(6)))
    #     self.estimated_base_params, self.estimated_dh, self.estimated_tool_params = self.unpack_param_vec(params)
    #     batch = self.generate_sphere_dataset(50)
    #     self.norm = np.linalg.norm(self.full_jac(batch))
    #     self.display_metrics()
    #     return self.norm
    
    # def runtime_optimize(self):
    #     bounds = self.set_bounds()
    #     result = minimize(self.runtime_cost, x0=self.init_params()[6:-6], method='Nelder-Mead',
    #                       bounds=list(zip(bounds[1], bounds[0])), options={"disp": True, "fatol": 1e-17, "adaptive": True})
                                                                                                  
    #     print(result)
        
def main(args):
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)
    model = HayatiModel(config)
    if args.generate:
        model.generate_dataset()
    model.optimize()

    # model.optimize_base()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Name of .json configuration file. Default: ar_20.json", default="ar_20.json")
    parser.add_argument("-g", "--generate", help="Generate dataset for selected method. Default: false", type=bool, default=False)
    args = parser.parse_args()
    main(args)
