import numpy as np
from math import cos, sin, pi, sqrt, atan2, asin, log10, acos, copysign
from typing import Union
import csv
from scipy.optimize import minimize, lsq_linear
from random import random
from pyswarms import single
import argparse
import json
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True, threshold=np.inf, precision=5)

FIELDNAMES_OPTIONS = {
    "random" : ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'px_r', 'py_r', 'pz_r', 'rx_r', 'ry_r', 'rz_r'],
    "circles": ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'px_r', 'py_r', 'pz_r', 'joint'],
}

TASK_SCALE = np.diag([1, 1, 1, 0, 0, 0])
BIG_NUMBER = 10e300


def cross(a:np.ndarray,b:np.ndarray)->np.ndarray:
    return np.cross(a,b)

class HayatiModel:
    def __init__(self, config):
        self.optimization_method = config['optimization_method']
        self.dataset_file = config['dataset_file']
        self.base_circles_dataset_file = config['base_circles_dataset_file']
        self.tool_circles_dataset_file = config['tool_circles_dataset_file']
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

        self.angle_dist = self.linear_dist = self.base_dist = self.tool_dist = 0
        
        self.joint_limits_general_h = config["joint_limits_general_h"]
        self.joint_limits_general_l = config["joint_limits_general_l"]

        self.joint_limits_circle_h = config["joint_limits_circle_h"]
        self.joint_limits_circle_l = config["joint_limits_circle_l"]

        self.cartesian_limits = config["cartesian_limits"]
        self.max_z_angle = config["max_z_angle"]
        
        self.general_samples_number = config["general_samples_number"]
        self.circle_samples_number = config["circle_samples_number"]

        self.zero_tracker_position = config["zero_tracker_position"]

        self.measurable_params_mask = np.array([0, 1, 2, 3, 4, 5], dtype='int')

        self.identifiability_mask = np.ones(36, dtype='int')
        self.koef = 0.001
        self.lm_koef = 0.01
        self.norm = 10
        self.prev_norm = 0
        self.num_point = 0

    # Metrics and validation routines
    def init_metrics(self):
        self.cond = None
        self.norm = None
        self.max_x_error = self.max_y_error = self.max_z_error = self.max_dist_error = 0
        self.min_x_error = self.min_y_error = self.min_z_error = self.min_dist_error = BIG_NUMBER

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
        backup_norm = self.norm
        print(f"\nTesting estimated params on {samples} random samples\n")
        self.init_metrics()
        dataset = self.generate_random_dataset(samples, disable_limits=True)
        for row in dataset:
            estimated_position = self.fk(row[:6], 'estimated')[:3, 3]
            self.calculate_metrics(row[6:9] - estimated_position)
        self.display_metrics()
        self.norm = backup_norm
        input()

    def draw_plot(self, dataset):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for row in dataset:
            estimated_pos = self.fk(row[:6], 'estimated')[:3, 3]
            ax.scatter(estimated_pos[0], estimated_pos[1], estimated_pos[2], marker="o")
            ax.scatter(row[6], row[7], row[8], marker="x")

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()

    # Math and kinematic modeling routines
    def x_rot(self, angle: Union[int, float]) -> np.ndarray:
        mat = np.array([[1, 0, 0, 0],
                        [0, cos(angle), -sin(angle), 0],
                        [0, sin(angle), cos(angle), 0],
                        [0, 0, 0, 1]],dtype='float')
    
        return mat
    
    def y_rot(self, angle: Union[int, float]) -> np.ndarray:
        mat = np.array([[cos(angle), 0, sin(angle), 0],
                        [0, 1, 0, 0],
                        [-sin(angle), 0, cos(angle), 0],
                        [0, 0, 0, 1]],dtype='float')
        return mat
    
    def z_rot(self, angle: Union[int, float]) -> np.ndarray:
        mat = np.array([[cos(angle), -sin(angle), 0, 0],
                        [sin(angle), cos(angle), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]],dtype='float')
        return mat
    
    def arbitrary_axis_rot(self, axis: np.ndarray, angle: float) -> np.ndarray:
        nu = 1 - cos(angle)
        x, y, z = axis
        mat = np.array([[cos(angle)+(nu*(x**2)),  (nu*x*y)-(sin(angle)*z), (nu*x*z)+(sin(angle)*y)],
                        [(nu*y*x)+(sin(angle)*z), cos(angle)+(nu*(y**2)),  (nu*y*z)-(sin(angle)*x)],
                        [(nu*z*x)-(sin(angle)*y), (nu*z*y)+(sin(angle)*x), cos(angle)+(nu*(z**2))]], dtype='float')
        return mat
    
    def trans(self, vector: np.ndarray) -> np.ndarray:
        mat = np.array([[1, 0, 0, vector[0]],
                        [0, 1, 0, vector[1]],
                        [0, 0, 1, vector[2]],
                        [0, 0, 0, 1]],dtype='float')
        return mat
    
    def extract_zyx_euler(self, mat: np.ndarray) -> np.ndarray:
        r = Rotation.from_matrix(mat)
        return r.as_euler('ZYX')
    
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
    
    # Dataset routines
    def generate_dataset(self):
        print("Generating main dataset...")
        self.write_dataset(self.optimal_random_dataset(self.general_samples_number), self.dataset_file, FIELDNAMES_OPTIONS["random"])
        print("Generating circles datasets...")
        self.write_dataset(self.generate_base_circles_dataset(self.circle_samples_number), self.base_circles_dataset_file, FIELDNAMES_OPTIONS["circles"])
        self.write_dataset(self.generate_tool_circles_dataset(self.circle_samples_number), self.tool_circles_dataset_file, FIELDNAMES_OPTIONS["circles"])
        print('done')

    def make_circle(self, samples, axis, initial_position):
        dataset = np.array([], dtype='float').reshape(0, len(FIELDNAMES_OPTIONS["circles"]))
        inc1 = (self.joint_limits_circle_h[axis - 1] - self.joint_limits_circle_l[axis - 1])/samples
        angles = initial_position
        angles[axis - 1] = self.joint_limits_circle_l[axis - 1]
        for _ in range(samples):
            real_position = self.fk(angles, 'real')[:3, 3]
            dataset = np.concatenate((dataset, np.concatenate((angles, real_position, [axis]), axis=0).reshape(1, -1)), axis=0)
            angles[axis - 1] += inc1
        
        return dataset
    
    def generate_base_circles_dataset(self, samples):
        return np.concatenate((self.make_circle(samples, 1, [0, 0, 1.57, 0, -1.57, 0]),
                               self.make_circle(samples, 2, [0, 0, 0, 0, 0, 0])), axis=0)
    
    def generate_tool_circles_dataset(self, samples):
        return np.concatenate((self.make_circle(samples, 5, [0, 0, 1.57, 0, 0, 0]),
                               self.make_circle(samples, 6, [0, 0, 1.57, 0, -1.57, 0])), axis=0)
    
    def satisfies_cartesian_limits(self, pose):
        position = pose[:3, 3]
        z_angle = acos(pose[:3, 2] @ [0, 0, 1])
        return (position[0] > self.cartesian_limits[0][0] and position[0] < self.cartesian_limits[0][1] and \
                position[1] > self.cartesian_limits[1][0] and position[1] < self.cartesian_limits[1][1] and \
                position[2] > self.cartesian_limits[2][0] and position[2] < self.cartesian_limits[2][1] and \
                abs(z_angle) < self.max_z_angle)
    
    def optimal_random_dataset(self, samples):
        backup_base = self.nominal_base_params.copy()
        self.nominal_base_params = np.zeros(6)

        best_value = BIG_NUMBER
        prev_best_value = BIG_NUMBER * 10
        dataset = self.generate_random_dataset(samples)

        while prev_best_value - best_value > 0.01:
            dataset, val = self.conf_plus(dataset)
            print(val)
            dataset, val = self.conf_minus(dataset)
            print(val)
            prev_best_value = best_value
            best_value = val

        self.nominal_base_params = backup_base
        return dataset

    def conf_plus(self, dataset):
        def cost(angle_set):
            nominal_pose = self.fk(angle_set, 'nominal')
            if self.satisfies_cartesian_limits(nominal_pose):
                nominal_position = nominal_pose[:3, 3]
                nominal_orientation = self.extract_zyx_euler(nominal_pose[:3, :3])
                string = np.concatenate((angle_set, nominal_position, nominal_orientation))
                new_dataset = np.concatenate((dataset, string.reshape(1, -1)), axis=0)
                jac, _ = self.full_jac(new_dataset)
                jac, _ = self.remove_redundant_params(jac)
                cond = np.linalg.cond(jac)
                params_number = self.identifiability_mask.sum()
                return cond + 100/params_number
            return BIG_NUMBER
        
        result = minimize(cost, x0=dataset[0, :6], method='Nelder-Mead',
                          bounds=list(zip(np.array(self.joint_limits_general_l), np.array(self.joint_limits_general_h))), options={"maxiter": 150})
        
        real_pose = self.fk(result.x, 'real')
        real_position = real_pose[:3, 3]
        real_orientation = self.extract_zyx_euler(real_pose[:3, :3])
        string = np.concatenate((result.x, real_position, real_orientation))
        new_dataset = np.concatenate((dataset, string.reshape(1, -1)), axis=0)

        return new_dataset, result.fun
    
    def conf_minus(self, dataset):
        index = 0
        min_cost = BIG_NUMBER
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

    def generate_random_dataset(self, samples, disable_limits=False):
        dataset = np.zeros((samples, len(FIELDNAMES_OPTIONS["random"])))
        for i in range(samples):
            dataset[i] = self.make_random_sample(disable_limits)
        return dataset
    
    def make_random_sample(self, disable_limits):
        while True:
            angle_set = np.array([self.joint_limits_general_l[axis] + random() * (self.joint_limits_general_h[axis] - self.joint_limits_general_l[axis]) for axis in range(6)], dtype='float')
            nominal_pose = self.fk(angle_set, 'nominal')
            if self.satisfies_cartesian_limits(nominal_pose) or disable_limits:
                real_pose = self.fk(angle_set, 'real')
                real_position = real_pose[:3, 3]
                real_orientation = self.extract_zyx_euler(real_pose[:3, :3])
                return np.concatenate((angle_set, real_position, real_orientation), axis=0)

    def write_dataset(self, dataset: np.ndarray, filename: str, fieldnames: list[str]):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in dataset:
                writer.writerow({name: row[index] for index, name in enumerate(fieldnames)})
        
    def read_dataset(self, filename: str, fieldnames: list[str]) -> Union[np.ndarray, np.ndarray]:
        dataset = np.array([], dtype='float').reshape(0, len(fieldnames))
        with open(filename, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                dataset = np.concatenate((dataset, np.array([float(row[field]) for field in fieldnames]).reshape(1, -1)), axis=0)
        return dataset
              
    # TODO: make some normal conversion here
    def write_results(self, filename='results.json'):
        tfs = self.get_transforms([0, 0, 0, 0, 0, 0], self.estimated_dh)
        mcx_params = []
        offsets = []
        for index, tf in enumerate(tfs):
            offset = tf[:3, 3]
            rotation = self.extract_zyx_euler(tf[:3, :3])

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

    # Optimization routines
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
            jac_sub[3:6, 1] = np.flip(xi_rot)
            # theta
            jac_sub[:3, 3] = cross(z_vec, p_vec)
            jac_sub[3:6, 3] = np.flip(z_vec)

            if parallel_mask[i - 1] == 0:
                # d
                jac_sub[:3, 2] = z_vec
            else:
                # beta
                jac_sub[:3, 2] = cross(y_vec_plus, p_vec_plus)
                jac_sub[3:6, 2] = np.flip(y_vec_plus)

            jac[:, (6+(i-1)*4):(10+(i-1)*4)] = jac_sub

        return jac[self.measurable_params_mask, :]

    def scaling_matrix(self, jac: np.ndarray) -> np.ndarray:
        diag = np.linalg.norm(jac, axis=0)
        diag[diag < 10**(-8)] = 1
        return np.linalg.inv(np.diag(diag))

    def remove_redundant_params(self, jac: np.ndarray, base_only=False, offsets_only=False) -> np.ndarray:
        if base_only:
            self.identifiability_mask = np.zeros(36, dtype='int')
            self.identifiability_mask[:3] = 1
            return jac[:, :3], np.eye(36, 36)
        
        if offsets_only:
            self.identifiability_mask = np.zeros(36, dtype='int')
            ind = [9, 13, 17, 21, 25, 29]
            self.identifiability_mask[ind] = 1
            return jac[:, ind], np.eye(36, 36)

        self.identifiability_mask = np.ones(36, dtype='int')
        scale = self.scaling_matrix(jac)
        jac = jac @ scale
        idx = np.argwhere(np.all(jac[..., :] == 0, axis=0))

        # idx = np.array(list(set([0, 1, 2, 3, 4, 5, -1, -2, -3, -4, -5, -6]).union(set(idx.flatten()))), dtype='int').reshape(-1, 1)
        idx = np.array(list(set([-1, -2, -3, -4, -5, -6]).union(set(idx.flatten()))), dtype='int').reshape(-1, 1)
        
        jac = np.delete(jac, idx, axis=1)
        for index in idx.flatten():
            self.identifiability_mask[index] = 0
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

    def full_jac(self, dataset, base_only=False, offsets_only=False):
        self.init_metrics()
        jac = np.array([], dtype='float').reshape(0, 36)
        error_vec = np.array([], dtype='float')
        for row in dataset:
            estimated_pose = self.fk(row[:6], 'estimated')
            estimated_coordinates = np.concatenate((estimated_pose[:3, 3], self.extract_zyx_euler(estimated_pose[:3, :3])))
            real_coordinates = row[6:]
            cur_err = real_coordinates[self.measurable_params_mask] - estimated_coordinates[self.measurable_params_mask]
            # if abs(cur_err[0]) > OUTLIER_THRESHOLD or abs(cur_err[1]) > OUTLIER_THRESHOLD or abs(cur_err[2]) > OUTLIER_THRESHOLD:
            #     continue
            if base_only or offsets_only:
                jac = np.concatenate((jac, self.calibration_jacobian(row[:6], self.estimated_base_params,
                                                                     self.estimated_dh, self.estimated_tool_params)), axis=0)
                error_vec = np.concatenate((error_vec, cur_err), axis=0)

            else:
                jac = np.concatenate((jac, TASK_SCALE @ self.calibration_jacobian(row[:6], self.estimated_base_params,
                                                                                    self.estimated_dh, self.estimated_tool_params)), axis=0)
                error_vec = np.concatenate((error_vec, TASK_SCALE @ cur_err), axis=0)

            self.calculate_metrics(cur_err)                              
            
        return jac, error_vec
        
    # def end_of_cycle_action(self, dataset):
    #     val = input()
    #     try:
    #         new_val = float(val)
    #     except ValueError:
    #         if val == 'exit':
    #             self.write_results(self.results_file)
    #             return True
    #         if val == 'test':
    #             self.test_estimated_params()
    #         if val == "plot":
    #             self.draw_plot(dataset)
    #     else:
    #         self.koef = new_val
    #     return False

    def remove_outliers(self, dataset, threshold):
        data_new = np.array([], dtype='float').reshape(0, dataset.shape[1])
        for row in dataset:
            estimated_pose = self.fk(row[:6], 'estimated')
            estimated_coordinates = np.concatenate((estimated_pose[:3, 3], self.extract_zyx_euler(estimated_pose[:3, :3])))
            real_coordinates = row[6:]
            cur_err = real_coordinates[self.measurable_params_mask] - estimated_coordinates[self.measurable_params_mask]
            if abs(cur_err[0]) < threshold and abs(cur_err[1]) < threshold and abs(cur_err[2]) < threshold:
                data_new = np.concatenate((data_new, row.reshape(1, -1)), axis=0)
        return data_new

    def optimize(self):
        self.measurable_params_mask = np.array([0, 1, 2, 3, 4, 5], dtype='int')
        dataset = self.read_dataset(self.dataset_file, FIELDNAMES_OPTIONS["random"])
        while self.norm > 0.001:
            jac, error_vec = self.full_jac(dataset)
            self.norm = np.linalg.norm(error_vec)
            # print(error_vec)
            if self.norm > self.prev_norm and self.optimization_method == "levenberg_marquardt" and self.lm_koef < 1e5:
                self.lm_koef *= 10
            elif self.norm < self.prev_norm and self.optimization_method == "levenberg_marquardt" and self.lm_koef > 1e-10:
                self.lm_koef /= 10
            self.prev_norm = self.norm

            new_jac, scale = self.remove_redundant_params(jac)
            if self.optimization_method == "gauss_newton":
                inc = lsq_linear(new_jac, error_vec).x
            elif self.optimization_method == "levenberg_marquardt":
                inc = np.linalg.inv((new_jac.T @ new_jac + self.lm_koef * np.eye(new_jac.shape[1], new_jac.shape[1]))) @ new_jac.T @ error_vec
            else:
                return
            
            self.update_params(inc, scale)
            self.display_metrics()
            val = input()
            try:
                new_val = float(val)
            except ValueError:
                if val == 'exit':
                    self.write_results(self.results_file)
                    return
                if val == 'test':
                    self.test_estimated_params()
                if val == "plot":
                    self.draw_plot(dataset)
                if val == "tool":
                    tool_dataset = self.read_dataset(self.tool_circles_dataset_file, FIELDNAMES_OPTIONS["circles"])
                    self.draw_plot(tool_dataset)
                if val == "rm":
                    threshold = float(input())
                    dataset = self.remove_outliers(dataset, threshold)
            else:
                self.koef = new_val


    def plane_and_circle(self, dataset, axis):
        req_dataset = dataset[dataset[:, -1] == axis][9:]
        vectors = req_dataset[:, 6:9]
        regressor = np.hstack((req_dataset[:, 7:9], np.ones(req_dataset.shape[0]).reshape(-1, 1)))
        x = req_dataset[:, 6].reshape(-1, 1)
        plane = np.linalg.lstsq(regressor, x, rcond=None)[0].flatten()

        n_vec = np.array([1, -plane[0], -plane[1]], dtype='float')
        n_vec = n_vec / np.linalg.norm(n_vec)

        prod = cross(n_vec, np.array([1, 0, 0], dtype='float'))
        rot_vec = prod / np.linalg.norm(prod)

        angle = acos(n_vec @ np.array([1, 0, 0], dtype='float'))
        mat = self.arbitrary_axis_rot(rot_vec, angle)
        rotated_vectors = vectors @ mat.T

        regressor = np.hstack((rotated_vectors[:, 1:3], np.ones(rotated_vectors.shape[0]).reshape(-1, 1)))
        w = np.power(rotated_vectors[:, 1], 2) + np.power(rotated_vectors[:, 2], 2)
        a, b, c = np.linalg.lstsq(regressor, w, rcond=None)[0].flatten()
        x_c = rotated_vectors[0][0]
        y_c = a/2
        z_c = b/2
        r = sqrt(c + y_c**2 + z_c**2)
        circle = [mat.T @ np.array([x_c, y_c, z_c], dtype='float'), r]
        return plane, circle
    
    def plane_intersection_axis(self, plane_1, plane_2):
        prod = cross(np.array([1, -plane_1[0], -plane_1[1]], dtype='float'), np.array([1, -plane_2[0], -plane_2[1]], dtype='float'))
        return prod / np.linalg.norm(prod)

    def calibrate_base(self):
        self.measurable_params_mask = np.array([0, 1, 2], dtype='int')
        dataset = self.read_dataset(self.base_circles_dataset_file, FIELDNAMES_OPTIONS["circles"])
        req_dataset = dataset[dataset[:, -1] == 1][3:]
        self.draw_plot(req_dataset)

        first_plane, _ = self.plane_and_circle(dataset, 1)
        second_plane, _ = self.plane_and_circle(dataset, 2)
        z = np.array([1, -first_plane[0], -first_plane[1]], dtype='float') / np.linalg.norm([1, -first_plane[0], -first_plane[1]])
        y = self.plane_intersection_axis(first_plane, second_plane)
        x = cross(y, z)
        rot = np.array([x, y, z], dtype='float')
        self.estimated_base_params[3:] = self.extract_zyx_euler(rot.T)
        print(self.estimated_base_params[3:])
        
        while self.norm > 0.001:
            jac, error_vec = self.full_jac(dataset, True)
            self.norm = np.linalg.norm(error_vec)
            # print(error_vec)
            if self.norm > self.prev_norm and self.optimization_method == "levenberg_marquardt" and self.lm_koef < 1e5:
                self.lm_koef *= 10
            elif self.norm < self.prev_norm and self.optimization_method == "levenberg_marquardt" and self.lm_koef > 1e-10:
                self.lm_koef /= 10
            self.prev_norm = self.norm

            new_jac, scale = self.remove_redundant_params(jac, True)
            if self.optimization_method == "gauss_newton":
                inc = lsq_linear(new_jac, error_vec).x
            elif self.optimization_method == "levenberg_marquardt":
                inc = np.linalg.inv((new_jac.T @ new_jac + self.lm_koef * np.eye(new_jac.shape[1], new_jac.shape[1]))) @ new_jac.T @ error_vec
            else:
                return
            
            self.update_params(inc, scale)
            self.display_metrics()
            val = input()
            try:
                new_val = float(val)
            except ValueError:
                if val == 'exit':
                    return
                if val == "plot":
                    self.draw_plot(dataset)
            else:
                self.koef = new_val

    def calibrate_tool(self):
        dataset = self.read_dataset(self.tool_circles_dataset_file, FIELDNAMES_OPTIONS["circles"])
        first_plane, first_circle = self.plane_and_circle(dataset, 5)
        second_plane, second_circle = self.plane_and_circle(dataset, 6)

        vector_1 = np.array([1, -first_plane[0], -first_plane[1]]) / np.linalg.norm([1, -first_plane[0], -first_plane[1]])
        vector_2 = np.array([1, -second_plane[0], -second_plane[1]]) / np.linalg.norm([1, -second_plane[0], -second_plane[1]])
        point_1 = first_circle[0]
        point_2 = second_circle[0]

        mutual_moment = vector_2 @ cross(point_1, vector_1) + vector_1 @ cross(point_2, vector_2)
        axis = -copysign(1, mutual_moment) * cross(vector_1, vector_2) / np.linalg.norm(cross(vector_1, vector_2))

        alpha_5 = -atan2(np.linalg.norm(axis), vector_1 @ vector_2)
        a_5 = (point_1 - point_2) @ axis

        axis_5_params = [a_5, alpha_5, self.estimated_dh[4][2], self.estimated_dh[4][3], self.estimated_dh[4][4]]
        axis_6_params = [second_circle[1], self.estimated_dh[5][1], self.estimated_dh[5][2], self.estimated_dh[5][3], self.estimated_dh[5][4]]

        self.estimated_dh[4] = axis_5_params
        self.estimated_dh[5] = axis_6_params
        # req_dataset = dataset[dataset[:, -1] == 6][3:]
        # self.draw_plot(req_dataset)

def main(args):
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)
    model = HayatiModel(config)
    if args.generate:
        model.generate_dataset()

    # model.generate_dataset()
    model.calibrate_base()
    model.calibrate_tool()
    model.optimize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Name of .json configuration file. Default: ar_5.json", default="ar_5.json")
    parser.add_argument("-g", "--generate", help="Generate dataset for selected method. Default: false", type=bool, default=False)
    args = parser.parse_args()
    main(args)
