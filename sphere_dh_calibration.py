import numpy as np
from math import cos, sin, pi, sqrt, atan2, asin, log10
from typing import Union
import csv
from scipy.optimize import minimize, lsq_linear
from random import random
import argparse
import json
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True, threshold=np.inf, precision=5)

RADIUS = 0.025
BIG_NUMBER = 10e300
FIELDNAMES_OPTIONS = {
    "sphere" : ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'px_n', 'py_n', 'pz_n'],
    "random" : ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'px_r', 'py_r', 'pz_r', 'rx_r', 'ry_r', 'rz_r'],
}

def cross(a:np.ndarray,b:np.ndarray)->np.ndarray:
    return np.cross(a,b)

class HayatiModel:
    def __init__(self, config):
        self.optimization_method = config['optimization_method']
        self.fieldnames = FIELDNAMES_OPTIONS['sphere']
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
        
        self.initial_rotation_matrix = np.array([[0,  1,  0],
                                                 [1,  0,  0],
                                                 [0,  0, -1]], dtype='float')
        

        self.sphere_points = config["sphere_points"]
        self.joint_limits_general_l = config["joint_limits_general_l"]
        self.joint_limits_general_h = config["joint_limits_general_h"]

        self.bypass_sphere_models = True
        self.samples_number = config["general_samples_number"]
        self.current_point = 0

        self.measurable_params_mask = np.array([0, 1, 2], dtype='int')

        self.identifiability_mask = np.ones(36, dtype='int')
        self.koef = 0.001
        self.norm = 10
        self.lm_koef = 0.01
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

    def generate_angles(self, initial_pose, initial_angles, samples):
        angles = [initial_angles]
        pose = initial_pose
        for i in range(samples):
            pose = initial_pose @ self.z_rot(random()*pi) @ self.y_rot(random()*pi) @ self.x_rot(random()*pi)
            angle_set, val = self.numerical_ik(pose, angles[i - 1], 'estimated')
            if val < 1.0e-5:
                angles.append(angle_set)
        return angles

    def generate_sphere_dataset(self, point_number, samples=300):
        dataset = np.array([], dtype='float').reshape(0, len(self.fieldnames))

        point = self.sphere_points[self.num_point : self.num_point + 1][point_number]
        pose = np.eye(4, 4)
        pose[0:3, 0:3] = self.initial_rotation_matrix
        pose[:3, 3] = np.array(point)
        iterations = 0
        val = BIG_NUMBER
        while val > 1.0e-5 and iterations < 1000:
            angle_set, val = self.numerical_ik(pose, [self.joint_limits_general_l[axis] + random() * (self.joint_limits_general_h[axis] - self.joint_limits_general_l[axis]) for axis in range(6)], 'estimated')
            iterations += 1
        if iterations == 1000:
            print(f"Unable to find good solution for point {point}")
            return None
        angles = self.generate_angles(pose, angle_set, samples)

        for angle_set in angles:
            dataset = np.concatenate((dataset, np.concatenate((angle_set, point)).reshape(1, len(self.fieldnames))), axis=0)

        return dataset

    def direct_sphere_model(self, reference_position: np.ndarray, angles: np.ndarray) -> np.ndarray:
        x0, y0, z0 = reference_position
        xc, yc, zc = self.fk(angles, 'real')[:3, 3] - reference_position

        if self.bypass_sphere_models:
            return np.array([xc, yc, zc], dtype='float')

        b = -2*xc
        c = xc**2 + (y0 - yc)**2 + (z0 - zc)**2 - RADIUS**2
        x_det = (-b - sqrt(b**2 - 4*c))/2

        b = -2*yc
        c = yc**2 + (x0 - xc)**2 + (z0 - zc)**2 - RADIUS**2
        y_det = (-b - sqrt(b**2 - 4*c))/2

        b = -2*zc
        c = zc**2 + (x0 - xc)**2 + (y0 - yc)**2 - RADIUS**2
        z_det = (-b - sqrt(b**2 - 4*c))/2

        return -(np.array([x_det, y_det, z_det], dtype='float') + RADIUS)
    
    def inverse_sphere_model(self, detector_readings: np.ndarray) -> np.ndarray:
        if self.bypass_sphere_models:
            return detector_readings

        xd, yd, zd = -detector_readings - RADIUS

        b1 = 0.5 * (xd**2 - zd**2) / xd
        a1 = zd / xd
        b2 = 0.5 * (yd**2 - zd**2) / yd
        a2 = zd / yd

        a = (a1**2 + a2**2 + 1)
        b = 2*(a1 * b1 + a2 * b2 - zd)
        c = b1**2 + b2**2 + zd**2 - RADIUS**2

        z = 0.5 * (-b + sqrt(b**2 - 4*a*c)) / a
        x = a1 * z + b1
        y = a2 * z + b2

        return np.array([x, y, z], dtype='float')

    def generate_random_dataset(self, samples):
        dataset = np.zeros((samples, len(FIELDNAMES_OPTIONS['random'])))
        for i in range(samples):
            angle_set = np.array([self.joint_limits_general_l[axis] + random() * (self.joint_limits_general_h[axis] - self.joint_limits_general_l[axis]) for axis in range(6)], dtype='float')
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

    def remove_redundant_params(self, jac: np.ndarray) -> np.ndarray:
        self.identifiability_mask = np.ones(36, dtype='int')
        scale = self.scaling_matrix(jac)
        jac = jac @ scale
        idx = np.argwhere(np.all(jac[..., :] == 0, axis=0))
        idx = np.array(list(set([0, 1, 2, 3, 4, 5, -1, -2, -3, -4, -5, -6]).union(set(idx.flatten()))), dtype='int').reshape(-1, 1)
        
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

    def full_jac(self, dataset):
        self.init_metrics()
        jac = np.array([], dtype='float').reshape(0, 36)
        error_vec = np.array([], dtype='float')
        for row in dataset:
            jac = np.concatenate((jac,self.calibration_jacobian(row[:6], self.estimated_base_params,
                                                                self.estimated_dh, self.estimated_tool_params)), axis=0)
            cur_err = self.inverse_sphere_model(self.direct_sphere_model(row[6:], row[:6]))
            self.calculate_metrics(cur_err)
            error_vec = np.concatenate((error_vec, cur_err))

        return jac, error_vec

    def end_of_cycle_action(self, dataset):
        val = input()
        try:
            new_val = float(val)
        except ValueError:
            if val == 'exit':
                self.write_results(self.results_file)
                return True
            if val == 'test':
                self.test_estimated_params()
            if val == "plot":
                self.draw_plot(dataset)
            if val == "point":
                self.current_point = int(input)
        else:
            self.koef = new_val
        return False
    
    def optimize(self):
        while self.norm > 0.001:
            dataset = self.generate_sphere_dataset(self.current_point, self.samples_number)
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
            if self.end_of_cycle_action(dataset):
                break
        
def main(args):
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)
    model = HayatiModel(config)
    model.optimize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Name of .json configuration file. Default: ar_20_sphere.json", default="ar_20_sphere.json")
    args = parser.parse_args()
    main(args)
