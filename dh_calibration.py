import numpy as np

from math import cos, sin, pi, sqrt, atan2
from typing import Union
import random
import csv
from scipy.optimize import minimize
import json
import time
from multipledispatch import dispatch

np.set_printoptions(suppress=True, threshold=np.inf)
RADIUS = 0.025

class HayatiModel:
    def __init__(self):
        # DH params: [a, alpha, d/beta, theta_offset, parallel_axis]. Angle beta is used instead of d if axis is nearly parallel to the previous
        self.nominal_dh = [[0, pi/2,     0.19,   pi/2, 0], #0-1
                           [0.8,  0,     0,      pi/2, 1], #1-2
                           [0.72, 0,     0,      0,    1], #2-3
                           [0,   -pi/2, -0.191, -pi/2, 0], #3-4
                           [0,   -pi/2,  0.13,   0,    0], #4-5
                           [0,    0,     0.069,  0,    0]] #5-6
        
        self.nominal_base_params = [0.4, 0.4, 0, 0, 0, 0]
        self.nominal_tool_params = [0, 0, 0, 0, 0, 0]
        
        self.estimated_dh = None
        self.estimated_base_params = None
        self.estimated_tool_params = None

        self.real_dh = [[0, 1.56,     0.19,   1.56, 0], #0-1
                        [0.801,  0,     0,      1.57, 1], #1-2
                        [0.722, 0.001,     0,      0,    1], #2-3
                        [0.001,   -1.57, -0.191, -1.57, 0], #3-4
                        [-0.001,   -1.57,  0.1305,   0,    0], #4-5
                        [0,    0,     0.0685,  0,    0]] #5-6
        self.real_base_params = [0.41, 0.399, 0, 0, 0, 0]
        self.real_tool_params = [0, 0, 0, 0, 0, 0]

        self.measurable_params_mask = np.array([0, 1, 2], dtype='int')
        self.identifiability_mask = np.ones(36, dtype='int')
        self.koef = 0.001

    
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

    def numerical_ik(self, target_pose: np.ndarray, initial_guess: Union[np.ndarray, list]) -> np.ndarray:
        def cost(x, target_pose):
            pose = self.fk(x, 'nominal')
            res = np.linalg.norm(pose - target_pose)
            return res

        result = minimize(cost, x0=initial_guess, args=target_pose, method='BFGS', options={"gtol":1e-17})
        return result.x, result.fun

    def generate_angles(self, inc_rx, inc_ry, inc_rz, initial_pose, initial_angles, samples):
        angles = [initial_angles]
        nominal_positions = [self.fk(initial_angles, 'nominal')[:3, 3]]
        real_positions = [self.fk(initial_angles, 'real')[:3, 3]]

        pose = initial_pose
        for i in range(samples):
            # pose = pose @ self.z_rot(inc_rz) @ self.y_rot(inc_ry) @ self.x_rot(inc_rx)
            # angle_set, val = self.numerical_ik(pose, angles[i - 1])
            # if val < 1.0e-5:
            #     angles.append(angle_set)
            #     nominal_positions.append(self.fk(angle_set, 'nominal')[:3, 3])
            #     real_positions.append(self.fk(angle_set, 'real')[:3, 3])
            angle_set = (np.random.rand(6) - 0.5) * 2 * pi
            angles.append(angle_set)
            nominal_positions.append(self.fk(angle_set, 'nominal')[:3, 3])
            real_positions.append(self.fk(angle_set, 'real')[:3, 3])

        with open('data.csv', 'w', newline='') as csvfile:
            fieldnames = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6',
                        'px_n', 'py_n', 'pz_n', 'px_r', 'py_r', 'pz_r']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for q_vector, pos_n, pos_r in zip(angles, nominal_positions, real_positions):
                writer.writerow({
                    'q1': q_vector[0],
                    'q2': q_vector[1],
                    'q3': q_vector[2],
                    'q4': q_vector[3],
                    'q5': q_vector[4],
                    'q6': q_vector[5],
                    'px_n': pos_n[0],
                    'py_n': pos_n[1],
                    'pz_n': pos_n[2],
                    'px_r': pos_r[0],
                    'py_r': pos_r[1],
                    'pz_r': pos_r[2]
                })

    def read_dataset(self, file_name: str) -> Union[np.ndarray, np.ndarray]:
        angles = []
        nominal_positions = []
        real_positions = []
        with open(file_name, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                angles.append([float(row['q1']), float(row['q2']), float(row['q3']), float(row['q4']), float(row['q5']), float(row['q6'])])
                nominal_positions.append([float(row['px_n']), float(row['py_n']), float(row['pz_n'])])
                real_positions.append([float(row['px_r']), float(row['py_r']), float(row['pz_r'])])

        return np.array(angles), np.array(nominal_positions), np.array(real_positions)
     
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

        # print(from_base_trans)
        # print()
        # print(from_tool_trans)

        # Jac structure: [[tx_b, ty_b, tz_b, rx_b, ry_b, rz_b], [d_a, d_alpha, d_d/d_beta, d_theta_off] - len(tfs) times, [tx_t, ty_t, tz_t, rx_t, ry_t, rz_t]]
        # [tx_b, ty_b, tz_b, rx_b, ry_b, rz_b] always equals eye(6, 6)
        # [tx_t, ty_t, tz_t, rx_t, ry_t, rz_t] = / R  0 \ ,  R - rotation matrix of tool relative to base
        #                                        \ 0  R /
        # Vertical order: [tx, ty, tz, rx, ry, rz]
        # Model is redundant in number of params, but as calibration method uses only position measurements, base rotation and
        # tool offset are unobservable. So, total number of remaining params is guaranteed to be less than 30 for 6R robot.
        
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

            # print('x_init: ', from_base_trans[i - 1][0:3, 0])
            # print('x_rot: ', xi_rot)
            z_vec = from_base_trans[i - 1][0:3, 2]
            # zi_rot = self.z_rot(offsets[i - 1] + angles[i - 1])[0:3, 0:3] @ z_vec
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

            # print(jac_sub[self.measurable_params_mask, :])

            jac[:, (6+(i-1)*4):(10+(i-1)*4)] = jac_sub

        return jac[self.measurable_params_mask, :]

    # def direct_sphere_model(self, reference_pose: np.ndarray, angles: list) -> list:
    #     detector_readings = []
    #     x0, y0, z0 = reference_pose[:3, 3]
    #     for pose in angles:
    #         xc, yc, zc = self.fk(pose, 'real')[:3, 3]

    #         b = -2*xc
    #         c = xc**2 + (y0 - yc)**2 + (z0 - zc)**2 - RADIUS**2
    #         x_det = (-b - sqrt(b**2 - 4*c))/2

    #         b = -2*yc
    #         c = yc**2 + (x0 - xc)**2 + (z0 - zc)**2 - RADIUS**2
    #         y_det = (-b - sqrt(b**2 - 4*c))/2

    #         b = -2*zc
    #         c = zc**2 + (x0 - xc)**2 + (y0 - yc)**2 - RADIUS**2
    #         z_det = (-b - sqrt(b**2 - 4*c))/2
    #         detector_readings.append(np.array([x_det, y_det, z_det]))

    #     return detector_readings
    
    # def inverse_sphere_model(self, reference_pose: np.ndarray, angles: list) -> list:
    #     detector_readings = []
    #     x0, y0, z0 = reference_pose[:3, 3]
    #     for pose in angles:
    #         xc, yc, zc = self.fk(pose, 'real')[:3, 3]

    #         b = -2*xc
    #         c = xc**2 + (y0 - yc)**2 + (z0 - zc)**2 - RADIUS**2
    #         x_det = (-b - sqrt(b**2 - 4*c))/2

    #         b = -2*yc
    #         c = yc**2 + (x0 - xc)**2 + (z0 - zc)**2 - RADIUS**2
    #         y_det = (-b - sqrt(b**2 - 4*c))/2

    #         b = -2*zc
    #         c = zc**2 + (x0 - xc)**2 + (y0 - yc)**2 - RADIUS**2
    #         z_det = (-b - sqrt(b**2 - 4*c))/2
    #         detector_readings.append(np.array([x_det, y_det, z_det]))

    #     return detector_readings

    def scaling_matrix(self, jac: np.ndarray) -> np.ndarray:
        diag = np.linalg.norm(jac, axis=0)
        diag[diag < 10**(-8)] = 1
        return np.linalg.inv(np.diag(diag))

    def remove_redundant_params(self, jac: np.ndarray) -> np.ndarray:
        self.identifiability_mask = np.ones(36, dtype='int')
        scale = self.scaling_matrix(jac)
        jac = jac @ scale
        idx = np.argwhere(np.all(jac[..., :] == 0, axis=0))
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
        print("Cond: ", cond)
        return jac, scale
    
    def pack_param_vec(self, base_i,  dh_i, tool_i):
        base = np.array(base_i)
        tool = np.array(tool_i)
        dh = np.array(dh_i)
        dh = np.delete(dh, 4, 1)
        return np.concatenate((base, dh.flatten(), tool))

    def unpack_param_vec(self, vec):
        base = vec[:6]
        dh = np.reshape(vec[6:-6], (6, int((vec.size - 12) / 6 )))
        dh = np.concatenate((dh, np.array(self.nominal_dh)[:, 4].reshape(6, 1)), axis=1).tolist()
        tool = vec[-6:]
        return base, dh, tool

    def update_params(self, inc, scale):
        vec = self.pack_param_vec(self.estimated_base_params, self.estimated_dh, self.estimated_tool_params)
        mapped_inc = inc[np.cumsum(self.identifiability_mask) - 1]
        mapped_inc[self.identifiability_mask == 0] = 0
        mapped_inc = np.linalg.inv(scale) @ mapped_inc
        vec += mapped_inc * self.koef
        self.estimated_base_params, self.estimated_dh, self.estimated_tool_params = self.unpack_param_vec(vec)

    def gauss_newton_ls(self):
        angles, _, real_positions = self.read_dataset('data.csv')
        self.estimated_dh = self.nominal_dh
        self.estimated_base_params = self.nominal_base_params
        self.estimated_tool_params = self.nominal_tool_params

        while True:
            jac = np.array([], dtype='float').reshape(0, 36)
            error_vec = np.array([], dtype='float')
            for index, angle_set in enumerate(angles):
                jac = np.concatenate((jac, self.calibration_jacobian(angle_set, self.estimated_base_params,
                                                                     self.estimated_dh, self.estimated_tool_params)), axis=0)
                estimated_position = self.fk(angle_set, 'estimated')[:3, 3]
                error_vec = np.concatenate((error_vec, real_positions[index] - estimated_position), axis=0)

            new_jac, scale = self.remove_redundant_params(jac)
            inc = np.linalg.pinv(new_jac) @ error_vec
            self.update_params(inc, scale)
            print("Norm: ", np.linalg.norm(error_vec))
            print("Identifiables: ", self.identifiability_mask, f" ({self.identifiability_mask.sum()})")
            print("Base: ", self.estimated_base_params)
            print("DH: ", self.estimated_dh)
            print("Tool: ", self.estimated_tool_params)
            input()

    
def cross(a:np.ndarray,b:np.ndarray)->np.ndarray:
    return np.cross(a,b)

def main():
    model = HayatiModel()

    # Dataset generation
    # target_pose = np.array([[1, 0,  0,  0.5],
    #                         [0, 0, -1,  0.5],
    #                         [0, 1,  0,  1.0],
    #                         [0, 0,  0,  1   ]], dtype="float")
    # first_result, _ = model.numerical_ik(target_pose, [0, 1, 1, 0, 0, 0])
    # model.generate_angles(0.01, 0.01, 0.01, target_pose, first_result, 300)

    # model.calibration_jacobian([0, 0, 0, 0, 0, 0], model.nominal_base_params, model.nominal_dh, model.nominal_tool_params)

    model.gauss_newton_ls()
    # print(model.fk([0, 1, 1, 0, 0, 0], 'nominal'))


if __name__ == "__main__":
    main()


