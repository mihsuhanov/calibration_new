import numpy as np

from math import cos, sin, pi, sqrt, atan2
from typing import Union
import random
import csv
from scipy.optimize import minimize
import json
import time
np.set_printoptions(suppress=True)

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.pyplot import MultipleLocator

class AR6DOFManipulatorCalibration():
    def __init__(self) -> None:
        self.variable_length_keys = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # [0]*12 #  
        self.variable_offsets_keys = [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        self.length = [130, 460, 100, 275, 30, 135, 70] + [0, 0, 0, 0, 0]
        self.offsets = [0] * 6 + [pi/2, 0, 0, -pi/2, pi/2, 0]
        self.static_offset = [pi, -pi/2, pi, -pi/2, pi, 0]
        self.sensor_pose = 33
        self.upadateDHparams(self.length)
    
    def setLength(self, length: np.ndarray) -> None:
        self.length = length
        self.upadateDHparams(length)
    
    def setOffsets(self, offsets) -> None:
        self.offsets = offsets
        self.upadateDHparams(self.length)
        
    def upadateDHparams(self, length):
        self.dh_params = np.array([[length[0], self.offsets[6], length[7]],
             [length[8], self.offsets[7], -length[1]],
             [length[2], self.offsets[8], length[3]],
             [-length[4], self.offsets[9], length[9]],
             [length[5], self.offsets[10], length[10]],
             [length[6] + self.sensor_pose, self.offsets[11], length[11]]])
        
    def transforn_matrix_dh(self, q:float, d:float, alp:float, a:float) -> np.ndarray:
        matrix = np.array([[cos(q), -cos(alp)*sin(q), sin(alp)*sin(q), a*cos(q)],
                        [sin(q),  cos(alp)*cos(q), -sin(alp)*cos(q), a*sin(q)],
                        [0, sin(alp), cos(alp), d],
                        [0, 0, 0, 1]])
        return matrix

    def fk(self, q_vector: np.ndarray) -> np.ndarray:
        res = np.eye(4)
        # res = cp.eye(4)
        for params, q_i, offset, static_offset in zip(self.dh_params, q_vector, self.offsets, self.static_offset):
            # mat = self.transforn_matrix_dh(q_i + offset + static_offset, *params)
            # res = cp.matmul(res, cp.array(mat))
        # cp.cuda.Stream.null.synchronize()
            res = res@self.transforn_matrix_dh(q_i + offset + static_offset, *params)
        return res

    def tf_list(self, q_vector):
        res = np.eye(4)
        tfs_base = []
        tfs = []
        for params, q_i, offset, static_offset in zip(self.dh_params, q_vector, self.offsets[0:6], self.static_offset[0:6]):
            tf = self.transforn_matrix_dh(q_i + offset + static_offset, *params)
            res = res@tf
            tfs.append(tf)
            tfs_base.append(res)
        return tfs_base, tfs
    
    def y_rotate_matrix(self, angle: float) -> np.ndarray:
        mat = np.array([[cos(angle), 0, sin(angle), 0],
                        [0, 1, 0, 0],
                        [-sin(angle), 0, cos(angle), 0],
                        [0, 0, 0, 1]])
        return mat

    def z_rotate_matrix(self, angle: float) -> np.ndarray:
        mat = np.array([[cos(angle), -sin(angle), 0, 0],
                        [sin(angle), cos(angle), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        return mat

    def x_rotate_matrix(self, angle: float) -> np.ndarray:
        mat = np.array([[1, 0, 0, 0],
                        [0, cos(angle), -sin(angle), 0],
                        [0, sin(angle), cos(angle), 0],
                        [0, 0, 0, 1]])
        
        return mat

    def translate_matrix(self, vector: list) -> np.ndarray:
        mat = np.array([[1, 0, 0, vector[0]],
                        [0, 1, 0, vector[1]],
                        [0, 0, 1, vector[2]],
                        [0, 0, 0, 1]])

        return mat

    def tf_list_in_motorcortex_cs(self, q_vector):
        tf1 = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, self.length[0]],
                        [0, 0, 0, 1]])
        
        tf2 = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, self.length[1]],
                        [0, 0, 0, 1]])
        
        tf3 = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, self.length[3]],
                        [0, 0, 0, 1]])

        tf4 = np.array([[1, 0, 0, 0],
                        [0, 1, 0, self.length[2] - self.length[4]],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        
        tf5 = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, self.length[5]],
                        [0, 0, 0, 1]])

        tf6 = np.array([[1, 0, 0, 0],
                        [0, 1, 0, -self.length[6]],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

        tf7 = np.array([[1, 0, 0, 0],
                        [0, 0, -1, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1]])
        
        tf1 = self.z_rotate_matrix(q_vector[0] + self.offsets[0])@tf1@self.x_rotate_matrix(pi/2-self.offsets[6])@self.translate_matrix([-self.length[7], 0, 0])
        tf2 = self.y_rotate_matrix(q_vector[1] + self.offsets[1])@self.translate_matrix([0, self.length[8], 0])@self.z_rotate_matrix(-self.offsets[7])@tf2
        tf3 = self.y_rotate_matrix(q_vector[2] + self.offsets[2])@tf3
        tf4 = self.y_rotate_matrix(q_vector[3] + self.offsets[3])@tf4@self.x_rotate_matrix(-pi/2-self.offsets[9])@self.translate_matrix([-self.length[9], 0, 0])
        tf5 = self.z_rotate_matrix(q_vector[4] + self.offsets[4])@tf5@self.x_rotate_matrix(-pi/2+self.offsets[10])@self.translate_matrix([self.length[10], 0, 0])
        tf6 = self.y_rotate_matrix(-q_vector[5] - self.offsets[5])@tf6@self.translate_matrix([self.length[11], 0, 0])
        
        tf6_7 = tf6#@tf7

        tfs = [tf1, tf2, tf3, tf4, tf5, tf6_7]

        base_tfs = []
        res = np.eye(4)
        for tf in tfs:
            res = res@tf
            base_tfs.append(res)

        return base_tfs, tfs

    def optimize(self, qs:np.ndarray, points:np.ndarray, eulerBase_0: list, transBase_0: list, estimate=False, print_iter=True) -> dict:
        iteration = 0
        result_error = 0

        def cost(params):
            nonlocal iteration
            nonlocal result_error

            euler_base = params[-3:]
            base_trans = params[-6:-3]
            new_variable_length = params[:sum(self.variable_length_keys)].tolist()
            new_variable_offsets = params[sum(self.variable_length_keys):sum(self.variable_length_keys)+sum(self.variable_offsets_keys)].tolist()

            for i, key in enumerate(self.variable_length_keys):
                if not key:
                    new_variable_length.insert(i, self.length[i])
            
            for i, key in enumerate(self.variable_offsets_keys):
                if not key:
                    new_variable_offsets.insert(i, self.offsets[i])

            self.setLength(new_variable_length)
            self.setOffsets(new_variable_offsets)

            tBase = tf_euler_trans(euler_base, base_trans)

            error = 0
            for point, q_vector in zip(points, qs):
                fk_t_base = tBase@self.fk(q_vector)
                fk_point_base = fk_t_base[:3, 3]
                diff = fk_point_base - point
                res = np.linalg.norm(diff)
                error += res

            iteration += 1
            result_error = error
            if print_iter:
                print("[Iteration: {0}] Error: {1}".format(iteration, error))

            return error
        
        variable_length = []
        variable_offsets = []
        for key, length in zip(self.variable_length_keys, self.length):
            if key:
                variable_length.append(length)

        for key, offset in zip(self.variable_offsets_keys, self.offsets):
            if key:
                variable_offsets.append(offset)

        params_0 = variable_length + variable_offsets + transBase_0 + eulerBase_0

        bounds = []
        for length in variable_length:
            bounds.append((length-5, length+5))
        
        for offset in variable_offsets:
            bounds.append((offset-0.1, offset+0.1))
        
        for coord in transBase_0:
            bounds.append((coord-1, coord+1))
        
        for coord in eulerBase_0:
            bounds.append((coord-0.01, coord+0.10))

        print("\n\n\n-----Start minimaze by all parameters-----")
        s_time = time.time()
        minimaze_result = minimize(cost, params_0, bounds=tuple(bounds), method='L-BFGS-B',options={"ftol":1e-7, "gtol":1e-9})
        print("-----Stop minimaze by all parameters. Time: {0} -----".format(time.time() - s_time))
        
        new_lengths = minimaze_result.x[:sum(self.variable_length_keys)].tolist()
        new_offsets = minimaze_result.x[sum(self.variable_length_keys):sum(self.variable_length_keys)+sum(self.variable_offsets_keys)].tolist()

        for i, key in enumerate(self.variable_length_keys):
            if not key:
                new_lengths.insert(i, self.length[i])
        
        for i, key in enumerate(self.variable_offsets_keys):
            if not key:
                new_offsets.insert(i, self.offsets[i])

        result = np.round(np.array(new_lengths + new_offsets + minimaze_result.x[-6:].tolist()), 4).tolist()

        res = {
            "result": result,
            "iterations": iteration,
            "result_fun": result_error
        }

        if estimate:

            self.setLength(new_lengths)
            self.setOffsets(new_offsets)
            tBase = tf_euler_trans(minimaze_result.x[-3:], minimaze_result.x[-6:-3])
            err = self.estimate_position(qs, points, tBase)
            res['estimate'] = err
        
        return res
    
    def optimize_base(self, qs:np.ndarray, points:np.ndarray, eulerBase_0: list, transBase_0: list, bounds=None, estimate=False, print_iter=True) -> dict:
        iteration = 0
        result_error = 0

        def cost(params):
            nonlocal iteration
            nonlocal result_error

            euler_base = params[3:]
            base_trans = params[:3]

            tBase = tf_euler_trans(euler_base, base_trans)

            error = 0
            for point, q_vector in zip(points, qs):
                fk_t_base = tBase@self.fk(q_vector)
                fk_point_base = fk_t_base[:3, 3]
                diff = fk_point_base - point
                res = np.linalg.norm(diff)
                error += res

            iteration += 1
            result_error = error
            if print_iter:
                print("[Iteration: {0}] Error: {1}".format(iteration, error))

            return error

        params_0 = transBase_0 + eulerBase_0
        
        print("\n\n\n-----Start minimaze by base parameters-----")
        s_time = time.time()
        minimaze_result = minimize(cost, params_0, bounds=bounds, method='L-BFGS-B', options={"ftol":1e-7, "gtol":1e-9})
        print("-----Stop minimaze by base parameters. Time: {0} -----".format(time.time() - s_time))

        res = {
            "result": minimaze_result.x.round(decimals=4, out=minimaze_result.x).tolist(),
            "iterations": iteration,
            "result_fun": result_error
        }

        if estimate:
            tBase = tf_euler_trans(minimaze_result.x[3:], minimaze_result.x[:3])
            err = self.estimate_position(qs, points, tBase)
            res['estimate'] = err
        
        return res

    def estimate_position(self, qs: list, points: list, Tbase: np.ndarray) -> dict:
        x_err = []
        y_err = []
        z_err = []
        dist_err = []
        noize_point = 0
        noize_index = []
        i = 0
        for point, q_vector in zip(points, qs):
            fk_t_base = Tbase@self.fk(q_vector)
            fk_point_base = fk_t_base[:3, 3]
            error = abs(fk_point_base) - abs(point)
            # if np.linalg.norm(error) > 3:
            #     noize_point +=1
            #     noize_index.append(i)
            # else:
            x_err.append(abs(error[0]))
            y_err.append(abs(error[1]))
            z_err.append(abs(error[2]))
            dist_err.append(sqrt(x_err[-1]**2 + y_err[-1]**2 + z_err[-1]**2))
            i += 1
        max_dist = max(dist_err)
        min_dist = min(dist_err)
        delta_dist = max_dist - min_dist
        step = 30
        err_rasp = [0]*step
        for dist in dist_err:
            for i in range(step):
                if delta_dist/step*i < dist and dist < delta_dist/step*(i+1):
                    err_rasp[i] += 1
                    continue

        print(noize_index)

        return {
                'x_err_max': round(max(x_err), 4),
                'y_err_max': round(max(y_err), 4),
                'z_err_max': round(max(z_err), 4),
                'dist_err_max': round(max(dist_err), 4),
                'x_err_mean': round(np.mean(x_err), 6),
                'y_err_mean': round(np.mean(y_err), 6),
                'z_err_mean': round(np.mean(z_err), 6),
                'dist_err_mean': round(np.mean(dist_err), 6),
                'x_err_median': round(np.median(x_err), 6),
                'y_err_median': round(np.median(y_err), 6),
                'z_err_median': round(np.median(z_err), 6),
                'dist_err_median': round(np.median(dist_err), 6),
                'disposal': {'min':min_dist, 'max':max_dist, 'data': err_rasp},
                'noize_point': noize_point
            }

    def estimate_fk(self, qs: list, points_fk: list) -> dict:
        x_max = 0
        y_max = 0
        z_max = 0
        for point, q_vector in zip(points_fk, qs):
            fk_model = self.fk(q_vector)
            fk_model_pose = fk_model[:3, 3]
            error = abs(fk_model_pose) - abs(point)
            if abs(error[0]) > x_max:
                x_max = abs(error[0])
            if abs(error[1]) > y_max:
                y_max = abs(error[1])
            if abs(error[2]) > z_max:
                z_max = abs(error[2])
        dist_err = sqrt(x_max**2 + y_max**2 + z_max**2)
        return {'x_err': x_max, 'y_err': y_max, 'z_err': z_max, 'dist_err': dist_err}

def create_dataset(offsets, length, roll_b, pitch_b, yaw_b, trans_b, points_num):
    robot = AR6DOFManipulatorCalibration()
    robot.setLength(length)
    robot.setOffsets(offsets)

    Tbase = tf_euler_trans([roll_b, pitch_b, yaw_b], trans_b)

    js = []
    points = []
    points_fk = []

    for i in range(points_num):
        new_q = []
        for j in range(6):
            new_q.append(random.random()*pi - pi/2)
        js.append(new_q)
        point = robot.fk(new_q)
        point_base = Tbase@point
        points.append(point_base[0:3, 3])

    with open('data.csv', 'w', newline='') as csvfile:
        fieldnames = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6',
                      'px', 'py', 'pz', 'fk_z', 'fk_y', 'fk_z']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for q_vector, point, point_fk in zip(js, points, points_fk):
            writer.writerow({
                'q1': q_vector[0],
                'q2': q_vector[1],
                'q3': q_vector[2],
                'q4': q_vector[3],
                'q5': q_vector[4],
                'q6': q_vector[5],
                'px': point[0],
                'py': point[1],
                'pz': point[2],
                'fk_x': point_fk[0],
                'fk_y': point_fk[1],
                'fk_z': point_fk[2]
            })

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

    return np.array(js), np.array(points), np.array(points_fk)

def tf_euler_trans(euler: list, trans: list) -> np.ndarray:
    rx = np.array([[1, 0, 0],
                   [0, cos(euler[0]), -sin(euler[0])],
                   [0, sin(euler[0]), cos(euler[0])]])

    ry = np.array([[cos(euler[1]), 0, sin(euler[1])],
                   [0, 1, 0],
                   [-sin(euler[1]), 0, cos(euler[1])]])

    rz = np.array([[cos(euler[2]), sin(euler[2]), 0],
                     [sin(euler[2]), -cos(euler[2]), 0],
                     [0, 0, 1]])
    
    R = rx@ry@rz
    R = R.tolist()
    R[0].append(trans[0])
    R[1].append(trans[1])
    R[2].append(trans[2])
    R.append([0,0,0,1])
    return np.array(R)

def draw_point_with_robot(qs: list, points: list, eulerBase: list, transBase: list) -> None:
    robot = AR6DOFManipulatorCalibration()

    tBase = tf_euler_trans(eulerBase, transBase)
    inv_tBase = np.linalg.inv(tBase)

    manip_point = []
    for point in points:
        point = point.tolist()+[1]
        t = inv_tBase@np.array(point)
        manip_point.append(t[:3]*0.001)
    manip_point = np.array(manip_point)

    x_major_locator = MultipleLocator(0.1)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    def animate_robot(i):
        if i < len(qs):
            tfs, _ = robot.tf_list(qs[i])
            ax.clear()
            ax.xaxis.set_major_locator(x_major_locator)
            ax.yaxis.set_major_locator(x_major_locator)
            ax.zaxis.set_major_locator(x_major_locator)

            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            ax.set_xlim(-0.7, 0.7) 
            ax.set_ylim(-0.7, 0.7) 
            ax.set_zlim(-0.7, 0.7)

            ax.plot(manip_point[:,0][:i], manip_point[:,1][:i], manip_point[:,2][:i])

            for i in range(len(tfs)):
                if i == 0 :
                    ax.plot([0, tfs[i][0,3]*0.001], [0, tfs[i][1,3]*0.001], [0, tfs[i][2,3]*0.001], linewidth=10, label=r'$z=y=x$')
                else:
                    ax.plot([tfs[i-1][0,3]*0.001, tfs[i][0,3]*0.001], [tfs[i-1][1,3]*0.001, tfs[i][1,3]*0.001], [tfs[i-1][2,3]*0.001, tfs[i][2,3]*0.001], linewidth=10, label=r'$z=y=x$')
    ani = animation.FuncAnimation(fig, animate_robot, interval=100)
    plt.show()

def draw_disposal_data(est, points):
    data_len = len(est['disposal']['data'])
    data_min = est['disposal']['min']
    data_max = est['disposal']['max']

    step = (est['disposal']['max'] - est['disposal']['min'])/data_len
    x_data = []
    y_data = np.array(est['disposal']['data'])/len(points)
    for i in range(data_len):
        x_data.append(data_min + step*i)
    plt.grid()
    plt.plot(x_data, y_data)
    plt.show()

def draw_raw_sensor_data(points):
    x = []
    y = []
    y_median = []
    alpha = 0.5
    index = 1
    for i, point in enumerate(points):
        x.append(i)
        y.append(point[index])
        if len(y_median) == 0:
            y_median.append(point[index])
        else:
            y_median.append(point[index]*alpha + y_median[i-1]*(1-alpha))

    plt.plot(x, y)
    # plt.plot(x, y_median)
    plt.show()

def trans_euler_from_tf(tf):
    sy = sqrt(tf[0,0] * tf[0,0] +  tf[1,0] * tf[1,0])
 
    singular = sy < 1e-6
 
    if  not singular :
        x = atan2(tf[2,1] , tf[2,2])
        y = atan2(-tf[2,0], sy)
        z = atan2(tf[1,0], tf[0,0])
    else :
        x = atan2(-tf[1,2], tf[1,1])
        y = atan2(-tf[2,0], sy)
        z = 0

    trans = tf[:3,3]*0.001
 
    return np.array(trans.tolist() + [x, y, z])

def main():
    qs, points, points_fk = read_dataset("calibration.csv")
    eulerBase = [ 0, 0, 0]
    transBase = [ 0, 0, 0]
    baseBounds = ((-600, 300), (-2000, -700), (-300, 500), (-0.15, 0,15), (-0.15, 0.15), (-pi, pi))

    noize = []
    noize.reverse()
    qs_list = qs.tolist()
    points_list = points.tolist()
    points_fk_list = points_fk.tolist()
    for index in noize:
        qs_list.pop(index)
        points_list.pop(index)
        points_fk_list.pop(index)
    qs = np.array(qs_list)
    points = np.array(points_list)
    points_fk = np.array(points_fk_list)
    
    robot = AR6DOFManipulatorCalibration()

    optimize_len = [
        127.7177,
        457.1569,
        97.7467,
        273.1823,
        32.2533,
        130.0,
        71.8016,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    optimize_off = [
        0.0,
        0.01,
        -0.01,
        -0.003,
        0.01,
        0.0,
        1.5708,
        0.0,
        0.0,
        -1.5708,
        1.5708,
        0.0,
    ]

    # robot.setLength(optimize_len)
    # robot.setOffsets(optimize_off)

    # q = [0, 0, 0, 0, 0, 0]
    # tfs_base_kin, tf_kin = robot.tf_list(q)
    # tfs_base_motorcortex, tf_motorcortex = robot.tf_list_in_motorcortex_cs(q)

    # for index, tf in enumerate(tf_motorcortex):
    #     print("Joint {0} angles: {1}, {2}, {3}; positions: {4} {5} {6}".format(index + 1,*trans_euler_from_tf(tf)))
    
    # print("Offsets: {0}".format([0.011,
    #     0.0085,
    #     -0.0445,
    #     0.0988,
    #     0.0036,
    #     -0.1]))


    # print(tfs_base_kin[-1])
    # print(tfs_base_motorcortex[-1])

    # tBase = tf_euler_trans(eulerBase, transBase)
    # robot.setOffsets(optimize_off)
    # est = robot.estimate_position(qs, points, tBase)
    # print(json.dumps(est, indent=4))

    base_pose = robot.optimize_base(qs, points, eulerBase, transBase, bounds=baseBounds, estimate=True, print_iter=False)
    if base_pose['estimate']['dist_err_mean'] > 20:
        print("Base estimate error > 20 mm. Select another start point")
        print(json.dumps(base_pose, indent=4))
        return

    print(json.dumps(base_pose, indent=4))

    eulerBase = base_pose["result"][3:]
    transBase = base_pose['result'][:3]
    res = robot.optimize(qs, points, eulerBase, transBase, estimate=True, print_iter=True)
    est = res['estimate']
    print(json.dumps(res, indent=4))

    draw_disposal_data(est, points)

    # print(robot.estimate_fk(qs, points_fk))
    # draw_raw_sensor_data(points)

if __name__=="__main__":
    main()