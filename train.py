import deepxde as dde
import numpy as np
import pandas as pd
from deepxde.backend import pytorch
from argparse import ArgumentParser
import os
from datetime import datetime
import random

parser = ArgumentParser()
parser.add_argument('--lam-load', type=float, default=7.0)
parser.add_argument('--mu', type=float, default=4.0)
parser.add_argument('--speckles', type=int, default=40)
parser.add_argument('--epochs', type=int, default=10000)
parser.add_argument('--fp-resolution', type=int, default=400)
parser.add_argument('--bc-resolution', type=int, default=200)

def main(args, save_path):
    lam_load = dde.Variable(args.lam_load)
    mu = dde.Variable(args.mu)

    force_data = pd.read_csv('cook-data/force-data/force.csv')
    time_list = force_data['Time']
    force_list = force_data['Force']
    time_steps = len(time_list)

    x_force_list = []
    y_force_list = []
    num_points_force = args.fp_resolution
    for i in range(num_points_force):
        x = (np.random.random()*4.0 + 44.0)
        y_max = (x/3) + 44.0
        y_min = (11*x/12)
        y = (np.random.random() * (y_max - y_min)) + y_min
        x_force_list.append(x)
        y_force_list.append(y)
    x_force_list = np.array(x_force_list)
    y_force_list = np.array(y_force_list)

    points_force = []
    value_force = []
    for i in range(time_steps):
        t = time_list[i]
        force = force_list[i]
        for j in range(len(x_force_list)):
            points_force.append([x_force_list[j],y_force_list[j],t])
            value_force.append([force])
    points_force = np.array(points_force)
    value_force = np.array(value_force)

    num_points_upper_lower = args.bc_resolution
    upper_lower_list_x = np.random.rand(num_points_upper_lower) * 48.0
    upper_list_y = upper_lower_list_x/3 + 44.0
    lower_list_y = 11*upper_lower_list_x/12
    points_upper_lower = []
    value_upper_lower = []
    for i in range(time_steps):
        t = time_list[i]
        for j in range(args.bc_resolution):
            points_upper_lower.append([upper_lower_list_x[j], upper_list_y[j], t])
            value_upper_lower.append([0.0])
            points_upper_lower.append([upper_lower_list_x[j], lower_list_y[j], t])
            value_upper_lower.append([0.0])
    points_upper_lower = np.array(points_upper_lower)
    value_upper_lower  = np.array(value_upper_lower)

    data_folder = './cook-data/'
    dic_node = np.load(data_folder + 'node-data/dic_node.npy', allow_pickle=True, encoding='bytes').item()
    dic_disp = []
    for i in range(time_steps):
        disp = np.load(data_folder + 'disp-data/' + str(i) + '_dic_disp.npy', allow_pickle=True, encoding='bytes').item()
        dic_disp.append(disp)
    dic_disp = np.array(dic_disp)

    dic_bc_upper = []
    dic_bc_right = []
    dic_bc_lower = []
    dic_bc_left  = []
    for index in dic_node:
        x = dic_node[index][0]
        y = dic_node[index][1]
        if x == 0:
            dic_bc_left.append([index, x, y])
        if x == 48:
            dic_bc_right.append([index, x, y])
        if y == (x/3) + 44.0:
            dic_bc_upper.append([index, x, y])
        if y == 11*x/12:
            dic_bc_lower.append([index, x, y])
    dic_bc_upper = np.array(dic_bc_upper)
    dic_bc_right = np.array(dic_bc_right)
    dic_bc_lower = np.array(dic_bc_lower)
    dic_bc_left  = np.array(dic_bc_left)
    dic_bc_upper = dic_bc_upper[np.argsort(dic_bc_upper[:,1]),:]
    dic_bc_right = dic_bc_right[np.argsort(dic_bc_right[:,2])[::-1],:]
    dic_bc_lower = dic_bc_lower[np.argsort(dic_bc_lower[:,1])[::-1],:]
    dic_bc_left  = dic_bc_left[np.argsort(dic_bc_left[:,2]),:]

    time_list = np.linspace(0, 1, time_steps)

    x_new_list = []
    y_new_list = []
    coord_old_list = []
    for i in range(time_steps):
        x_new_list.append([])
        y_new_list.append([])
        coord_old_list.append([])
        for ele_list in [dic_bc_upper, dic_bc_right, dic_bc_lower, dic_bc_left]:
            for ele in ele_list:
                index = str(ele[0])
                x_new_list[-1].append(float(ele[1]) + float(dic_disp[i][index][0]) )
                y_new_list[-1].append(float(ele[2]) + float(dic_disp[i][index][1]) )
                coord_old_list[-1].append([float(ele[1]), float(ele[2]), time_list[i]])
    x_new_list = np.array(x_new_list)
    y_new_list = np.array(y_new_list)
    coord_old_list = np.array(coord_old_list)

    net = dde.maps.FNN([3] + [30]*3 + [3], "tanh", "Glorot uniform")
    # net.apply_output_transform(transform)

    spatial_domain = dde.geometry.geometry_2d.Polygon([[0.0, 0.0], [48.0, 44.0], [48.0, 60.0], [0.0, 44.0]])
    temporal_domain = dde.geometry.TimeDomain(0.0, 1.0)
    spatio_temporal_domain = dde.geometry.GeometryXTime(spatial_domain, temporal_domain)

    def pde(x, y):
        F33 = y[:,2:3]
        F11 = dde.grad.jacobian(y, x, i=0, j=0)
        F12 = dde.grad.jacobian(y, x, i=0, j=1)
        F21 = dde.grad.jacobian(y, x, i=1, j=0)
        F22 = dde.grad.jacobian(y, x, i=1, j=1)

        incom = (F22*F11 - F21*F12)*F33 - 1

        I1 = F11* F11 + F21* F21 + F12* F12 + F22* F22 + F33 * F33

        coe0 = 2.0 * 0.5
        coe1 = 2.0 * 2.0/(20.0*lam_load**2)
        coe2 = 2.0 * 3.0 * 11.0/(1050*lam_load**4)
        coe3 = 2.0 * 4.0 * 19.0/(7000*lam_load**6)
        coe4 = 2.0 * 5.0 * 519/(673750*lam_load**8)
        coe  = (coe0 + coe1 * I1 + coe2 * (I1 ** 2) + coe3 * (I1 ** 3) + coe4 * (I1 ** 4))

        p = coe * F33 * F33

        P11 = (-p*F22*F33 + coe*F11)
        P12 = ( p*F21*F33 + coe*F12)
        P21 = ( p*F12*F33 + coe*F21)
        P22 = (-p*F11*F33 + coe*F22)

        BoF1 = dde.grad.jacobian(P11, x, i=0, j=0) + dde.grad.jacobian(P12, x, i=0, j=1)
        BoF2 = dde.grad.jacobian(P21, x, i=0, j=0) + dde.grad.jacobian(P22, x, i=0, j=1)

        return [incom, BoF1, BoF2]

    def boundary_upper_lower(x, on_boundary):
        on_upper = np.isclose(x[1], (x[0]/3) + 44.0)
        on_lower = np.isclose(x[1], 11*x[0]/12)
        on_left  = np.isclose(x[0], 0.0)
        on_right = np.isclose(x[0], 48.0)
        return on_boundary and (on_upper or on_lower) and (not (on_left or on_right))

    def bc_func(x, y, X):
        F33 = y[:,2:3]
        F11 = dde.grad.jacobian(y, x, i=0, j=0)
        F12 = dde.grad.jacobian(y, x, i=0, j=1)
        F21 = dde.grad.jacobian(y, x, i=1, j=0)
        F22 = dde.grad.jacobian(y, x, i=1, j=1)

        incom = (F22*F11 - F21*F12)*F33 - 1

        I1 = F11* F11 + F21* F21 + F12* F12 + F22* F22 + F33 * F33

        coe0 = 2.0 * 0.5
        coe1 = 2.0 * 2.0/(20.0*lam_load**2)
        coe2 = 2.0 * 3.0 * 11.0/(1050*lam_load**4)
        coe3 = 2.0 * 4.0 * 19.0/(7000*lam_load**6)
        coe4 = 2.0 * 5.0 * 519/(673750*lam_load**8)
        coe  = (coe0 + coe1 * I1 + coe2 * (I1 ** 2) + coe3 * (I1 ** 3) + coe4 * (I1 ** 4))

        p = coe * F33 * F33

        P11 = (-p*F22*F33 + coe*F11)
        P12 = ( p*F21*F33 + coe*F12)
        P21 = ( p*F12*F33 + coe*F21)
        P22 = (-p*F11*F33 + coe*F22)

        return pytorch.sqrt(P12**2 + P22**2)

    def func_total_force(x, y, X):
        F33 = y[:,2:3]
        F11 = dde.grad.jacobian(y, x, i=0, j=0)
        F12 = dde.grad.jacobian(y, x, i=0, j=1)
        F21 = dde.grad.jacobian(y, x, i=1, j=0)
        F22 = dde.grad.jacobian(y, x, i=1, j=1)

        incom = F22*F11 - F21*F12 - 1

        I1 = F11* F11 + F21* F21 + F12* F12 + F22* F22 + F33 * F33

        coe0 = 2.0 * 0.5
        coe1 = 2.0 * 2.0/(20.0*lam_load**2)
        coe2 = 2.0 * 3.0 * 11.0/(1050*lam_load**4)
        coe3 = 2.0 * 4.0 * 19.0/(7000*lam_load**6)
        coe4 = 2.0 * 5.0 * 519/(673750*lam_load**8)
        coe  = (coe0 + coe1 * I1 + coe2 * (I1 ** 2) + coe3 * (I1 ** 3) + coe4 * (I1 ** 4))

        p = coe * F33 * F33

        P11 = (-p*F22*F33 + coe*F11)
        P12 = ( p*F21*F33 + coe*F12)
        P21 = ( p*F12*F33 + coe*F21)
        P22 = (-p*F11*F33 + coe*F22)

        return P11 * mu

    # def boundary_speckle_x(x, y, X):
    #     return y[:,0:1]
    # def boundary_speckle_y(x, y, X):
    #     return y[:,1:2]

    bc_upper_lower = dde.RobinBC(points_upper_lower, bc_func, value_upper_lower)
    bc_right_force = dde.PeriodicBC(points_force, func_total_force, value_force, time_steps, num_points_force)
    # bc_speckle_x   = dde.RobinBC(inner_speckle_list,boundary_speckle_x,inner_speckle_list_value_x)
    # bc_speckle_y   = dde.RobinBC(inner_speckle_list,boundary_speckle_y,inner_speckle_list_value_y)

    data = dde.data.TimePDE(spatio_temporal_domain, pde, [bc_upper_lower, bc_right_force], num_domain=20000, num_boundary=0,num_initial=0, num_test=2000)
    model = dde.Model(data, net)

    # weights = [1 for i in range(7)]
    # weights[0]  = 10 # Incom
    # weights[3]  = 10 # UpperLower
    # weights[4]  = 1000 # Speckle-x
    # weights[5]  = 400 # Speckle-y
    # weights[6] = 100

    model.compile("adam", lr=0.001, external_trainable_variables=[lam_load], loss_weights = None)
    variable = dde.callbacks.VariableValue([lam_load], period=100, filename=f'./{save_path}/variable_history',precision=9)

    num_epochs = args.epochs
    model.train(epochs=num_epochs, display_every=1000, callbacks=[variable])
    model.save(f'./{save_path}/pinn-btp-{num_epochs}')

    np.save(f'./{save_path}/train_x.npy', data.train_x)
    np.save(f'./{save_path}/train_x_all.npy', data.train_x_all)
    np.save(f'./{save_path}/train_x_bc.npy', data.train_x_bc)
    np.save(f'./{save_path}/test_x.npy', data.test_x)
    np.save(f'./{save_path}/steps.npy', model.losshistory.steps)
    np.save(f'./{save_path}/loss_train.npy', model.losshistory.loss_train)
    np.save(f'./{save_path}/loss_test.npy', model.losshistory.loss_test)
    # np.save(f'./{save_path}/loss_weights.npy', model.losshistory.loss_weights)

if __name__ == '__main__':
    args = parser.parse_args()
    timestamp = datetime.now().strftime("%d-%m-%y-%H:%M:%S")
    save_path = f'./{timestamp}-{str(args.epochs)}-{str(args.lam_load)}-{str(args.mu)}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    main(args=args, save_path=save_path)
