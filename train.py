import deepxde as dde
import numpy as np
import pandas as pd
from deepxde.backend import pytorch
from argparse import ArgumentParser
import random

def main():
    parser = ArgumentParser()
    parser.add_argument('--noise', type=float, default=32)
    parser.add_argument('--speckles', type=int, default=40)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--fp-resolution', type=int, default=400)
    parser.add_argument('--bc-resolution', type=int, default=200)
    args = parser.parse_args()

    force_data = pd.read_csv('cook-data/force.csv')
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

    data_folder = '../cook-data/'
    dic_node = np.load(data_folder + 'dic_node.npy', allow_pickle=True, encoding='bytes').item()
    dic_disp = []
    for i in range(time_steps):
        disp = np.load(data_folder + str(i) + '_dic_disp.npy', allow_pickle=True, encoding='bytes').item()
        dic_disp.append(disp)

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
    # Reorder
    #
    # information of the internal point (outside of the inhomogeiouty)
    x_new_list = []
    y_new_list = []
    coord_old_list = []
    #
    # information of the internal point (on the boundary of the inhomogeiouty)
    x_new_inner_list = []
    y_new_inner_list = []
    coord_old_inner_list = []
    #
    for i in range(num_time):
        x_new_list.append([])
        y_new_list.append([])
        x_new_inner_list.append([])
        y_new_inner_list.append([])
        coord_old_list.append([])
        coord_old_inner_list.append([])
        for ele_list in [dic_bc_upper,dic_bc_right,dic_bc_lower,dic_bc_left]:
            for ele in ele_list:
                index = int(ele[0])
                x_new_list[-1].append( ele[1] + dic_displacement[i][index][0] ) 
                y_new_list[-1].append( ele[2] + dic_displacement[i][index][1] )
                coord_old_list[-1].append([ele[1],ele[2],time_list[i]])
        #
        for ele in dic_bc_inner:
            index = int(ele[0])
            # add the error
            err_x = noise * np.random.random() * (float(np.random.random()>0.5) * 2 - 1.0) + 1.0
            err_y = noise * np.random.random() * (float(np.random.random()>0.5) * 2 - 1.0) + 1.0
            #
            x_new_inner_list[-1].append( ele[1] + dic_displacement[i][index][0] * err_x ) 
            y_new_inner_list[-1].append( ele[2] + dic_displacement[i][index][1] * err_y )
            #
            coord_old_inner_list[-1].append([ele[1],ele[2],time_list[i]])
    #
    x_new_list = np.array(x_new_list)
    y_new_list = np.array(y_new_list)
    coord_old_list = np.array(coord_old_list)
    x_new_inner_list = np.array(x_new_inner_list)
    y_new_inner_list = np.array(y_new_inner_list)
    coord_old_inner_list = np.array(coord_old_inner_list)
    # 
    dim = coord_old_inner_list.shape
    inner_speckle_list = coord_old_inner_list.reshape([dim[0]*dim[1],3])
    #
    dim = x_new_inner_list.shape
    inner_speckle_list_value_x = x_new_inner_list.reshape([dim[0]*dim[1],1])
    dim = y_new_inner_list.shape
    inner_speckle_list_value_y = y_new_inner_list.reshape([dim[0]*dim[1],1])







    pass

if __name__ == '__main__':
    main()