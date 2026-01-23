# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 19:37:08 2023
This script used to generate the IRs features from neighbor representation
@author: 14026
"""

import torch
import torch.utils.data as data
import torch.nn.functional as f
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import itertools
import time
import numpy as np
import math
import os
import csv
import torch.optim.lr_scheduler as lr_scheduler  
from read_neighbor import read_neighbor_list
from generate_feature import generate_feature_mat
from generate_feature import generate_Q_feature
from symmetry_functionsV2 import *
from holstein_model_V3 import Net
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def generate_Q_and_its_neighbors_feature(step,Q):  
    Q_feature_list_one_configuration = []
    for i in range(0,lattice_num):
        Q_feature_list_one_configuration_one_lattice = []
        count_all = 0
        for j in range(0,len(num_of_nodes_neighbor_each_layer)):
            count =0
            num_of_points = num_of_nodes_neighbor_each_layer[j]
            Q_temp = []
            while (count < num_of_points):
                Q_temp.append(Q[0,neighbor_2d[i,count_all]].item())
                count = count+1
                count_all = count_all + 1
            Q_feature_list_one_configuration_one_lattice.append(Q_temp)
            
        Q_feature_list_one_configuration.append(Q_feature_list_one_configuration_one_lattice)
    return Q_feature_list_one_configuration  #the q value of the center point, the qs value of the 4 neighbors around it, the qs value of the 4 neighbors on the next layer...


################################################################################################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
neighbor_ramp = 8
Lsize = 40
lattice_num = Lsize * Lsize
num_site = Lsize * Lsize
path_save_file ='./model/'
dir_train = "./training_data/"
dir_out = "./model/"


torch.set_default_dtype(torch.float32)

neighbor_2d, num_input, neighbor_type = read_neighbor_list("./neighbor/neighbor_" + str(Lsize) + "_ramp" + str(neighbor_ramp) + ".csv", neighbor_ramp, Lsize)

num_neighbor_layers = neighbor_type.shape[0]
num_of_nodes_neighbor_each_layer = []

for i in range(0,num_neighbor_layers):
    if (neighbor_type[i].item() == 0):
        num_of_nodes_neighbor_each_layer.append(1)
    elif (neighbor_type[i].item() == 1 or neighbor_type[i].item() == 2):
        num_of_nodes_neighbor_each_layer.append(4)
    elif (neighbor_type[i].item() == 3):
        num_of_nodes_neighbor_each_layer.append(8)
    else:
        print("I missed someting!")
 
print(neighbor_2d[0])
      
Q_tensor_1 = torch.load(dir_train + "Q_random_200.pt").to(device)
force_tensor_1 = torch.load(dir_train + "force_random_200.pt").to(device)
Q_tensor_2 = torch.load(dir_train + "Q_quench_400.pt").to(device)
force_tensor_2 = torch.load(dir_train + "force_quench_400.pt").to(device)
Q_tensor=torch.cat((Q_tensor_1, Q_tensor_2))
force_tensor= torch.cat((force_tensor_1, force_tensor_2))

num_config, lattice_num = Q_tensor.shape

num_of_input = num_config * lattice_num

Q_feature_list = []
F_feature_list = []


torch_data_set = data.TensorDataset(Q_tensor, force_tensor)
loader = data.DataLoader(dataset=torch_data_set, batch_size=1, shuffle=False)

IR_A1 = []
IR_A2 = []
IR_B1 = []
IR_B2 = []
IR_X = []
IR_Y = []


for step, (Q, F) in enumerate(loader):
    Q_feature_list  = generate_Q_and_its_neighbors_feature(step,Q)  # Q_feature_list contains the neighbors of each lattice, shape [1600,45]
    print(step)
    IR_A1_one = []
    IR_A2_one = []
    IR_B1_one = []
    IR_B2_one = []
    IR_X_one = []
    IR_Y_one = []
    for i in range(0,len(Q_feature_list)):
        A1,A2,B1,B2,E = Changebasis(Q_feature_list[i],neighbor_type)
        flat_E = np.array(E, dtype=object).flatten().tolist()
        if (step == 0 and i == 0):
            print("number of A1: ", len(A1))
            print("number of A2: ", len(A2))
            print("number of B1: ", len(B1))
            print("number of B2: ", len(B2))
            print("number of E: ", len(E))
            input_node_sample = (A1+A2+B1+B2+E)
        
        IR_A1_one.append(A1)
        IR_A2_one.append(A2)
        IR_B1_one.append(B1)
        IR_B2_one.append(B2)
        temp_x = []
        temp_y =[]
        for j in range(0,len(E)):
            temp_x.append(E[j][0])
            temp_y.append(E[j][1])
        IR_X_one.append(temp_x)
        IR_Y_one.append(temp_y)
    
    IR_A1.append(IR_A1_one)
    IR_A2.append(IR_A2_one)
    IR_B1.append(IR_B1_one)
    IR_B2.append(IR_B2_one)
    IR_X.append(IR_X_one)
    IR_Y.append(IR_Y_one)
    

IR_A1_tensor = torch.tensor(IR_A1).float().to(device)
IR_A2_tensor = torch.tensor(IR_A2).float().to(device)
IR_B1_tensor = torch.tensor(IR_B1).float().to(device)
IR_B2_tensor = torch.tensor(IR_B2).float().to(device)
IR_X_tensor = torch.tensor(IR_X).float().to(device)
IR_Y_tensor = torch.tensor(IR_Y).float().to(device)

torch.save(IR_A1_tensor, path_save_file+'IR_A1'+'.pt')
torch.save(IR_A2_tensor, path_save_file+'IR_A2'+'.pt')
torch.save(IR_B1_tensor, path_save_file+'IR_B1'+'.pt')
torch.save(IR_B2_tensor, path_save_file+'IR_B2'+'.pt')
torch.save(IR_X_tensor, path_save_file+'IR_X'+'.pt')
torch.save(IR_Y_tensor, path_save_file+'IR_Y'+'.pt')
torch.save(force_tensor, path_save_file+'force_tensor'+'.pt')



