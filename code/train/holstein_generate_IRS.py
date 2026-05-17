# -*- coding: utf-8 -*-
"""

This script used to generate the IRs features from neighbor representation

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
import itertools
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

def Changebasis(neighbor_layers,neighbor_type):  # This function is used to generate the IRs from input Qs. Something like eq.66 and 67 did. The center is already known as A1.
    A1=[]
    A2=[]
    B1=[]
    B2=[]
    E=[]
    for i in range(0,len(neighbor_layers)):
        length = len(neighbor_layers[i])
        if (length ==1):
            A1.append(neighbor_layers[i][0])
        elif (length == 4 and neighbor_type[i] == 1):
            A1.append(neighbor_layers[i][0]+ neighbor_layers[i][1]+neighbor_layers[i][2]+neighbor_layers[i][3])
            B1.append(neighbor_layers[i][0]- neighbor_layers[i][1]+neighbor_layers[i][2]-neighbor_layers[i][3])
            fE1=neighbor_layers[i][0]- neighbor_layers[i][2]
            fE2=neighbor_layers[i][1] - neighbor_layers[i][3]
         
            tempE = []
            tempE.append(fE1)
            tempE.append(fE2)
            
            E.append(tempE)
            
        elif (length == 4 and neighbor_type[i] == 2):
            A1.append(neighbor_layers[i][0]+ neighbor_layers[i][1]+neighbor_layers[i][2]+neighbor_layers[i][3])
            #B1.append(neighbor_layers[i][0]- neighbor_layers[i][1]+neighbor_layers[i][2]-neighbor_layers[i][3])
            B2.append(neighbor_layers[i][0]- neighbor_layers[i][1]+neighbor_layers[i][2]-neighbor_layers[i][3])
            fE1=neighbor_layers[i][0] - neighbor_layers[i][1] - neighbor_layers[i][2] + neighbor_layers[i][3]  # check this! neighbor_layers[i][0] - neighbor_layers[i][1] - neighbor_layers[i][2] + neighbor_layers[i][3]
            fE2=neighbor_layers[i][0] + neighbor_layers[i][1] - neighbor_layers[i][2] - neighbor_layers[i][3]  # check this! neighbor_layers[i][0] - neighbor_layers[i][1] + neighbor_layers[i][2] - neighbor_layers[i][3]
         
            tempE = []
            tempE.append(fE1)
            tempE.append(fE2)
            
            E.append(tempE)
            
        elif (length == 8):  # We have 8 neighbours around the center, so we will use the following to generate IRs. 
            A1.append(neighbor_layers[i][0]+ neighbor_layers[i][1]+neighbor_layers[i][2]+neighbor_layers[i][3]+neighbor_layers[i][4]+ neighbor_layers[i][5]+neighbor_layers[i][6]+neighbor_layers[i][7])
            A2.append(neighbor_layers[i][0]- neighbor_layers[i][1]+neighbor_layers[i][2]-neighbor_layers[i][3]+neighbor_layers[i][4]- neighbor_layers[i][5]+neighbor_layers[i][6]-neighbor_layers[i][7])
            
            B1.append(neighbor_layers[i][0]- neighbor_layers[i][1]-neighbor_layers[i][2]+neighbor_layers[i][3]+neighbor_layers[i][4]- neighbor_layers[i][5]-neighbor_layers[i][6]+neighbor_layers[i][7])
            B2.append(neighbor_layers[i][0]+ neighbor_layers[i][1]-neighbor_layers[i][2]-neighbor_layers[i][3]+neighbor_layers[i][4]+ neighbor_layers[i][5]-neighbor_layers[i][6]-neighbor_layers[i][7])
            fE1=neighbor_layers[i][0] + neighbor_layers[i][7] - neighbor_layers[i][4] - neighbor_layers[i][3]
            fE2=neighbor_layers[i][2] - neighbor_layers[i][6] - neighbor_layers[i][5] + neighbor_layers[i][1]
            tempE = []
            tempE.append(fE1)
            tempE.append(fE2)
            E.append(tempE)
            fE1=neighbor_layers[i][1] + neighbor_layers[i][6] - neighbor_layers[i][5] - neighbor_layers[i][2]
            fE2=neighbor_layers[i][0] + neighbor_layers[i][3] - neighbor_layers[i][7] - neighbor_layers[i][4]
            tempE = []
            tempE.append(fE1)
            tempE.append(fE2)
            E.append(tempE)
    return A1,A2,B1,B2,E   # return the values of each of IRs.

def rotation_90 (input_Q_single_layer):
    if(len(input_Q_single_layer) == 8):
        rotation_90_matrix = np.matrix("0,0,0,0,0,0,1,0;0,0,0,0,0,0,0,1;1,0,0,0,0,0,0,0;0,1,0,0,0,0,0,0;0,0,1,0,0,0,0,0;0,0,0,1,0,0,0,0;0,0,0,0,1,0,0,0;0,0,0,0,0,1,0,0")
    else:
        rotation_90_matrix = np.matrix("0,0,0,1;1,0,0,0;0,1,0,0;0,0,1,0")
    
    return rotation_90_matrix * np.asmatrix(input_Q_single_layer).transpose()

def rotation_180 (input_Q_single_layer):
    if(len(input_Q_single_layer) == 8):
        rotation_180_matrix = np.matrix("0,0,0,0,1,0,0,0;0,0,0,0,0,1,0,0;0,0,0,0,0,0,1,0;0,0,0,0,0,0,0,1;1,0,0,0,0,0,0,0;0,1,0,0,0,0,0,0;0,0,1,0,0,0,0,0;0,0,0,1,0,0,0,0")
    else:
        rotation_180_matrix = np.matrix("0,0,1,0;0,0,0,1;1,0,0,0;0,1,0,0")
    return rotation_180_matrix * np.asmatrix(input_Q_single_layer).transpose()

def rotation_270 (input_Q_single_layer):
    if(len(input_Q_single_layer) == 8):
        rotation_270_matrix = np.matrix("0,0,1,0,0,0,0,0;0,0,0,1,0,0,0,0;0,0,0,0,1,0,0,0;0,0,0,0,0,1,0,0;0,0,0,0,0,0,1,0;0,0,0,0,0,0,0,1;1,0,0,0,0,0,0,0;0,1,0,0,0,0,0,0")
    else:
        rotation_270_matrix = np.matrix("0,1,0,0;0,0,1,0;0,0,0,1;1,0,0,0")
    return rotation_270_matrix * np.asmatrix(input_Q_single_layer).transpose()


def reflection_p (input_Q_single_layer, type_4):
    if(len(input_Q_single_layer) == 8):
        reflection_p_matrix = np.matrix("0,0,0,1,0,0,0,0;0,0,1,0,0,0,0,0;0,1,0,0,0,0,0,0;1,0,0,0,0,0,0,0;0,0,0,0,0,0,0,1;0,0,0,0,0,0,1,0;0,0,0,0,0,1,0,0;0,0,0,0,1,0,0,0")
    elif (type_4 == 1):
        reflection_p_matrix = np.matrix("0,0,1,0;0,1,0,0;1,0,0,0;0,0,0,1")
    else:
        reflection_p_matrix = np.matrix("0,1,0,0;1,0,0,0;0,0,0,1;0,0,1,0")
    return reflection_p_matrix * np.asmatrix(input_Q_single_layer).transpose()

def reflection_h (input_Q_single_layer, type_4):
    if(len(input_Q_single_layer) == 8):
        reflection_h_matrix = np.matrix("0,0,0,0,0,0,0,1;0,0,0,0,0,0,1,0;0,0,0,0,0,1,0,0;0,0,0,0,1,0,0,0;0,0,0,1,0,0,0,0;0,0,1,0,0,0,0,0;0,1,0,0,0,0,0,0;1,0,0,0,0,0,0,0")
    elif (type_4 == 1):
        reflection_h_matrix = np.matrix("1,0,0,0;0,0,0,1;0,0,1,0;0,1,0,0")
    else:
        reflection_h_matrix = np.matrix("0,0,0,1;0,0,1,0;0,1,0,0;1,0,0,0")
    return reflection_h_matrix * np.asmatrix(input_Q_single_layer).transpose()

def reflection_diagonal_45 (input_Q_single_layer, type_4):
    if(len(input_Q_single_layer) == 8):
        reflection_diagonal_45_matrix = np.matrix("0,1,0,0,0,0,0,0;1,0,0,0,0,0,0,0;0,0,0,0,0,0,0,1;0,0,0,0,0,0,1,0;0,0,0,0,0,1,0,0;0,0,0,0,1,0,0,0;0,0,0,1,0,0,0,0;0,0,1,0,0,0,0,0")
    elif (type_4 == 1):
        reflection_diagonal_45_matrix = np.matrix("0,1,0,0;1,0,0,0;0,0,0,1;0,0,1,0")
    else:
        reflection_diagonal_45_matrix = np.matrix("1,0,0,0;0,0,0,1;0,0,1,0;0,1,0,0")
    return reflection_diagonal_45_matrix * np.asmatrix(input_Q_single_layer).transpose()

def reflection_diagonal_135 (input_Q_single_layer, type_4):
    if(len(input_Q_single_layer) == 8):
        reflection_diagonal_135_matrix = np.matrix("0,0,0,0,0,1,0,0;0,0,0,0,1,0,0,0;0,0,0,1,0,0,0,0;0,0,1,0,0,0,0,0;0,1,0,0,0,0,0,0;1,0,0,0,0,0,0,0;0,0,0,0,0,0,0,1;0,0,0,0,0,0,1,0")
    elif (type_4 == 1):
        reflection_diagonal_135_matrix = np.matrix("0,0,0,1;0,0,1,0;0,1,0,0;1,0,0,0")
    else:
        reflection_diagonal_135_matrix = np.matrix("0,0,1,0;0,1,0,0;1,0,0,0;0,0,0,1")
    return reflection_diagonal_135_matrix * np.asmatrix(input_Q_single_layer).transpose()


def FindIRs_after_transformation(input_Q,neighbor_type): #This function is to generate the input nodes (IRs) After the corresponding D4 symmetry transformation
    all_nodes = []
    input_after_transformation_90 = []
    input_after_transformation_180 = []
    input_after_transformation_270 = []
    input_after_transformation_reflection_p = []
    input_after_transformation_reflection_h= []
    input_after_transformation_diagonal_45  = []
    input_after_transformation_diagonal_135  = []
    for i in range(0,len(input_Q)):
        type_4 = neighbor_type[i]
        input_Q_single_layer = input_Q[i]
        #print("This is input_Q: ", input_Q_single_layer)
        if (len(input_Q_single_layer) == 1):
            input_after_transformation_90.append(input_Q_single_layer)
            input_after_transformation_180.append(input_Q_single_layer)
            input_after_transformation_270.append(input_Q_single_layer)
            input_after_transformation_reflection_p.append(input_Q_single_layer)
            input_after_transformation_reflection_h.append(input_Q_single_layer)
            input_after_transformation_diagonal_45.append(input_Q_single_layer)
            input_after_transformation_diagonal_135.append(input_Q_single_layer)
        else:
            input_Q_transformed_90 = list(itertools.chain.from_iterable(rotation_90(input_Q_single_layer).tolist()))  #rotate counterclockwise by 90 degrees.
            input_Q_transformed_180 = list(itertools.chain.from_iterable(rotation_180(input_Q_single_layer).tolist()))
            input_Q_transformed_270 = list(itertools.chain.from_iterable(rotation_270(input_Q_single_layer).tolist())) #rotate counterclockwise by 270 degrees.
        
            input_Q_transformed_reflection_p = list(itertools.chain.from_iterable(reflection_p(input_Q_single_layer,type_4).tolist())) # reflection via the line perpendicular to the x axis
            input_Q_transformed_reflection_h = list(itertools.chain.from_iterable(reflection_h(input_Q_single_layer,type_4).tolist()))  # reflection via the line horizontal to the x axis
            input_Q_reflection_diagonal_45 = list(itertools.chain.from_iterable(reflection_diagonal_45(input_Q_single_layer,type_4).tolist())) # reflection via the diagonal line (45 degree to the x axis)
            input_Q_transformed_reflection_diagonal_135 = list(itertools.chain.from_iterable(reflection_diagonal_135(input_Q_single_layer,type_4).tolist()))
            
            input_after_transformation_90.append(input_Q_transformed_90)
            input_after_transformation_180.append(input_Q_transformed_180)
            input_after_transformation_270.append(input_Q_transformed_270)
            input_after_transformation_reflection_p.append(input_Q_transformed_reflection_p)
            input_after_transformation_reflection_h.append(input_Q_transformed_reflection_h)
            input_after_transformation_diagonal_45.append(input_Q_reflection_diagonal_45)
            input_after_transformation_diagonal_135.append(input_Q_transformed_reflection_diagonal_135)
            
    input_nodes_transformed_90 = Changebasis(input_after_transformation_90,neighbor_type)        # A12,A22,B12,B22,E2             
    #A12.insert(0,input_Q_include_center[0])
    #input_nodes_transformed_90 = A12+A22+B12+B22+E2
    all_nodes.append(input_nodes_transformed_90)

    input_nodes_transformed_180 = Changebasis(input_after_transformation_180,neighbor_type)                     
    #A13.insert(0,input_Q_include_center[0])
    #input_nodes_transformed_180 = A13+A23+B13+B23+E3
    all_nodes.append(input_nodes_transformed_180)

    input_nodes_transformed_270 = Changebasis(input_after_transformation_270,neighbor_type)                     
    #A14.insert(0,input_Q_include_center[0])
    #input_nodes_transformed_270 = A14+A24+B14+B24+E4
    all_nodes.append(input_nodes_transformed_270)

    input_nodes_transformed_reflection_p = Changebasis(input_after_transformation_reflection_p,neighbor_type)                     
    #A15.insert(0,input_Q_include_center[0])
    #input_nodes_transformed_reflection_p = A15+A25+B15+B25+E5
    all_nodes.append(input_nodes_transformed_reflection_p)

    input_nodes_transformed_reflection_h = Changebasis(input_after_transformation_reflection_h,neighbor_type)                     
    #A16.insert(0,input_Q_include_center[0])
    #input_nodes_transformed_reflection_h = A16+A26+B16+B26+E6
    all_nodes.append(input_nodes_transformed_reflection_h)

    input_nodes_transformed_diagonal_45 = Changebasis(input_after_transformation_diagonal_45,neighbor_type)                     
    #A17.insert(0,input_Q_include_center[0])
    #input_nodes_transformed_diagonal_45 = A17+A27+B17+B27+E7
    all_nodes.append(input_nodes_transformed_diagonal_45)

    input_nodes_transformed_diagonal_135 = Changebasis(input_after_transformation_diagonal_135,neighbor_type)                     
    #A18.insert(0,input_Q_include_center[0])
    #input_nodes_transformed_diagonal_135 = A18+A28+B18+B28+E8
    all_nodes.append(input_nodes_transformed_diagonal_135)
            
            
    
    return all_nodes   # In order to check each of the transformation, I add all the corresponding input nodes to the list "all_nodes"


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



