
import torch
import torch.utils.data as data
import torch.nn.functional as f
import torch.nn as nn
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
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



def find_inputs(sizes_A1,sizes_A2,sizes_B1,sizes_B2,sizes_E,scaled_A1,scaled_A2,scaled_B1,scaled_B2,scaled_X,scaled_Y):
    partner_A1 = []
    partner_A2 = []
    partner_B1 = []
    partner_B2 = []
    E_E = []
    scalar_E = []
    num_config = scaled_A1.shape[0]
    num_lattice = scaled_A1.shape[1]
    first_layer_input_A1 = torch.zeros([num_config,num_lattice,1]).to(device) # assume A1 to B2 has the same shape till dim =1
    first_layer_input_A2 = torch.zeros([num_config,num_lattice,1]).to(device)
    first_layer_input_B1 = torch.zeros([num_config,num_lattice,1]).to(device)
    first_layer_input_B2 = torch.zeros([num_config,num_lattice,1]).to(device)
    for i in range(0,len(sizes_A1)-1):
        num_partner_for_A1 = 0
        num_partner_for_A2 = 0
        num_partner_for_B1 = 0
        num_partner_for_B2 = 0
        num_E_E =int( ((sizes_E[i] * sizes_E[i]) - sizes_E[i]) / 2)
        num_scalar_E = sizes_A1[i] * sizes_E[i] + sizes_A2[i] * sizes_E[i] + sizes_B1[i] * sizes_E[i] + sizes_B2[i] * sizes_E[i]
        for p in range(0,(sizes_A1[i])):
            for q in range(0,(sizes_A1[i])):
                if (p < q):
                    num_partner_for_A1 = num_partner_for_A1 + 1
                    if (i==0):
                        #first_layer_input_A1.append(scaled_A1[::,::,p] * scaled_A1[::,::,q])
                        first_layer_input_A1 = torch.cat((first_layer_input_A1, (scaled_A1[::,::,p] * scaled_A1[::,::,q]).view(num_config,-1,1)),dim=2)
        for p in range(0,(sizes_A2[i])):
            for q in range(0,(sizes_A2[i])):
                if (p < q):
                    num_partner_for_A1 = num_partner_for_A1 + 1
                    if (i==0):
                        #first_layer_input_A1.append(scaled_A2[::,::,p] * scaled_A2[::,::,q])  
                        first_layer_input_A1 = torch.cat((first_layer_input_A1, (scaled_A2[::,::,p] * scaled_A2[::,::,q]).view(num_config,-1,1)),dim=2)
        for p in range(0,(sizes_B1[i])):
            for q in range(0,(sizes_B1[i])):
                if (p < q):
                    num_partner_for_A1 = num_partner_for_A1 + 1
                    if (i==0):
                        #first_layer_input_A1.append(scaled_B1[::,::,p] * scaled_B1[::,::,q])  
                        first_layer_input_A1 = torch.cat((first_layer_input_A1, (scaled_B1[::,::,p] * scaled_B1[::,::,q]).view(num_config,-1,1)),dim=2)
        for p in range(0,(sizes_B2[i])):
            for q in range(0,(sizes_B2[i])):
                if (p < q):
                    num_partner_for_A1 = num_partner_for_A1 + 1
                    if (i==0):
                        #first_layer_input_A1.append(scaled_B2[::,::,p] * scaled_B2[::,::,q])
                        first_layer_input_A1 = torch.cat((first_layer_input_A1, (scaled_B2[::,::,p] * scaled_B2[::,::,q]).view(num_config,-1,1)),dim=2)
                    
                        
                     
        for p in range (0,(sizes_A1[i])):
            for q in range (0,(sizes_A2[i])):
                num_partner_for_A2 = num_partner_for_A2 + 1
                if (i==0):
                    #first_layer_input_A2.append(scaled_A1[::,::,p] * scaled_A2[::,::,q])
                    first_layer_input_A2 = torch.cat((first_layer_input_A2, (scaled_A1[::,::,p] * scaled_A2[::,::,q]).view(num_config,-1,1)),dim=2)
        for p in range (0,(sizes_B1[i])):
            for q in range (0,(sizes_B2[i])):
                num_partner_for_A2 = num_partner_for_A2 + 1
                if (i==0):
                    #first_layer_input_A2.append(scaled_B1[::,::,p] * scaled_B2[::,::,q])
                    first_layer_input_A2 = torch.cat((first_layer_input_A2, (scaled_B1[::,::,p] * scaled_B2[::,::,q]).view(num_config,-1,1)),dim=2)
                
               
                    
                 
        for p in range (0,(sizes_A1[i])):
            for q in range (0,(sizes_B1[i])):
                num_partner_for_B1 = num_partner_for_B1 + 1
                if (i==0):
                    #first_layer_input_B1.append(scaled_A1[::,::,p] * scaled_B1[::,::,q])
                    first_layer_input_B1 = torch.cat((first_layer_input_B1, (scaled_A1[::,::,p] * scaled_B1[::,::,q]).view(num_config,-1,1)),dim=2)
        for p in range (0,(sizes_A2[i])):
            for q in range (0,(sizes_B2[i])):
                num_partner_for_B1 = num_partner_for_B1 + 1
                if (i==0):
                    #first_layer_input_B1.append(scaled_A2[::,::,p] * scaled_B2[::,::,q])
                    first_layer_input_B1 = torch.cat((first_layer_input_B1, (scaled_A2[::,::,p] * scaled_B2[::,::,q]).view(num_config,-1,1)),dim=2)
                
                
                
             
        for p in range (0,(sizes_A1[i])):
            for q in range (0,(sizes_B2[i])):
                num_partner_for_B2 = num_partner_for_B2 + 1
                if (i==0):
                    #first_layer_input_B2.append(scaled_A1[::,::,p] * scaled_B2[::,::,q])
                    first_layer_input_B2 = torch.cat((first_layer_input_B2, (scaled_A1[::,::,p] * scaled_B2[::,::,q]).view(num_config,-1,1)),dim=2)
        for p in range (0,(sizes_A2[i])):
            for q in range (0,(sizes_B1[i])):
                num_partner_for_B2 = num_partner_for_B2 + 1
                if (i==0):
                    #first_layer_input_B2.append(scaled_A2[::,::,p] * scaled_B1[::,::,q])
                    first_layer_input_B2 = torch.cat((first_layer_input_B2, (scaled_A2[::,::,p] * scaled_B1[::,::,q]).view(num_config,-1,1)),dim=2)
                    
        partner_A1.append(num_partner_for_A1)
        partner_A2.append(num_partner_for_A2)
        partner_B1.append(num_partner_for_B1)
        partner_B2.append(num_partner_for_B2)
        E_E.append(num_E_E)
        scalar_E.append(num_scalar_E)
        
       
    return first_layer_input_A1[::,::,1:],first_layer_input_A2[::,::,1:],first_layer_input_B1[::,::,1:],first_layer_input_B2[::,::,1:], partner_A1, partner_A2, partner_B1, partner_B2, E_E, scalar_E
    

        
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



def screenANDscale_tensor(IR_tensor):
    tensor_mean = torch.mean(IR_tensor).item()
    tensor_dev = torch.std(IR_tensor).item()
    IR_tensor[IR_tensor > (tensor_mean + 3*tensor_dev)] = tensor_mean + 3*tensor_dev
    IR_tensor[IR_tensor < (tensor_mean - 3*tensor_dev)] = tensor_mean - 3*tensor_dev
    
    tensor_max = torch.max(IR_tensor)
    tensor_min = torch.min(IR_tensor)
    return 2 * (IR_tensor - tensor_min) / (tensor_max - tensor_min) - 1
    
    
def screenANDscale_tensor2(IR_tensor, tensor_max):
    tensor_min = -1 * tensor_max
    return 2 * (IR_tensor - tensor_min) / (tensor_max - tensor_min) - 1
###########################################################################################
sizes_A1 = [9,32,8,4,1] 
sizes_A2 = [3,32,8,4,0] 
sizes_B1 = [6,32,8,4,0]
sizes_B2 = [5,32,8,4,0] 
sizes_E =  [11,32,8,4,0] 
torch.manual_seed(7856677)

from_bench_mark = False
epoch_start = 0
duration= 30000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
neighbor_ramp = 8
Lsize = 40
lattice_num = Lsize * Lsize
num_site = Lsize * Lsize
path_save_file ='./model/'
dir_train = "./scan/"
dir_out = "./model/"





A1 = torch.load(dir_train+"IR_A1.pt").to(device)  # directly load the IR features
A2 = torch.load(dir_train+"IR_A2.pt").to(device)
B1 = torch.load(dir_train+"IR_B1.pt").to(device)
B2 = torch.load(dir_train+"IR_B2.pt").to(device)
X = torch.load(dir_train+"IR_X.pt").to(device)
Y = torch.load(dir_train+"IR_Y.pt").to(device)
force = torch.load(dir_train+"force_tensor.pt").to(device)


parameters=[]
parameters.append(max(np.abs(torch.max(A1).item()), np.abs(torch.min(A1).item())))
parameters.append(max(np.abs(torch.max(A2).item()), np.abs(torch.min(A2).item())))
parameters.append(max(np.abs(torch.max(B1).item()), np.abs(torch.min(B1).item())))
parameters.append(max(np.abs(torch.max(B2).item()), np.abs(torch.min(B2).item())))
parameters.append(max(np.abs(torch.max(X).item()), np.abs(torch.min(X).item())))
parameters.append(max(np.abs(torch.max(Y).item()), np.abs(torch.min(Y).item())))

with open('para.csv','a',newline='') as file:
    writer = csv.writer(file)
    writer.writerows([parameters])   

indices = []

for i in range(0, A1.shape[0]):
    if ((i+1) % 3 != 0):
        indices.append(i)
    
indices = torch.tensor(indices).to(device)

dim = 0

#Read IRs as input

scaled_A1 = screenANDscale_tensor2(torch.index_select(A1,dim,indices),parameters[0]).float().to(device)
scaled_A2 = screenANDscale_tensor2(torch.index_select(A2,dim,indices),parameters[1]).float().to(device)
scaled_B1 = screenANDscale_tensor2(torch.index_select(B1,dim,indices),parameters[2]).float().to(device)
scaled_B2 = screenANDscale_tensor2(torch.index_select(B2,dim,indices),parameters[3]).float().to(device)
scaled_X = screenANDscale_tensor2(torch.index_select(X,dim,indices),parameters[4]).float().to(device)
scaled_Y = screenANDscale_tensor2(torch.index_select(Y,dim,indices),parameters[5]).float().to(device)
force_train = torch.index_select(force,dim,indices).float().to(device)
scaled_force = force_train.view(force_train.shape[0], -1, 1).float().to(device)


first_layer_input_A1,first_layer_input_A2,first_layer_input_B1,first_layer_input_B2, partner_A1, partner_A2, partner_B1, partner_B2, E_E, scalar_E = find_inputs(sizes_A1,sizes_A2,sizes_B1,sizes_B2,sizes_E,scaled_A1,scaled_A2,scaled_B1,scaled_B2,scaled_X,scaled_Y)

num_A1 = []
num_A2 = []
num_B1 = []
num_B2 = []
num_E = []
print("######These information should be pasted to model script######")
for i in range(0,len(sizes_A1) - 1):
    num_A1.append(sizes_A1[i]+ partner_A1[i] + E_E[i])
    print(sizes_A1[i], partner_A1[i], E_E[i])
    num_A2.append(sizes_A2[i]+ partner_A2[i] + E_E[i])
    num_B1.append(sizes_B1[i]+ partner_B1[i] + E_E[i])
    num_B2.append(sizes_B2[i]+ partner_B2[i] + E_E[i])
    num_E.append(sizes_E[i]+ scalar_E[i])
print("num_A1=",num_A1)
print("num_A2=",num_A2)
print("num_B1=",num_B1)
print("num_B2=",num_B2)
print("num_E=",num_E)


'''

torch_data_set = data.TensorDataset(scaled_A1,scaled_A2,scaled_B1,scaled_B2,scaled_X,scaled_Y,scaled_force)
loader = data.DataLoader(dataset=torch_data_set, batch_size=1, shuffle=True)

########################START OF TRAINING#######################################################



net = Net().to(device)
learning_rate =0.001
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=5, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-6, verbose=True)
loss_func = torch.nn.MSELoss()




saved_loss = 1000
for epoch in range(epoch_start,epoch_start+duration):  #epoch_start,epoch_start+duration
    loss_perstep = []
    for step, (scaled_A1,scaled_A2,scaled_B1,scaled_B2,scaled_X,scaled_Y,scaled_force) in enumerate(loader):
        out_fea = scaled_force[0]
        optimizer.zero_grad()
        result_A1, result_A2, result_B1, result_B2, result_X, result_Y = net(scaled_A1[0],scaled_A2[0],scaled_B1[0],scaled_B2[0],scaled_X[0],scaled_Y[0])
        loss = loss_func(out_fea,result_A1)
        loss_perstep.append(loss.item())
        loss.backward()
        optimizer.step()
    average_loss = sum(loss_perstep) / len(loss_perstep)
    if (average_loss < saved_loss ):                                  
        torch.save({'model':net.state_dict(),'optimizer':optimizer.state_dict()},"./model/model_status_best"+".pt")
        saved_loss = average_loss
        best_epoch = []
        best_epoch.append(epoch)
        with open('para.csv','a',newline='') as file:
            writer = csv.writer(file)
            writer.writerows([best_epoch])    
    if ((epoch+1) % 500 == 0):
        torch.save({'model':net.state_dict(),'optimizer':optimizer.state_dict()},"./model/model_status"+str(epoch)+".pt")
    scheduler.step(average_loss)
    info_per_epoch=[]
    info_per_epoch.append(epoch)
    info_per_epoch.append(optimizer.param_groups[-1]['lr'])
    info_per_epoch.append(average_loss)
    with open('loss.csv','a',newline='') as file:
        writer = csv.writer(file)
        writer.writerows([info_per_epoch])    



'''


