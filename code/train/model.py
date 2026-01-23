import torch
import torch.nn.functional as f
import torch.utils.data as data
import torch.nn as nn
import math


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
lattice_num = 40 * 40 # number of lattice sites

sizes_A1 = [9,32,8,4,1] #Number of A1 nodes at each layer
sizes_A2 = [3,32,8,4,0] #Number of A2 nodes at each layer
sizes_B1 = [6,32,8,4,0] #Number of B1 nodes at each layer
sizes_B2 = [5,32,8,4,0] #Number of B2 nodes at each layer
sizes_E =  [11,32,8,4,0]   #Number of doublet nodes at each layer

#The following number of nodes are read from the the training script
num_A1= [128, 2512, 148, 34] #Number of A1 nodes at each auxiliary layer
num_A2= [115, 2576, 164, 42] #Number of A2 nodes at each auxiliary layer
num_B1= [130, 2576, 164, 42] #Number of B1 nodes at each auxiliary layer
num_B2= [123, 2576, 164, 42] #Number of B2 nodes at each auxiliary layer
num_E= [264, 4128, 264, 68] #Number of doublet nodes at each auxiliary layer

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.A1_0 = torch.nn.Linear(num_A1[0], sizes_A1[1], bias =False)  
        self.A1_1 = torch.nn.Linear(num_A1[1], sizes_A1[2], bias =False)
        self.A1_2 = torch.nn.Linear(num_A1[2], sizes_A1[3], bias =False)
        self.A1_3 = torch.nn.Linear(num_A1[3], sizes_A1[4], bias =False)
        #self.A1_4 = torch.nn.Linear(num_A1[4], sizes_A1[5], bias =False)
        
        
        self.A2_0 = torch.nn.Linear(num_A2[0], sizes_A2[1], bias =False)
        self.A2_1 = torch.nn.Linear(num_A2[1], sizes_A2[2], bias =False)
        self.A2_2 = torch.nn.Linear(num_A2[2], sizes_A2[3], bias =False)
        self.A2_3 = torch.nn.Linear(num_A2[3], sizes_A2[4], bias =False)
        #self.A2_4 = torch.nn.Linear(num_A2[4], sizes_A2[5], bias =False)
        
        
        self.B1_0 = torch.nn.Linear(num_B1[0], sizes_B1[1], bias =False)
        self.B1_1 = torch.nn.Linear(num_B1[1], sizes_B1[2], bias =False)
        self.B1_2 = torch.nn.Linear(num_B1[2], sizes_B1[3], bias =False)
        self.B1_3 = torch.nn.Linear(num_B1[3], sizes_B1[4], bias =False)
        #self.B1_4 = torch.nn.Linear(num_B1[4], sizes_B1[5], bias =False)
        
        
        self.B2_0 = torch.nn.Linear(num_B2[0], sizes_B2[1], bias =False)
        self.B2_1 = torch.nn.Linear(num_B2[1], sizes_B2[2], bias =False)
        self.B2_2 = torch.nn.Linear(num_B2[2], sizes_B2[3], bias =False)
        self.B2_3 = torch.nn.Linear(num_B2[3], sizes_B2[4], bias =False)
        #self.B2_4 = torch.nn.Linear(num_B2[4], sizes_B2[5], bias =False)
        
        
        self.E_0 = torch.nn.Linear(num_E[0], sizes_E[1], bias =False)
        self.E_1 = torch.nn.Linear(num_E[1], sizes_E[2], bias =False)
        self.E_2 = torch.nn.Linear(num_E[2], sizes_E[3], bias =False)
        self.E_3 = torch.nn.Linear(num_E[3], sizes_E[4], bias =False)
        #self.E_4 = torch.nn.Linear(num_E[4], sizes_E[5], bias =False)
        
        
        
        nn.init.xavier_uniform_(self.A1_0.weight, nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(self.A1_1.weight, nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(self.A1_2.weight, nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(self.A1_3.weight, nn.init.calculate_gain('sigmoid'))
        #nn.init.xavier_uniform_(self.A1_4.weight, nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(self.A2_0.weight, nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(self.A2_1.weight, nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(self.A2_2.weight, nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(self.A2_3.weight, nn.init.calculate_gain('sigmoid'))
        #nn.init.xavier_uniform_(self.A2_4.weight, nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(self.B1_0.weight, nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(self.B1_1.weight, nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(self.B1_2.weight, nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(self.B1_3.weight, nn.init.calculate_gain('sigmoid'))
        #nn.init.xavier_uniform_(self.B1_4.weight, nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(self.B2_0.weight, nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(self.B2_1.weight, nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(self.B2_2.weight, nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(self.B2_3.weight, nn.init.calculate_gain('sigmoid'))
        #nn.init.xavier_uniform_(self.B2_4.weight, nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(self.E_0.weight, nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(self.E_1.weight, nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(self.E_2.weight, nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(self.E_3.weight, nn.init.calculate_gain('sigmoid'))
        #nn.init.xavier_uniform_(self.E_4.weight, nn.init.calculate_gain('sigmoid'))
        
        strv = 1. / math.sqrt(num_A1[0])
        bias_value_A10 = (-2 * strv) * torch.rand(lattice_num,sizes_A1[1]) + strv
        self.A1_0_bias = torch.nn.Parameter(bias_value_A10)
        
        strv = 1. / math.sqrt(num_A1[1])
        bias_value_A11 = (-2 * strv) * torch.rand(lattice_num,sizes_A1[2]) + strv
        self.A1_1_bias = torch.nn.Parameter(bias_value_A11)
        
        strv = 1. / math.sqrt(num_A1[2])
        bias_value_A12 = (-2 * strv) * torch.rand(lattice_num,sizes_A1[3]) + strv
        self.A1_2_bias = torch.nn.Parameter(bias_value_A12)
        
        strv = 1. / math.sqrt(num_A1[3])
        bias_value_A13 = (-2 * strv) * torch.rand(lattice_num,sizes_A1[4]) + strv
        self.A1_3_bias = torch.nn.Parameter(bias_value_A13)
        
        #strv = 1. / math.sqrt(num_A1[4])
        #bias_value_A14 = (-2 * strv) * torch.rand(lattice_num,sizes_A1[5]) + strv
        #self.A1_4_bias = torch.nn.Parameter(bias_value_A14)
        
        strv = 1. / math.sqrt(num_A2[0])
        bias_value_A20 = (-2 * strv) * torch.rand(lattice_num,sizes_A2[1]) + strv
        self.A2_0_bias = torch.nn.Parameter(bias_value_A20)
        
        strv = 1. / math.sqrt(num_A2[1])
        bias_value_A21 = (-2 * strv) * torch.rand(lattice_num,sizes_A2[2]) + strv
        self.A2_1_bias = torch.nn.Parameter(bias_value_A21)
        
        strv = 1. / math.sqrt(num_A2[2])
        bias_value_A22 = (-2 * strv) * torch.rand(lattice_num,sizes_A2[3]) + strv
        self.A2_2_bias = torch.nn.Parameter(bias_value_A22)
        
        strv = 1. / math.sqrt(num_A2[3])
        bias_value_A23 = (-2 * strv) * torch.rand(lattice_num,sizes_A2[4]) + strv
        self.A2_3_bias = torch.nn.Parameter(bias_value_A23)
        
        #strv = 1. / math.sqrt(num_A2[4])
        #bias_value_A24 = (-2 * strv) * torch.rand(lattice_num,sizes_A2[5]) + strv
        #self.A2_4_bias = torch.nn.Parameter(bias_value_A24)
        
        strv = 1. / math.sqrt(num_B1[0])
        bias_value_B10 = (-2 * strv) * torch.rand(lattice_num, sizes_B1[1]) + strv
        self.B1_0_bias = torch.nn.Parameter(bias_value_B10)
        
        strv = 1. / math.sqrt(num_B1[1])
        bias_value_B11 = (-2 * strv) * torch.rand(lattice_num,sizes_B1[2]) + strv
        self.B1_1_bias = torch.nn.Parameter(bias_value_B11)
        
        strv = 1. / math.sqrt(num_B1[2])
        bias_value_B12 = (-2 * strv) * torch.rand(lattice_num,sizes_B1[3]) + strv
        self.B1_2_bias = torch.nn.Parameter(bias_value_B12)
        
        strv = 1. / math.sqrt(num_B1[3])
        bias_value_B13 = (-2 * strv) * torch.rand(lattice_num,sizes_B1[4]) + strv
        self.B1_3_bias = torch.nn.Parameter(bias_value_B13)
        
        #strv = 1. / math.sqrt(num_B1[4])
        #bias_value_B14 = (-2 * strv) * torch.rand(lattice_num,sizes_B1[5]) + strv
        #self.B1_4_bias = torch.nn.Parameter(bias_value_B14)
        
        strv = 1. / math.sqrt(num_B2[0])
        bias_value_B20 = (-2 * strv) * torch.rand(lattice_num, sizes_B2[1]) + strv
        self.B2_0_bias = torch.nn.Parameter(bias_value_B20)
        
        strv = 1. / math.sqrt(num_B2[1])
        bias_value_B21 = (-2 * strv) * torch.rand(lattice_num, sizes_B2[2]) + strv
        self.B2_1_bias = torch.nn.Parameter(bias_value_B21)
        
        strv = 1. / math.sqrt(num_B2[2])
        bias_value_B22 = (-2 * strv) * torch.rand(lattice_num, sizes_B2[3]) + strv
        self.B2_2_bias = torch.nn.Parameter(bias_value_B22)
        
        strv = 1. / math.sqrt(num_B2[3])
        bias_value_B23 = (-2 * strv) * torch.rand(lattice_num, sizes_B2[4]) + strv
        self.B2_3_bias = torch.nn.Parameter(bias_value_B23)
        
        #strv = 1. / math.sqrt(num_B2[4])
        #bias_value_B24 = (-2 * strv) * torch.rand(lattice_num, sizes_B2[5]) + strv
        #self.B2_4_bias = torch.nn.Parameter(bias_value_B24)
        
        strv = 1. / math.sqrt(num_E[0])
        bias_value_E0 = (-2 * strv) * torch.rand(lattice_num, sizes_E[1]) + strv
        self.E_0_bias = torch.nn.Parameter(bias_value_E0)
        
        strv = 1. / math.sqrt(num_E[1])
        bias_value_E1 = (-2 * strv) * torch.rand(lattice_num, sizes_E[2]) + strv
        self.E_1_bias = torch.nn.Parameter(bias_value_E1)
        
        strv = 1. / math.sqrt(num_E[2])
        bias_value_E2 = (-2 * strv) * torch.rand(lattice_num, sizes_E[3]) + strv
        self.E_2_bias = torch.nn.Parameter(bias_value_E2)
        
        strv = 1. / math.sqrt(num_E[3])
        bias_value_E3 = (-2 * strv) * torch.rand(lattice_num, sizes_E[4]) + strv
        self.E_3_bias = torch.nn.Parameter(bias_value_E3)
        
        #strv = 1. / math.sqrt(num_E[4])
        #bias_value_E4 = (-2 * strv) * torch.rand(lattice_num, sizes_E[5]) + strv
        #self.E_4_bias = torch.nn.Parameter(bias_value_E4)
        
        
        
    def forward(self, A1,A2,B1,B2,X,Y):  #ActivationFunction_scalar(y,bias):
        i = 0
        A1,A2,B1,B2,X,Y = self.combine_inputs(A1,A2,B1,B2,X,Y,i)
        A1 = self.A1_0(A1)
        A1 = self.ActivationFunction_scalar(A1, self.A1_0_bias)
        
        A2 = self.A2_0(A2)
        A2 = self.ActivationFunction_scalar(A2, self.A2_0_bias)
        
        B1 = self.B1_0(B1)
        B1 = self.ActivationFunction_scalar(B1, self.B1_0_bias)
        
        B2 = self.B2_0(B2)
        B2 = self.ActivationFunction_scalar(B2, self.B2_0_bias)
        
        X = self.E_0(X)
        Y = self.E_0(Y)
        X, Y = self.ActivationFunction_doubleE(X,Y, self.E_0_bias)
        
        
        
        i = 1
        A1,A2,B1,B2,X,Y = self.combine_inputs(A1,A2,B1,B2,X,Y,i)
        A1 = self.A1_1(A1)
        A1 = self.ActivationFunction_scalar(A1, self.A1_1_bias)
        
        A2 = self.A2_1(A2)
        A2 = self.ActivationFunction_scalar(A2, self.A2_1_bias)
    
        B1 = self.B1_1(B1)
        B1 = self.ActivationFunction_scalar(B1, self.B1_1_bias)
        
        B2 = self.B2_1(B2)
        B2 = self.ActivationFunction_scalar(B2, self.B2_1_bias)
    
        X = self.E_1(X)
        Y = self.E_1(Y)
        X, Y = self.ActivationFunction_doubleE(X,Y, self.E_1_bias)
        
        
        
        i = 2
        A1,A2,B1,B2,X,Y = self.combine_inputs(A1,A2,B1,B2,X,Y,i)
        A1 = self.A1_2(A1)
        A1 = self.ActivationFunction_scalar(A1, self.A1_2_bias)
        
        A2 = self.A2_2(A2)
        A2 = self.ActivationFunction_scalar(A2, self.A2_2_bias)
    
        B1 = self.B1_2(B1)
        B1 = self.ActivationFunction_scalar(B1, self.B1_2_bias)
        
        B2 = self.B2_2(B2)
        B2 = self.ActivationFunction_scalar(B2, self.B2_2_bias)
    
        X = self.E_2(X)
        Y = self.E_2(Y)
        X, Y = self.ActivationFunction_doubleE(X,Y, self.E_2_bias)
        
        
        
        i = 3
        A1,A2,B1,B2,X,Y = self.combine_inputs(A1,A2,B1,B2,X,Y,i)
        A1 = self.A1_3(A1)
        #A1 = self.ActivationFunction_scalar(A1, self.A1_3_bias)
        
        A2 = self.A2_3(A2)
        #A2 = self.ActivationFunction_scalar(A2, self.A2_3_bias)
    
        B1 = self.B1_3(B1)
        #B1 = self.ActivationFunction_scalar(B1, self.B1_3_bias)
        
        B2 = self.B2_3(B2)
        #B2 = self.ActivationFunction_scalar(B2, self.B2_3_bias)
    
        X = self.E_3(X)
        Y = self.E_3(Y)
        #X, Y = self.ActivationFunction_doubleE(X,Y, self.E_3_bias)
        
        
        
        #i = 4
        #A1,A2,B1,B2,X,Y = self.combine_inputs(A1,A2,B1,B2,X,Y,i)
        #A1 = self.A1_4(A1)
        #A1 = self.ActivationFunction_scalar(A1, self.A1_4_bias)
        
        #A2 = self.A2_4(A2)
        #A2 = self.ActivationFunction_scalar(A2, self.A2_4_bias)
    
        #B1 = self.B1_4(B1)
        #B1 = self.ActivationFunction_scalar(B1, self.B1_4_bias)
        
        #B2 = self.B2_4(B2)
        #B2 = self.ActivationFunction_scalar(B2, self.B2_4_bias)
    
        #X = self.E_4(X)
        #Y = self.E_4(Y)
        #X, Y = self.ActivationFunction_doubleE(X,Y, self.E_4_bias)
        
        return A1, A2, B1, B2, X, Y
    
    def combine_inputs(self, A1,A2,B1,B2,X,Y,i):  #,sizes_A1,sizes_A2,sizes_B1,sizes_B2,sizes_E
        filter_A1_A1 = torch.triu(torch.ones(lattice_num,A1.shape[1],A1.shape[1]).to(device), diagonal=1).ge(0.5)
    
        filter_A2_A2 = torch.triu(torch.ones(lattice_num,A2.shape[1],A2.shape[1]).to(device), diagonal=1).ge(0.5)
        filter_B1_B1 = torch.triu(torch.ones(lattice_num,B1.shape[1],B1.shape[1]).to(device), diagonal=1).ge(0.5)
        filter_B2_B2 = torch.triu(torch.ones(lattice_num,B2.shape[1],B2.shape[1]).to(device), diagonal=1).ge(0.5)
        A1_from_A1A1_tensor = torch.masked_select((A1.view(lattice_num,-1,1) @ A1.view(lattice_num,1,-1)).view(lattice_num, A1.shape[1], A1.shape[1]),filter_A1_A1).view(lattice_num,-1)
        A1_from_A2A2_tensor = torch.masked_select((A2.view(lattice_num,-1,1) @ A2.view(lattice_num,1,-1)).view(lattice_num, A2.shape[1], A2.shape[1]),filter_A2_A2).view(lattice_num,-1)
        A1_from_B1B1_tensor = torch.masked_select((B1.view(lattice_num,-1,1) @ B1.view(lattice_num,1,-1)).view(lattice_num, B1.shape[1], B1.shape[1]),filter_B1_B1).view(lattice_num,-1)
        A1_from_B2B2_tensor = torch.masked_select((B2.view(lattice_num,-1,1) @ B2.view(lattice_num,1,-1)).view(lattice_num, B2.shape[1], B2.shape[1]),filter_B2_B2).view(lattice_num,-1)
        A2_from_A1A2_tensor = (A1.view(lattice_num,-1,1) @ A2.view(lattice_num,1,-1)).view(lattice_num,-1)
        A2_from_B1B2_tensor = (B1.view(lattice_num,-1,1) @ B2.view(lattice_num,1,-1)).view(lattice_num,-1)
        B1_from_A1B1_tensor = (A1.view(lattice_num,-1,1) @ B1.view(lattice_num,1,-1)).view(lattice_num,-1)
        B1_from_A2B2_tensor = (A2.view(lattice_num,-1,1) @ B2.view(lattice_num,1,-1)).view(lattice_num,-1)
        B2_from_A1B2_tensor = (A1.view(lattice_num,-1,1) @ B2.view(lattice_num,1,-1)).view(lattice_num,-1)
        B2_from_B1A2_tensor = (B1.view(lattice_num,-1,1) @ A2.view(lattice_num,1,-1)).view(lattice_num,-1)
        
        filter_XY = torch.triu(torch.ones(lattice_num,sizes_E[i],sizes_E[i]).to(device), diagonal=1).ge(0.5).to(device)
        XX_tensor = (X.view(lattice_num,-1,1) @ X.view(lattice_num,1,-1)).view(lattice_num, X.shape[1], X.shape[1])
        YY_tensor = (Y.view(lattice_num,-1,1) @ Y.view(lattice_num,1,-1)).view(lattice_num, Y.shape[1], Y.shape[1])
        XY_tensor = (X.view(lattice_num,-1,1) @ Y.view(lattice_num,1,-1)).view(lattice_num, X.shape[1], Y.shape[1])
        XY_tensor_t = torch.transpose(XY_tensor,1,2).to(device)
        A1_from_EE_tensor = torch.masked_select((XX_tensor + YY_tensor),filter_XY).view(lattice_num,-1)
        B1_from_EE_tensor = torch.masked_select((XX_tensor - YY_tensor),filter_XY).view(lattice_num,-1)
        B2_from_EE_tensor = torch.masked_select((XY_tensor + XY_tensor_t),filter_XY).view(lattice_num,-1)
        A2_from_EE_tensor = torch.masked_select((XY_tensor - XY_tensor_t),filter_XY).view(lattice_num,-1)
        
        X_from_A1E_tensor = (A1.view(lattice_num,-1,1) @ X.view(lattice_num,1,-1)).view(lattice_num,-1)
        Y_from_A1E_tensor = (A1.view(lattice_num,-1,1) @ Y.view(lattice_num,1,-1)).view(lattice_num,-1)
        X_from_A2E_tensor = ((-1*A2).view(lattice_num,-1,1) @ Y.view(lattice_num,1,-1)).view(lattice_num,-1)
        Y_from_A2E_tensor = (A2.view(lattice_num,-1,1) @ X.view(lattice_num,1,-1)).view(lattice_num,-1)
        X_from_B1E_tensor = ((-1*B1).view(lattice_num,-1,1) @ X.view(lattice_num,1,-1)).view(lattice_num,-1)
        Y_from_B1E_tensor = (B1.view(lattice_num,-1,1) @ Y.view(lattice_num,1,-1)).view(lattice_num,-1)
        X_from_B2E_tensor = (B2.view(lattice_num,-1,1) @ Y.view(lattice_num,1,-1)).view(lattice_num,-1)
        Y_from_B2E_tensor = (B2.view(lattice_num,-1,1) @ X.view(lattice_num,1,-1)).view(lattice_num,-1)
        
        
        A1 = torch.cat((A1,A1_from_A1A1_tensor,A1_from_A2A2_tensor,A1_from_B1B1_tensor, A1_from_B2B2_tensor, A1_from_EE_tensor),dim = 1).to(device)
        A2 = torch.cat((A2,A2_from_A1A2_tensor, A2_from_B1B2_tensor, A2_from_EE_tensor),dim = 1).to(device)
        B1 = torch.cat((B1,B1_from_A1B1_tensor, B1_from_A2B2_tensor, B1_from_EE_tensor),dim = 1).to(device)
        B2 = torch.cat((B2,B2_from_A1B2_tensor, B2_from_B1A2_tensor, B2_from_EE_tensor),dim = 1).to(device)
        X = torch.cat((X,X_from_A1E_tensor,X_from_A2E_tensor,X_from_B1E_tensor,X_from_B2E_tensor),dim = 1).to(device)
        Y = torch.cat((Y,Y_from_A1E_tensor,Y_from_A2E_tensor,Y_from_B1E_tensor,Y_from_B2E_tensor),dim = 1).to(device)
        
        return A1,A2,B1,B2,X,Y
    
    
    def ActivationFunction_scalar(self,y,bias):
        absolute_y = torch.abs(y)
        y_after_act = f.sigmoid(absolute_y + bias)
        mask = y.clone()
        mask[mask<0] = -1
        mask[mask>=0] = 1
        return (y_after_act * mask).to(device)
    
    
    
    def ActivationFunction_doubleE(self,x,y,bias):
        x1 = x.view(lattice_num,x.shape[1],1)  #x1 = x.view(lattice_num,x.shape[1],-1)  
        y1 = y.view(lattice_num,y.shape[1],1)
        xy = torch.cat((x1,y1),dim=2)
        xy2 = xy.view(xy.shape[0],xy.shape[1],xy.shape[2],1)
        xy_norm = torch.linalg.matrix_norm(xy2)
        xy_norm_after_act = f.sigmoid(xy_norm + bias)
        return (x/xy_norm * xy_norm_after_act).to(device), (y / xy_norm * xy_norm_after_act).to(device)
    




