import torch
import numpy as np
import math
import main_200
#from generate_feature import generate_Q_feature
from main_200 import device

class holstein_dynamics:

    def __init__(self, model, neighbor_2d, Q, velocity, Lsize=200, kT=0.2):
        self.ts = Lsize * Lsize
        self.ls = Lsize
        self.gamma = 0.2
        self.dt = 0.05
        self.kT = kT #kT should be the temp?
        self.kT_rand = 5.
        self.k0 = 1.0  # k0 is the elastic 
        self.k1 = 0.18  #k1 is qud
        self.G = 3.5
        self.mass = 5.0
        self.neighbor_2d = neighbor_2d

        self.a_x = math.exp(-self.gamma * self.dt); # 
        self.b_x = math.sqrt( (1. - pow(self.a_x, 2)) / self.mass );

        self.net = model
        
        if Q is not None:
            self.Q = Q
            
            
        else:
            self.Q = np.random.randn(self.ts) * math.sqrt(self.kT_rand / self.k1)

        if velocity is not None:
            self.velocity = velocity
            
        else:
            self.velocity = np.random.randn(self.ts) * math.sqrt(self.kT_rand / self.mass)

        self.force = self.calc_force(self.Q)
        
        self.occ = self.calc_occ(self.Q)


    def calc_force(self, Q):
        
        Q_tensor = torch.tensor(Q, dtype=torch.float64).view(1,-1).to(device)
        
        scaled_A1, scaled_A2, scaled_B1, scaled_B2, scaled_X, scaled_Y = main_200.prepare_input_IRs(Q_tensor)
        
        result_A1, result_A2, result_B1, result_B2, result_X, result_Y = self.net(scaled_A1[0],scaled_A2[0],scaled_B1[0],scaled_B2[0],scaled_X[0],scaled_Y[0])
        
        force_prediction = result_A1.view(-1)

        return force_prediction.cpu().detach().numpy()

    def calc_force_classical(self, Q):  
        Q_2d = Q.reshape(self.ls, self.ls)

        up = np.roll(Q_2d, 1, 0)
        down = np.roll(Q_2d, -1, 0)
        left = np.roll(Q_2d, -1, 1)
        right = np.roll(Q_2d, 1, 1)        

        force_classical = -self.k0 * Q_2d  - self.k1 * (up + down + right + left)  
        force_classical = force_classical.reshape(-1)
        return force_classical


    def calc_occ(self, Q):
        force_classical = self.calc_force_classical(Q)
        occ = (self.force - force_classical) / self.G + 0.5
        return occ        


    def step(self):
        for i in range(self.ts):
            self.Q[i] += self.velocity[i] * self.dt + 0.5 * (self.force[i] / self.mass) * self.dt * self.dt

        force_prev = self.force
        self.force = self.calc_force(self.Q)
        self.occ = self.calc_occ(self.Q)

        for i in range(self.ts):
            self.velocity[i] += 0.5 * self.dt * (self.force[i] + force_prev[i]) / self.mass
        for i in range(self.ts):
            self.velocity[i] = self.a_x * self.velocity[i] + self.b_x * math.sqrt(self.kT) * np.random.randn()









