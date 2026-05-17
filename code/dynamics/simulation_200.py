import pandas as pd
import numpy as np
import torch
from model import Net
from dynamics_200 import holstein_dynamics
import time
from main_200 import Lsize
from main_200 import neighbor_2d
from main_200 import kT
from main_200 import num_steps
from main_200 import par_steps
from main_200 import Q_data
from main_200 import velocity_data
from main_200 import device
import numpy as np

################Some Parameters############################
dir_out = './ml_simulation/200/'
start = 0  
###########################################################



torch.manual_seed(0)
torch.set_default_dtype(torch.float64)



# load velocity and q-data in main, change model, record velocity, record every 50 steps


net = Net()
net = net.float().to(device)
prev_model_data = torch.load("./model/model.pt", map_location=torch.device(device))  # now the model is saved as a dictionary, so need to read as a dictionary
net.load_state_dict(prev_model_data['model'])
net = net.eval()

lat_sys = holstein_dynamics(net, neighbor_2d, Q_data, velocity_data, Lsize, kT)

start_time = time.time()
for i in range(start, num_steps+1):
    if i%par_steps==0:
        np.savetxt(dir_out + "c" + str(i) + ".dat", np.c_[lat_sys.Q, lat_sys.force, lat_sys.occ, lat_sys.velocity], delimiter = '\t')
    lat_sys.step()

print("duration time (minutes): ", (time.time() - start_time) / 60.)

