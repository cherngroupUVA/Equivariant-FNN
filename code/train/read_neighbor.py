import csv
from os import listdir
from os.path import isfile, join
import pandas as pd
import torch
#import constant
import math

def read_neighbor_list(file_name, ramp, Lsize):
    num_site = Lsize * Lsize

    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        neighbor_list = list(reader)
    
    neighbor_list_length = torch.zeros([ramp + 1], dtype=torch.int32)    #to store how many neighbors are there at each layer
    sum  = 0
    for i in range(0, ramp+1):
        neighbor_list_length[i] = len(neighbor_list[i])     #totally how many neighbors are there around the center, and get the number of neighbors at each layer
        sum += len(neighbor_list[i])

    #print(neighbor_list_length)
    #print(sum)

    tot_length = sum

 
    neighbor_2d = torch.zeros([num_site, tot_length], dtype=torch.long).to(device)  # each center, all the numbers of the neighbors
    count = 0
    for i in range(len(neighbor_list)):
        for j in range(len(neighbor_list[i])):
            neighbor_list[i][j] = int(neighbor_list[i][j])

            m = int(count / tot_length)
            n = count % tot_length 
            neighbor_2d[m][n] = neighbor_list[i][j]

            count += 1
    
    #print(neighbor_2d)

    neighbor_type = torch.zeros(ramp + 1, dtype=torch.int).to(device)

    for i in range(1, ramp + 1):
        if neighbor_list_length[i] == 4:
            if neighbor_list[0][0] // Lsize == neighbor_list[i][0] // Lsize or neighbor_list[0][0] % Lsize == neighbor_list[i][0] % Lsize:
                neighbor_type[i] = 1
            else:
                neighbor_type[i] = 2
        if neighbor_list_length[i] == 8:
            neighbor_type[i] = 3

    return neighbor_2d, tot_length, neighbor_type


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")







