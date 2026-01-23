""" Analysis functions for generate irreducible representations bond """
import torch
#import constant
import math

torch.set_printoptions(linewidth=200, precision=3)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_feature_mat(Lsize, num_feature, neighbor_type):
    num_site = Lsize * Lsize
    num_layer = neighbor_type.size(0)

    ir4 = torch.tensor([[1, 1, 1, 1], [1, -1, 1, -1], [1, 0, -1, 0], [0, -1, 0, 1]], dtype=torch.double).to(device)

    ir4_2 = torch.tensor([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]], dtype=torch.double).to(device)

    ir8 = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1], [1, -1, 1, -1, 1, -1, 1, -1], [1, 1, -1, -1, 1, 1, -1, -1], [1, -1, -1, 1, 1, -1, -1, 1],
    [1, 1, 0, 0, -1, -1, 0, 0], [0, 0, -1, -1, 0, 0, 1, 1], [0, 0, 1, -1, 0, 0, -1, 1], [1, -1, 0, 0, -1, 1, 0, 0]], dtype=torch.double).to(device)

    mat_ir = torch.tensor([[1.]]).to(device)

    for i in range(1, num_layer):
        if neighbor_type[i] == 1:
            mat_ir = torch.block_diag(mat_ir, ir4)
        if neighbor_type[i] == 2:
            mat_ir = torch.block_diag(mat_ir, ir4_2)
        if neighbor_type[i] == 3:
            mat_ir = torch.block_diag(mat_ir, ir8)


    mat1_1d = torch.tensor([[0.]]).to(device)
    mat1_2d = torch.tensor([[1, 1], [0, 0]], dtype=torch.double).to(device)
    mat1_4 = torch.block_diag(mat1_1d, mat1_1d, mat1_2d)
    mat1_8 = torch.block_diag(mat1_1d, mat1_1d, mat1_1d, mat1_1d, mat1_2d, mat1_2d)

    mat1 = torch.tensor([[0.]]).to(device)
    for i in range(1, num_layer):
        if neighbor_type[i] == 1 or neighbor_type[i] == 2:
            mat1 = torch.block_diag(mat1, mat1_4)
        if neighbor_type[i] == 3:
            mat1 = torch.block_diag(mat1, mat1_8)

    
    mat2_1d = torch.tensor([[1.]]).to(device)
    mat2_2d = torch.tensor([[0, 0], [1, 1]], dtype=torch.double).to(device)
    mat2_4 = torch.block_diag(mat2_1d, mat2_1d, mat2_2d)
    mat2_8 = torch.block_diag(mat2_1d, mat2_1d, mat2_1d, mat2_1d, mat2_2d, mat2_2d)

    mat2 = torch.tensor([[1.]]).to(device)
    for i in range(1, num_layer):
        if neighbor_type[i] == 1 or neighbor_type[i] == 2:
            mat2 = torch.block_diag(mat2, mat2_4)
        if neighbor_type[i] == 3:
            mat2 = torch.block_diag(mat2, mat2_8)

    ref_index = torch.zeros(2, num_site, num_feature, dtype=torch.long).to(device)
    for i in range(num_site):
        for j in range(num_feature):
            ref_index[0][i][j] = i

        ref_index[1][i][0] = 0
        count = 1
        for j in range(1, num_layer):
            if neighbor_type[j] == 1:
                ref_index[1][i][count] = 1
                ref_index[1][i][count + 1] = 3
                ref_index[1][i][count + 2] = 5
                ref_index[1][i][count + 3] = 6
                count += 4
            if neighbor_type[j] == 2:
                ref_index[1][i][count] = 1
                ref_index[1][i][count + 1] = 4
                ref_index[1][i][count + 2] = 5
                ref_index[1][i][count + 3] = 6
                count += 4
            if neighbor_type[j] == 3:
                ref_index[1][i][count] = 1
                ref_index[1][i][count + 1] = 2
                ref_index[1][i][count + 2] = 3
                ref_index[1][i][count + 3] = 4
                ref_index[1][i][count + 4] = 5
                ref_index[1][i][count + 5] = 6
                ref_index[1][i][count + 6] = 5
                ref_index[1][i][count + 7] = 6
                count += 8

    return mat_ir, mat1, mat2, ref_index

ref_self_a2_content = torch.zeros(3, 3, 2).to(device)
ref_self_b1_content = torch.zeros(3, 3, 2).to(device)
ref_self_b2_content = torch.zeros(3, 3, 2).to(device)

ref_self_a2_content[1, 1, 0] = 1
ref_self_b1_content[1, 1, 0] = 1
ref_self_b2_content[1, 1, 0] = 1

ref_self_a2_content[-1, 1, 1] = 1
ref_self_b1_content[-1, 1, 1] = -1
ref_self_b2_content[-1, 1, 1] = -1

ref_self_a2_content[1, -1, 1] = 1
ref_self_b1_content[1, -1, 1] = -1
ref_self_b2_content[1, -1, 1] = -1

ref_self_a2_content[-1, -1, 0] = 1
ref_self_b1_content[-1, -1, 0] = 1
ref_self_b2_content[-1, -1, 0] = 1

ref_self_a2_content[1, -1, 0] = -1
ref_self_b1_content[1, -1, 0] = 1
ref_self_b2_content[1, -1, 0] = -1

ref_self_a2_content[-1, 1, 0] = -1
ref_self_b1_content[-1, 1, 0] = 1
ref_self_b2_content[-1, 1, 0] = -1

ref_self_a2_content[1, 1, 1] = -1
ref_self_b1_content[1, 1, 1] = -1
ref_self_b2_content[1, 1, 1] = 1

ref_self_a2_content[-1, -1, 1] = -1
ref_self_b1_content[-1, -1, 1] = -1
ref_self_b2_content[-1, -1, 1] = 1


def generate_Q_feature(Q, neighbor_2d, mat_ir, mat1, mat2, ref_index):
    num_site = neighbor_2d.size(0)
    num_feature = neighbor_2d.size(1)

    Q_2d = Q[neighbor_2d]
    

    mat_ir = mat_ir.expand(num_site, -1, -1)

    Q_2d = Q_2d.unsqueeze(2) 

    bf = torch.matmul(mat_ir, Q_2d)
    bf = bf.squeeze()
    

    mat1 = mat1.expand(num_site, -1, -1)
    mat2 = mat2.expand(num_site, -1, -1)

    #referene layer index is [13:19], total of 6

    ref_cont_Q0 = torch.ones(num_site, 1).to(device)    #for the central Q
    ref_cont_1d = torch.sign(bf[:, 13:17])     #for the reference a1 a2 b1 b2 representation
    ref_cont_1d[:, 0] = 1     #a1 is not changed, for the reference layer we should not change it as it introduces spurious symmetry, for other layers we can do either way
    ref_cont_2d = torch.nn.functional.normalize(bf[:, 17:19], dim=1)  #for the E irreducible representation reference

    ref_cont = torch.cat((ref_cont_Q0, ref_cont_1d, ref_cont_2d), 1)

    ref = ref_cont[ref_index[0], ref_index[1]]

    #=============== for the reference layer, transformation of a2 b1 b2 to be consistent with E
    ref_self_abb_index1 = torch.sign(bf[:, 17]).long()
    ref_self_abb_index2 = torch.sign(bf[:, 18]).long()
    ref_self_abb_index3 = torch.max(torch.abs(bf[:, 17:19]), 1)[1].long()

    ref_self_a2 = ref_self_a2_content[ref_self_abb_index1, ref_self_abb_index2, ref_self_abb_index3]
    ref_self_b1 = ref_self_b1_content[ref_self_abb_index1, ref_self_abb_index2, ref_self_abb_index3]
    ref_self_b2 = ref_self_b2_content[ref_self_abb_index1, ref_self_abb_index2, ref_self_abb_index3]

    ref[:, 14] = ref_self_a2
    ref[:, 15] = ref_self_b1
    ref[:, 16] = ref_self_b2
    #==============
    #print(bf.size(), ref.size())
    #print(mat1.size(), mat2.size())

    feature_ref_E2 = torch.max(torch.abs(bf[:, 17:19]), 1)[0] #this is the 2d reference multiply (1, 0), with 8-fold symmetry removed

    #matmul() cannot do batched matrix times batched vector directly, so have to use unsqueeze() and squeeze()    
    Q_feature = torch.sqrt(torch.matmul( mat1, torch.mul(bf, bf).unsqueeze(2) ).squeeze())

    Q_feature += torch.matmul( mat2, torch.mul(bf, ref).unsqueeze(2) ).squeeze()

    Q_feature[:, 18] = feature_ref_E2
    
    #n = 200
    #print(ref_self_a2[n], ref_self_b1[n], ref_self_b2[n])
    #print(bf[n])
    #print(ref_cont[n])
    #print(ref[n])
    #print(Q_feature[n])

    return Q_feature

'''
def generate_Q_feature(Q, neighbor_2d, mat_ir, mat1, mat2, ref_index):
    num_site = neighbor_2d.size(0)
    num_feature = neighbor_2d.size(1)

    Q_2d = Q[neighbor_2d]
    #print(Q_2d[0])

    mat_ir = mat_ir.expand(num_site, -1, -1)

    Q_2d = Q_2d.unsqueeze(2)

    bf = torch.matmul(mat_ir, Q_2d)
    bf = bf.squeeze()


    mat1 = mat1.expand(num_site, -1, -1)
    mat2 = mat2.expand(num_site, -1, -1)

    ref_cont_Q0 = torch.ones(num_site, 1).to(constant.device)    #for the central Q
    ref_cont_1d = torch.sign(bf[:, 13:17])      #for the reference a1 a2 b1 b2 representation
    ref_cont_2d = torch.nn.functional.normalize(bf[:, 17:19], dim=1)         #for the reference E representation

    ref_cont = torch.cat((ref_cont_Q0, ref_cont_1d, ref_cont_2d), 1)

    ref = ref_cont[ref_index[0], ref_index[1]]


    #print(bf.size(), ref.size())
    #print(mat1.size(), mat2.size())

    feature_ref_E2 = torch.max(torch.abs(bf[:, 17:19]), 1)[0] #this is the 2d reference multiply (1, 0), with 8-fold symmetry removed

    Q_feature = torch.sqrt(torch.matmul( mat1, torch.mul(bf, bf).unsqueeze(2) ).squeeze())

    Q_feature += torch.matmul( mat2, torch.mul(bf, ref).unsqueeze(2) ).squeeze()

    Q_feature[:, 18] = feature_ref_E2

    #print(bf[0])
    #print(ref[0])
    #print(Q_feature[0])

    return Q_feature
'''





