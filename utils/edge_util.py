import numpy as np


Neighbors_num  = 5


def compute_all_samples_index(num_point,indexs):
    samples_num = indexs.shape[0]
    all_index = []
    for i in range(samples_num):
        index = indexs[i,...]
        edge_num = int(indexs[i,0,0])
        point_neighbors = []
        for k in range(num_point):
            point_neighbors.append([])
        for j in range(edge_num):
            a = int(indexs[i, j+1, 0])
            b = int(indexs[i, j+1, 1])
            point_neighbors[int(indexs[i, j+1, 0])].append(int(indexs[i, j+1, 1]))
            point_neighbors[int(indexs[i, j+1, 1])].append(int(indexs[i, j+1, 0]))
        all_index.append(point_neighbors)
    return all_index

def get_point_by_index(point_cloud, allindex):

    #cound every point's neighbor points
    samples_num = point_cloud.shape[0]  # 32
    num_point = point_cloud.shape[1]  # 128
    newdata = np.zeros([samples_num,num_point,(Neighbors_num+1)*4],'float')
    for i in range(samples_num):
        data = point_cloud[i,...]
        neighbor_martrix = compute_matrix(data,allindex[i])
        newdata[i,:,:] = neighbor_martrix[:,:]
    return newdata

def compute_matrix(data, index):
    #
    point_num = len(index)
    neighbor_index = compute_neighbor_index(index)
    K = neighbor_index.shape[1]
    neighbor_martrix = np.zeros([point_num, K*4],'float')
    for i in range(point_num):
        for j in range(K):
            neighbor_martrix[i,j*4:(j+1)*4] = data[neighbor_index[i,j]]
    return neighbor_martrix

def compute_neighbor_index(index):
    K = Neighbors_num
    point_num = len(index)
    neighbor_index = np.zeros([point_num,K+1],'int')
    for i in range(point_num):
        neighbor_index[i,...] = i
    # Select K points from per point
    # neighbors num < K, add point self
    for i in range(point_num):
        if len(index[i]) < K:
            a = np.array(index[i])
            neighbor_index[i,1:1+len(index[i])] = a[:]
        else:
            a = np.array(index[i])
            idx = np.arange(len(a))
            np.random.shuffle(a)
            #index = index[idx]
            neighbor_index[i, 1:] = a[0:K]
    return neighbor_index

def compute_distance_matrix(point_cloud):
    #INPUT
    #
    return np.sqrt()