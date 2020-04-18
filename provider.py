import os
import sys
import numpy as np
import h5py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Download dataset for point cloud classification
DATA_DIR = os.path.join(BASE_DIR, 'data')


def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx,...], labels[idx]


def shuffle_data_mask(data, labels,index,mask):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)

    return data[idx,...], labels[idx], index[idx,...],mask[idx,...]


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx4 array, original batch of point clouds
        Return:
          BxNx4 array, rotated batch of point clouds
          change : 2017/7/20
    """

    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    wide = batch_data.shape[2]
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)

        rotation_matrix = np.array([[cosval, sinval, 0, 0],
                                    [-sinval, cosval, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])

        shape_pc = batch_data[k, ...]
        shape_pc_rotate = np.dot(shape_pc.reshape((-1, 4)), rotation_matrix)
        rotated_data[k, ...] = shape_pc_rotate.reshape(-1, wide)
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    wide = batch_data.shape[2]
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, sinval, 0, 0],
                                    [-sinval, cosval, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])

        shape_pc = batch_data[k, ...]
        shape_pc_rotate = np.dot(shape_pc.reshape((-1, 4)), rotation_matrix)
        rotated_data[k, ...] = shape_pc_rotate.reshape(-1, wide)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.008, clip=0.01):  # 0.01 0.05 changed
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    '''print('batch_data.shape:\n')  2048
    print(batch_data.shape)'''
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    '''print('jittered_data.shape:\n')
    print(jittered_data.shape)'''
    return jittered_data


def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)


def loadDataFile(filename):
    return load_h5(filename)


def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return (data, label, seg)


def loadDataFile_with_seg(filename):
    return load_h5_data_label_seg(filename)