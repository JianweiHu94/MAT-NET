import os
import sys
import tensorflow as tf
import numpy as np


def group_ma_tensor(pointdata,neighbor,batch_size,point_num):
    # pointdata B N 4
    # neighbor B N 16
    # return B N 16 4
    neighbor_num = neighbor.get_shape()[2].value
    B = np.zeros([batch_size, point_num, neighbor_num, 1], dtype=np.int32)
    for iB in range(batch_size):
        B[iB,:,:] = iB
    B = tf.constant(B)
   
    neighbor = tf.to_int32(neighbor)
    print(neighbor)
    print('group_ma_tensor')
    print(tf.concat(axis=3, values=[B,tf.expand_dims(neighbor,axis=-1)]))
    groupdata= tf.gather_nd(pointdata[:,:,0:4], tf.concat(axis=3, values=[B,tf.expand_dims(neighbor,axis=-1)]))
    print(groupdata)
    return groupdata

