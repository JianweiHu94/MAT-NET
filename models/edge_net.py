import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../models'))
import tf_util
import config
import edge_conditioned as econ

def edge_unit(point_cloud, mask, pooling,neighbornum, outchannel, scope, bn=False, activation_fn=tf.nn.relu, bn_decay=None, is_training=None):
    """
    :param point_cloud: tensor
    :param mask: tensor
    :param pooling: String
    :return: Variable tensor
    """

    coordinate_length = neighbornum  # adj points number * 4, will change
    input_image = tf.expand_dims(point_cloud, -1) # B N C*K 1
    #masked_result = mask
    mask = tf.expand_dims(mask, 0) # 1 32 512 16
 
    mask = tf.tile(mask,[outchannel,1,1,1])
    #print(mask.shape)
    #print(mask.shape)
    ww = point_cloud.get_shape()[2].value / coordinate_length

    kernel_1 = tf_util._variable_with_weight_decay(name='weights_1', shape=[1, ww, 1, outchannel], use_xavier=True,
                                                   stddev=0.001, wd=0.1)  # kernel_h, kernel_w, num_in_channels, output

    biases_1 = tf_util._variable_on_cpu('biases_1', [outchannel], tf.constant_initializer(0.0))

    outputs = tf.nn.conv2d(input_image,
                           kernel_1,
                           [1, 1, ww, 1],  # [1, stride_h, stride_w, 1]
                           padding='VALID')  # 4 -> 1
    outputs = tf.nn.bias_add(outputs, biases_1)
    if bn:
        outputs = tf_util.batch_norm_for_conv2d(outputs, is_training, bn_decay=bn_decay, scope='bn')
    if activation_fn is not None:
        outputs = activation_fn(outputs)
    #masked_result = mask
    outputs = tf.transpose(outputs,[3,0,1,2]) # 32 32 512 16
    outputs = tf.multiply(outputs, mask) # 32 32 512 16
    #print(outputs.shape)

    #max_index = tf.argmax(outputs,)
    #print(tf.shape(outputs))
    outputs = tf.transpose(outputs,[1,2,3,0]) # 32 512 16 32
   
    masked_result = outputs 
    print(masked_result.shape)
    #print(masked_result.shape)
    max_index_local = tf.squeeze(tf.argmax(outputs,2))
    if pooling == 'max':
        outputs = tf.nn.max_pool(outputs,
                                 ksize=[1, 1, coordinate_length, 1],
                                 strides=[1, 1, 1, 1],
                                 padding='VALID')
        
    elif pooling == 'avg':
        outputs = tf.nn.avg_pool(outputs,
                                 ksize=[1, 1, coordinate_length, 1],
                                 strides=[1, 1, 1, 1],
                                 padding='VALID')
    return outputs, kernel_1, max_index_local, masked_result

def edge_unit_with_ec(point_cloud, mask, pooling,neighbornum, outchannel, scope, bn=False, activation_fn=tf.nn.relu, bn_decay=None, is_training=None):
    """
    :param point_cloud: B N C*K
    :param mask: tensor
    :param pooling: String
    :return: Variable tensor
    """
    batch_size = point_cloud.get_shape()[0]
    point_num = point_cloud.get_shape()[1]

    coordinate_length = neighbornum  # adj points number * 4, will change
    #input_image = tf.expand_dims(point_cloud, -1) # B N 1 C*K
    mask = tf.expand_dims(mask, -1)
    ec = econ.create_ec(point_cloud, mask)
    ec_length = ec.get_shape()[3].value
    ec = tf.reshape(ec,[batch_size,point_num,-1])
    ec = tf.expand_dims(ec,axis=3)
    #ww = point_cloud.get_shape()[2].value / coordinate_length

    kernel_1 = tf_util._variable_with_weight_decay(name='weights_1', shape=[1, ec_length, 1, outchannel], use_xavier=True,
                                                   stddev=0.001, wd=0.1)  # kernel_h, kernel_w, num_in_channels, output

    biases_1 = tf_util._variable_on_cpu('biases_1', [outchannel], tf.constant_initializer(0.0))

    outputs = tf.nn.conv2d(ec,
                           kernel_1,
                           [1, 1, ec_length, 1],  # [1, stride_h, stride_w, 1]
                           padding='VALID')  # 4 -> 1
    outputs = tf.nn.bias_add(outputs, biases_1)
    if bn:
        outputs = tf_util.batch_norm_for_conv2d(outputs, is_training, bn_decay=bn_decay, scope='bn')
    if activation_fn is not None:
        outputs = activation_fn(outputs)


    outputs = outputs * mask
    #for i in range(100000):
    #    print(tf.shape(outputs))
    max_index = tf.squeeze(tf.argmax(tf.squeeze(outputs,-1)),-1)
    if pooling == 'max':
        outputs = tf.nn.max_pool(outputs,
                                 ksize=[1, 1, coordinate_length, 1],
                                 strides=[1, 1, 1, 1],
                                 padding='VALID')
    elif pooling == 'avg':
        outputs = tf.nn.avg_pool(outputs,
                                 ksize=[1, 1, coordinate_length, 1],
                                 strides=[1, 1, 1, 1],
                                 padding='VALID')
    return outputs, kernel_1

def edge_unit_without_pooling(data, mask, pooling,neighbornum, outchannel, scope, bn=False, activation_fn=tf.nn.relu, bn_decay=None, is_training=None):
    """
    :param point_cloud: B N C*K
    :param mask: tensor
    :param pooling: String
    :return: Variable tensor
    """
    batch_size = data.get_shape()[0]
    point_num = data.get_shape()[1]

    mask = tf.expand_dims(mask, -1)

    ww = data.get_shape()[2].value / neighbornum
    data = tf.reshape(data,[batch_size, point_num,-1])
    data = tf.expand_dims(data,-1)

    kernel_1 = tf_util._variable_with_weight_decay(name='weights_1', shape=[1, ww, 1, outchannel], use_xavier=True,
                                                   stddev=0.001, wd=0.1)  # kernel_h, kernel_w, num_in_channels, output

    biases_1 = tf_util._variable_on_cpu('biases_1', [outchannel], tf.constant_initializer(0.0))

    outputs = tf.nn.conv2d(data,
                           kernel_1,
                           [1, 1, ww, 1],  # [1, stride_h, stride_w, 1]
                           padding='VALID')  # 4 -> 1
    outputs = tf.nn.bias_add(outputs, biases_1)
    if bn:
        outputs = tf_util.batch_norm_for_conv2d(outputs, is_training, bn_decay=bn_decay, scope='bn')  # none values not supported
    if activation_fn is not None:
        outputs = activation_fn(outputs)


    outputs = outputs * mask
    #
    #
    # outputs = tf_util.conv2d(outputs, 32, [1, 1],
    #                      padding='VALID', stride=[1, 1],
    #                      bn=True, is_training=is_training,
    #                      scope='ec_conv2', bn_decay=bn_decay)
    # outputs = tf.reshape(outputs, [batch_size, -1])
    # outputs = tf_util.fully_connected(outputs, point_num*neighbornum*16, bn=True, is_training=is_training,
    #                               scope='tfc1', bn_decay=bn_decay)
    # outputs = tf_util.fully_connected(outputs, point_num*neighbornum*8, bn=True, is_training=is_training,
    #                               scope='tfc2', bn_decay=bn_decay)
    # outputs = tf_util.fully_connected(outputs, point_num*neighbornum*7, bn=True, is_training=is_training,
    #                                  scope='tfc3', bn_decay=bn_decay)
    tmp = tf.zeros([1,neighbornum * 7])
    for i in range(batch_size):
        #for j in range(batch_size):
        for j in range(point_num):
            edges = outputs[i,j,:]
            edges = tf.reshape(edges,[1,-1])
            with tf.variable_scope('ec_weights_%d_%d'%(i,j)) as sc:
                edges = tf_util.fully_connected(edges, neighbornum * 32, bn=True, is_training=is_training,
                                                                             scope='tfc1', bn_decay=bn_decay)

                rst = tf_util.fully_connected(edges, neighbornum * 7, bn=True, is_training=is_training,
                                                                             scope='tfc2', bn_decay=bn_decay)
                tmp = tf.concat([tmp,rst],axis=0)
            a = 1
    outputs = tmp[1:,]
    outputs = tf.reshape(outputs,[batch_size,point_num,neighbornum,-1])

    return outputs

def ec_to_weights(ec, mask,neighbornum, C= 3, bn_decay=None, is_training=None):
    # ec B N K*C
    batch_size = ec.get_shape()[0]
    point_num = ec.get_shape()[1]
    #K = ec.get_shape()[2]

    #ec_feature = edge_unit_with_ec(ec,mask,'max', config.neighbor_num, 32, scope='ec_conv1', bn=True, is_training=is_training,bn_decay=bn_decay)

    #ec_feature = tf.expand_dims(ec_feature, -1)
    #ec = tf.reshape(ec,[batch_size,point_num,neighbornum, C]) # B N K C

    with tf.variable_scope('edge_net_without_pooling') as sc:
        net, kernel = edge_unit(ec, mask, 'max', config.neighbor_num, 32, scope='conv1', bn=True, is_training=is_training,bn_decay=bn_decay)



    net = tf_util.conv2d(ec, 64, [1, C],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='ec_conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='ec_conv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='ec_conv3', bn_decay=bn_decay)

    net = tf_util.avg_pool2d(net, [1, 1024],
                             padding='VALID', scope='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='tfc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='tfc2', bn_decay=bn_decay)

    with tf.variable_scope('transform_XYZ') as sc:
        if (K == 4):
            weights = tf.get_variable('weights', [256, N * K],
                                      initializer=tf.constant_initializer(0.0),
                                      dtype=tf.float32)
            biases = tf.get_variable('biases', [4 * 4],
                                     initializer=tf.constant_initializer(0.0),
                                     dtype=tf.float32)
            biases += tf.constant(np.eye(K).flatten(), dtype=tf.float32)
            transform = tf.matmul(net, weights)
            transform = tf.nn.bias_add(transform, biases)

            transform = tf.reshape(transform, [batch_size, 4, 4])
    return transform
