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
import tf_grouping
import edge_conditioned as econ
from transform_nets_edge_net import input_transform_net_edge_net
from transform_nets import feature_transform_net, input_transform_net
import edge_net
def placeholder_inputs(batch_size, num_point,neighbor_num):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 4))
    index_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, neighbor_num))
    mask_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, neighbor_num))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl, index_pl, mask_pl

def get_model_point(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value #32
    num_point = point_cloud.get_shape()[1].value #1024
    
    with tf.variable_scope('transform_net1') as sc:
        transform = input_transform_net(point_cloud, is_training, bn_decay, K=4)

    point_cloud_transformed = tf.matmul(point_cloud, transform)
    input_image = tf.expand_dims(point_cloud_transformed, -1)

    net = tf_util.conv2d(input_image, 64, [1,4],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)

    with tf.variable_scope('transform_net2') as sc:
        transform = feature_transform_net(net, is_training, bn_decay, K=64)

    net_transformed = tf.matmul(tf.squeeze(net), transform)
    net_transformed = tf.expand_dims(net_transformed, [2])

    net = tf_util.conv2d(net_transformed, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)

    # Symmetric function: max pooling
    max_index = tf.squeeze(tf.argmax(net,1))
    net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='maxpool')
    

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp2')


    return net, transform, max_index

def get_model_groupdata(group_data, mask, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx4, output Bx40 """
    batch_size = group_data.get_shape()[0].value #32
    num_point = group_data.get_shape()[1].value #1024

    with tf.variable_scope('transform_net1') as sc:
        transform = input_transform_net_edge_net(group_data, mask, is_training, bn_decay, K=4)

    group_data_transformed = tf.matmul(tf.reshape(group_data,[batch_size,-1,4]), transform)
    group_data_transformed = tf.reshape(group_data_transformed,[batch_size,num_point,-1]) # B N K C
    #input_image = tf.expand_dims(group_data_transformed, -1)
    input_image = group_data_transformed
    with tf.variable_scope('edge_net1') as sc:
        net, kernel, max_index_local_neighbor, masked_result = edge_net.edge_unit(input_image, mask, 'max', config.neighbor_num,32, scope='conv1', bn=True, is_training=is_training, bn_decay=bn_decay)

    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)

    with tf.variable_scope('transform_net2') as sc:
        transform = feature_transform_net(net, is_training, bn_decay, K=64)

    net_transformed = tf.matmul(tf.squeeze(net), transform)
    net_transformed = tf.expand_dims(net_transformed, [2])

    net = tf_util.conv2d(net_transformed, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)

    # Symmetric function: max pooling
    max_index_neighbor = tf.squeeze(tf.argmax(net,1))
    net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='maxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp2')


    return net, transform, max_index_neighbor, max_index_local_neighbor
def get_model_ec(group_data, mask, is_training, bn_decay=None):
    # groupdata B N K*C
    batch_size = group_data.get_shape()[0].value
    num_point  = group_data.get_shape()[1].value
    ec = econ.create_ec(group_data, mask) # B N K ec_leghth
    ec_length = ec.get_shape()[3].value
    ec = tf.reshape(ec, [batch_size, num_point, -1])  # B N 9
    with tf.variable_scope('transform_net1_ec') as sc:
        transform = input_transform_net_edge_net(ec, mask, is_training, bn_decay, K=ec_length)

    ec_transformed = tf.matmul(tf.reshape(ec,[batch_size,-1,ec_length]),transform)
    ec_transformed = tf.reshape(ec_transformed,[batch_size,num_point,-1])
    input_image = ec_transformed

    with tf.variable_scope('ec_net1') as sc:
        net, kernel, max_index_local_ec, masked_result = edge_net.edge_unit(input_image, mask, 'max', config.neighbor_num,32, scope='conv1', bn=True, is_training=is_training,
                                bn_decay=bn_decay)

    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    
    with tf.variable_scope('transform_net2') as sc:
        transform = feature_transform_net(net, is_training, bn_decay, K=64)

    net_transformed = tf.matmul(tf.squeeze(net), transform)
    net_transformed = tf.expand_dims(net_transformed, [2])

    net = tf_util.conv2d(net_transformed, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)

    max_index_ec = tf.squeeze(tf.argmax(net,1))   # Symmetric function: max pooling
    net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='maxpool')


    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp2')


    return net, transform, max_index_ec, max_index_local_ec, masked_result
def get_model_mask(point_cloud, index,mask, is_training, bn_decay=None):
    """ Classification PointNet
    input:
    point_cloud : BxNx4
    index : BxNxK
    mask: BxNxK

    output
    Bx40
    """
    end_points = {}
    
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    neighbor_num = config.neighbor_num
    
    group_data = tf_grouping.group_ma(point_cloud, index, batch_size, num_point)
    group_data = tf.reshape(group_data, [batch_size, num_point, neighbor_num * 4])  # B N K*4
    with tf.variable_scope('ma_net') as sc:
        net_only_ma, transform_only_ma, max_index = get_model_point(point_cloud,is_training, bn_decay=None)
    end_points['transform_ma'] = transform_only_ma
    with tf.variable_scope('group_data_net') as sc:
        net_only_groupdata, transform_groupdata, max_index_neighbor, max_index_local_neighbor = get_model_groupdata(group_data,mask,is_training, bn_decay=None)
    end_points['transform_groupdata'] = transform_groupdata
    with tf.variable_scope('ec_net') as sc:
        net_only_ec, transform_ec, max_index_ec, max_index_local_ec, masked_result_ec = get_model_ec(group_data,mask,is_training, bn_decay=None)
    end_points['transform_ec'] = transform_ec
    end_points['max_index'] = max_index
    end_points['max_index_neighbor'] = max_index_neighbor
    end_points['max_index_ec'] = max_index_ec
    end_points['max_index_local_neighbor'] = max_index_local_neighbor
    end_points['max_index_local_ec'] = max_index_local_ec
    end_points['masked_result_ec'] = masked_result_ec

    net = tf.concat([net_only_ma,net_only_groupdata],1)
    net = tf.concat([net, net_only_ec], 1)

    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc3', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp2')

    net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc4')

    return net, end_points






def input_transform_net(point_cloud, is_training, bn_decay=None, K=4):
    """ Input (XYZ) Transform Net, input is BxNx3 gray image
        Return:
            Transformation matrix of size 3xK """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value

    input_image = tf.expand_dims(point_cloud, -1)
    net = tf_util.conv2d(input_image, 64, [1,4],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv3', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='tfc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='tfc2', bn_decay=bn_decay)

    with tf.variable_scope('transform_XYZ') as sc:
        if (K == 4):
            weights = tf.get_variable('weights', [256, 4 * 4],
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




def get_loss(pred, label, end_points, reg_weight=0.001):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)

    # Enforce the transformation as orthogonal matrix
    transform_ma = end_points['transform_ma']  # BxKxK
    K = transform_ma.get_shape()[1].value
    mat_diff = tf.matmul(transform_ma, tf.transpose(transform_ma, perm=[0,2,1]))
    mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
    mat_diff_ma_loss = tf.nn.l2_loss(mat_diff)
    tf.summary.scalar('ma loss', mat_diff_ma_loss)

    transform_groupdata = end_points['transform_groupdata']  # BxKxK
    K = transform_groupdata.get_shape()[1].value
    mat_diff_groupdata = tf.matmul(transform_groupdata, tf.transpose(transform_groupdata, perm=[0, 2, 1]))
    mat_diff_groupdata -= tf.constant(np.eye(K), dtype=tf.float32)
    mat_diff_groupdata_loss = tf.nn.l2_loss(mat_diff_groupdata)
    tf.summary.scalar('groupdata loss', mat_diff_groupdata_loss)

    transform_ec = end_points['transform_ec']  # BxKxK
    K = transform_ec.get_shape()[1].value
    mat_diff_ec = tf.matmul(transform_ec, tf.transpose(transform_ec, perm=[0, 2, 1]))
    mat_diff_ec -= tf.constant(np.eye(K), dtype=tf.float32)
    mat_diff_ec_loss = tf.nn.l2_loss(mat_diff_ec)
    tf.summary.scalar('ec loss', mat_diff_ec_loss)
    return classify_loss + reg_weight*mat_diff_ma_loss+reg_weight*mat_diff_groupdata_loss + reg_weight*mat_diff_ec_loss


if __name__ == '__main__':
    with tf.Graph().as_default():
        with tf.Session().as_default():
            pointclouds_pl, labels_pl, index_pl, mask_pl = placeholder_inputs(32, config.point_num, config.neighbor_num)
            tf.global_variables_initializer()
            inputs = tf.zeros((32, config.point_num, 4))
            mask = tf.zeros((32, config.point_num, config.neighbor_num))
            index = tf.zeros((32, config.point_num, config.neighbor_num))
            outputs, end_points = get_model_mask(pointclouds_pl, index_pl, mask_pl, tf.constant(True))
            loss = get_loss(outputs, labels_pl, end_points)
