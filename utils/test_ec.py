import tensorflow as tf


def create_ec(point_cloud,mask):
    '''


    :param point_cloud: B N C
    :return:
    B N C/4*8 (x y z r1 r2 mol arcsin(z/mol) arctan(y/x))

    '''
    batch_size = point_cloud.get_shape()[0].value
    point_num = point_cloud.get_shape()[1].value
    neighbor_num = mask.get_shape()[2].value
    point_dimension = point_cloud.get_shape()[2].value / neighbor_num # 4 or 3
    point_cloud = tf.reshape(point_cloud,[batch_size,point_num,neighbor_num,point_dimension])
    # compute xyzr
    ec_xyzr = point_cloud

    # get r1,r2
    ec_r2 = point_cloud[:,:,:,3]

    center_r1 = point_cloud[:,:,0,3] # B N
    center_r1_expand = tf.expand_dims(center_r1,axis=2) # B N 1
    ec_r1 = center_r1_expand

    center_ma_0 = tf.expand_dims(point_cloud[:,:,0,:],axis=2) # B N 1 4or3
    center_ma = center_ma_0

    for i in range(neighbor_num-1):
        ec_r1 = tf.concat([ec_r1,center_r1_expand],axis=2)  # B N neighbor_num
        center_ma = tf.concat([center_ma,center_ma_0],axis=2) # B N neighbor_num 4

    ec_r12 = tf.concat([tf.expand_dims(ec_r1,axis=3),tf.expand_dims(ec_r2,axis=3)],axis=3) # B N Neighbor_num 2

    ec_xyzr_sub = center_ma - point_cloud # B N neighbor_num 4
    ec_xyz = ec_xyzr_sub[:,:,:,0:3] # B N neighbor_num 3

    ec_mol = tf.sqrt(tf.pow(ec_xyz[:,:,:,0],2) + tf.pow(ec_xyz[:,:,:,1],2) + tf.pow(ec_xyz[:,:,:,2],2)) # B N Neighbor_num
    ec_mol = tf.expand_dims(ec_mol,axis=-1) # B N Neighbor_num 1

    ec_asin = tf.asin(ec_xyz[:,:,:,2] / (ec_mol[:,:,:,0]))
    ec_asin = tf.expand_dims(ec_asin,axis=3)
    ec_atan = tf.atan(ec_xyz[:,:,:,1] / (ec_xyz[:,:,:,0] + 0.000000001))
    ec_atan = tf.expand_dims(ec_atan, axis=3)
    ec = tf.concat([ec_xyzr_sub,ec_r12],axis=3)
    ec = tf.concat([ec,ec_mol], axis = 3)
    ec = tf.concat([ec,ec_asin], axis = 3)
    ec = tf.concat([ec, ec_atan], axis=3)
    a = 1
    return ec

if __name__ == '__main__':
    data = tf.constant([[[1,1,1,1,2,2,2,2]]],dtype=tf.float32)
    mask = tf.constant([[[1,1]]],dtype=tf.float32)
    mask = tf.expand_dims(mask,axis=3)
    print(data.get_shape())
    sess = tf.Session()
    result = sess.run(data)
    print(result)
    ec = create_ec(data,mask)
    result = sess.run(ec)
    print(result)
    sess.close()