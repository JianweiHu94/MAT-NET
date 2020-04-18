import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import tf_util
import edge_util
import config
import tf_grouping
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='matnet', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=config.point_num, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=config.max_epoch, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
NEIGHBOR_NUM = config.neighbor_num


MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure

MAX_NUM_POINT = 1024
NUM_CLASSES = 40

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

MAX_TEST_ACC = 0.85
CURRENT_TRAIN_ACC  =0.0
CURRENT_EPOCH  =0
CURRENT_TRAIN_EPOCH = 0

# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles(os.path.join(BASE_DIR, 'data/train_files.txt'))
TEST_FILES = provider.getDataFiles(os.path.join(BASE_DIR, 'data/test_files.txt'))
TRAIN_FILES_MASK = provider.getDataFiles(os.path.join(BASE_DIR, 'data/train_files_mask.txt'))
TEST_FILES_MASK = provider.getDataFiles(os.path.join(BASE_DIR, 'data/test_files_mask.txt'))
TRAIN_FILES_INDEX = provider.getDataFiles(os.path.join(BASE_DIR, 'data/train_files_index.txt'))
TEST_FILES_INDEX = provider.getDataFiles(os.path.join(BASE_DIR, 'data/test_files_index.txt'))

def log_string(out_str, LOG_FOUT):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learing_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def train(train_epoch):
    LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train_%d.txt'%train_epoch), 'w')
    LOG_FOUT.write(str(FLAGS) + '\n')
    log_string('train_files:%s' % TRAIN_FILES,LOG_FOUT)
    log_string('test_files:%s' % TEST_FILES,LOG_FOUT)
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl, index_pl, mask_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, NEIGHBOR_NUM)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            pred, end_points = MODEL.get_model_mask(pointclouds_pl, index_pl, mask_pl, is_training_pl, bn_decay=bn_decay)
            loss = MODEL.get_loss(pred, labels_pl, end_points)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        tf.set_random_seed(1)
        sess = tf.Session(config=config)

        # Add summary writers
        #merged = tf.merge_all_summaries()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()

        sess.run(init, {is_training_pl: True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'index_pl': index_pl,
               'mask_pl': mask_pl,
               'pred': (pred,end_points),
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}
        global MAX_TEST_ACC
        global CURRENT_TRAIN_ACC
        global CURRENT_EPOCH
        global CURRENT_TRAIN_EPOCH

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch), LOG_FOUT)
            sys.stdout.flush()

            train_acc = train_one_epoch(sess, ops, train_writer, LOG_FOUT)
            test_acc = eval_one_epoch(sess, ops, test_writer, LOG_FOUT)

            # Save the variables to disk.
            if test_acc > MAX_TEST_ACC:
                MAX_TEST_ACC = test_acc
                CURRENT_TRAIN_ACC = train_acc
                CURRENT_EPOCH = epoch
                CURRENT_TRAIN_EPOCH = train_epoch
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path, LOG_FOUT)
            log_string("Max test acc: %f" % MAX_TEST_ACC, LOG_FOUT)
            #log_string("Current train acc: %f" % CURRENT_TRAIN_ACC, LOG_FOUT)
            log_string("Current epoch: %d" % CURRENT_EPOCH, LOG_FOUT)
            log_string("Max test acc train epoch: %d" % CURRENT_TRAIN_EPOCH, LOG_FOUT)
            log_string("Current train epoch: %d" % train_epoch, LOG_FOUT)
        LOG_FOUT.close()


def train_one_epoch(sess, ops, train_writer,LOG_FOUT):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    # Shuffle train files
    train_file_idxs = np.arange(0, len(TRAIN_FILES))

    np.random.shuffle(train_file_idxs)
    
    for fn in range(len(TRAIN_FILES)):
        log_string('----' + str(fn) + '-----',LOG_FOUT)
        current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[fn]])
        current_mask, current_label = provider.loadDataFile(TRAIN_FILES_MASK[0])
        current_index, current_label = provider.loadDataFile(TRAIN_FILES_INDEX[0])
        current_label = np.squeeze(current_label)
        current_data, current_label, current_index, current_mask = provider.shuffle_data_mask(current_data,current_label,current_index,current_mask)
        file_size = 0
        if (current_data.shape[0] == current_mask.shape[0] == current_index.shape[0] == current_label.shape[0]) and (current_data.shape[1] == current_mask.shape[1] == current_index.shape[1]):
            file_size = current_data.shape[0]
        else:
            print("Train size equal\n")
            exit()
        num_batches = file_size // BATCH_SIZE
        
        last_num = file_size % BATCH_SIZE #19

        total_correct = 0
        total_seen = 0
        loss_sum = 0

        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            
            # Augment batched point clouds by rotation and jittering
            rotated_data = provider.rotate_point_cloud(current_data[start_idx:end_idx, :, :])
            jittered_data = provider.jitter_point_cloud(rotated_data)
            feed_dict = {ops['pointclouds_pl']: jittered_data,
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['index_pl']: current_index[start_idx:end_idx, :, :],
                         ops['mask_pl']:current_mask[start_idx:end_idx, :, :],
                         ops['is_training_pl']: is_training,}
            summary, step, _, loss_val, (pred_val,end_points) = sess.run([ops['merged'], ops['step'],
                ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)

            train_writer.add_summary(summary, step)
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += BATCH_SIZE
            loss_sum += loss_val
        
        
        log_string('mean loss: %f' % (loss_sum / float(num_batches)),LOG_FOUT)
        log_string('accuracy: %f' % (total_correct / float(total_seen)),LOG_FOUT)
    return total_correct / float(total_seen)


def eval_one_epoch(sess, ops, test_writer,LOG_FOUT):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    test_batch_size = BATCH_SIZE

    for fn in range(len(TEST_FILES)):
        log_string('----' + str(fn) + '-----',LOG_FOUT)
        current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
        current_mask, current_label = provider.loadDataFile(TEST_FILES_MASK[0])
        current_index, current_label = provider.loadDataFile(TEST_FILES_INDEX[0])
        current_label = np.squeeze(current_label)
        # current_data, current_label, current_index, current_mask = provider.shuffle_data_mask(current_data,current_label,
        #                                                                                       current_index,current_mask)
        file_size = 0
        if (current_data.shape[0] == current_mask.shape[0] == current_index.shape[0] == current_label.shape[0]) and (
                current_data.shape[1] == current_mask.shape[1] == current_index.shape[1]):
            file_size = current_data.shape[0]
        else:
            print("Test size equal\n")
            exit()

        num_batches = file_size // test_batch_size
        last_num = file_size % test_batch_size
        #last_data = current_data[-1*test_batch_size:-1,:,:]
        #last_label = current_label[num_batches*test_batch_size+1:-1]
        
        #print("last num %d"%last_num)
        for batch_idx in range(num_batches+1):
            start_idx = batch_idx * test_batch_size
            end_idx = (batch_idx+1) * test_batch_size
            if batch_idx == num_batches:
                #print("last 32")
                start_idx = file_size - test_batch_size
                end_idx = file_size
                #print("start_idx %d   end_ idx %d"%(start_idx,end_idx))
            feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['index_pl']: current_index[start_idx:end_idx, :, :],
                         ops['mask_pl']: current_mask[start_idx:end_idx, :, :],
                         ops['is_training_pl']: is_training}
            summary, step, loss_val, (pred_val, end_points) = sess.run([ops['merged'], ops['step'],
                ops['loss'], ops['pred']], feed_dict=feed_dict)
            pred_val = np.argmax(pred_val, 1)
            if batch_idx == num_batches:
                correct = np.sum(pred_val[-1*last_num:] == current_label[-1*last_num:])
                #print(current_label[-1*last_num:])
                total_seen += last_num
            else:
                correct = np.sum(pred_val == current_label[start_idx:end_idx])
                total_seen += BATCH_SIZE
            total_correct += correct
            loss_sum += (loss_val*test_batch_size)
            if batch_idx == num_batches:
                start_idx = file_size - last_num
                #print("start_idx %d"%start_idx)
            #print("total seen %d"%total_seen)
            for i in range(start_idx, end_idx):
                l = current_label[i]
                #print("%d "%l),
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i-start_idx] == l)
        #print total_seen_class
    print("total seen %d"%total_seen)
    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)),LOG_FOUT)
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)),LOG_FOUT)
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))),LOG_FOUT)
    return total_correct / float(total_seen)


if __name__ == "__main__":
    for i in range(40):
        train(i)

