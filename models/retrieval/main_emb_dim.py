from __future__ import print_function
from __future__ import absolute_import 
from __future__ import division

import numpy as np
import tensorflow as tf
import os
import pickle as pkl
import pdb

from model import visual_feature_embed, sent_feature_embed, shared_embed
from model import aligned_attention
from loss import triplet_loss, category_loss, modality_loss, lifted_structured_loss
from datautils import DatasetLoader
from moreutils import tsneVis, compClusterScores, pairwise_distances
from moreutils import compMapScore
# from moreutils import matlab_mapk as compMapScore
from moreutils import compute_nmi, recall_at_k
from moreutils import ModelParameters
from data_paths import paths

# parameters
params = ModelParameters()

params = paths(params, 'xmedianet') # 'flowers_simple', 'flowers_zs', 'nuswide', 'pascal', 'birds_zs', 'wiki_orig', 'wiki', 'xmedianet'

params.ExperimentDirectory = './experiment_xmedianet/512'
params.batchSize = 128
params.maxEpoch = 20
params.initLR = 0.001
params.Restore = False
params.printEvery = 10
params.embedDim = 512
params.categoryScale = 1.0
params.metricScale = 1.0
params.modalityScale = 0.0
params.margin = 1.0
params.saveEvery = 2
params.dropout = 1.0
params.shared = False
params.attention = True
params.skip = True
params.image_aug = True
params.createDir()
params.saveParams()

def train():
    train_data_loader = DatasetLoader(params.image_feat_path_train, params.sent_feat_path_train, params.label_path_train)
    im_feat_dim = train_data_loader.im_feat_shape
    sent_feat_dim = train_data_loader.sent_feat_shape
    train_num_samples = train_data_loader.no_samples
    steps_per_epoch = train_num_samples // params.batchSize
    num_steps = steps_per_epoch * params.maxEpoch
    
    with tf.Graph().as_default():
        # Setup placeholders for input variables.
        image_placeholder = tf.placeholder(dtype=tf.float32, shape=[params.batchSize, im_feat_dim])
        sent_placeholder = tf.placeholder(dtype=tf.float32, shape=[params.batchSize, sent_feat_dim])
        label_placeholder = tf.placeholder(dtype=tf.int32, shape=[params.batchSize])
        # is_training_placeholder = tf.placeholder(tf.bool)
        
        # create embedding model
        image_intermediate_embed_tensor = visual_feature_embed(image_placeholder, params.embedDim, dropout_ratio=params.dropout)#, is_training=is_training_placeholder)
        sent_intermediate_embed_tensor = sent_feature_embed(sent_placeholder, params.embedDim, dropout_ratio=params.dropout)#, is_training=is_training_placeholder)

        if params.attention:
            image_intermediate_embed_tensor, sent_intermediate_embed_tensor, _ = aligned_attention(image_intermediate_embed_tensor, sent_intermediate_embed_tensor, params.embedDim, skip=params.skip)#, is_training=is_training_placeholder)
        else:
            image_intermediate_embed_tensor = tf.nn.l2_normalize(image_intermediate_embed_tensor, 1, epsilon=1e-10)
            sent_intermediate_embed_tensor = tf.nn.l2_normalize(sent_intermediate_embed_tensor, 1, epsilon=1e-10)
        # shared layers

        if params.shared:
            image_embed_tensor = shared_embed(image_intermediate_embed_tensor, params.embedDim, dropout_ratio=params.dropout)
            sent_embed_tensor = shared_embed(sent_intermediate_embed_tensor, params.embedDim, reuse=True, dropout_ratio=params.dropout)
        else:
            image_embed_tensor = image_intermediate_embed_tensor
            sent_embed_tensor = sent_intermediate_embed_tensor

        # category loss
        class_loss = category_loss(image_embed_tensor, sent_embed_tensor, label_placeholder, params.numCategories)
        
        # metric loss
        metric_loss = triplet_loss(image_embed_tensor, sent_embed_tensor, label_placeholder, params.margin)
        # metric_loss, _ = batch_all_triplet_loss(tf.concat([label_placeholder, label_placeholder], axis = 0), tf.concat([image_embed_tensor, sent_embed_tensor], axis = 0), params.margin, squared=True)
        
        # modality loss
        # modal_loss = modality_loss(image_embed_tensor, sent_embed_tensor, lambda_placeholder, params.embedDim, params.batchSize)
        
        # total loss
        
        # total_loss = tf.reduce_mean(tf.reduce_sum(tf.square(image_embed_tensor-sent_embed_tensor),1))
        emb_loss = params.categoryScale*class_loss + params.metricScale*metric_loss
        # scaled_modal_loss = params.modalityScale*modal_loss

        total_loss = params.categoryScale*class_loss + params.metricScale*metric_loss #+ params.modalityScale*modal_loss
        
        # scopes for different functions to separate learning
        t_vars = tf.trainable_variables()
        # pdb.set_trace()
        visfeat_vars = [v for v in t_vars if 'vf_' in v.name] # only visual embedding layers
        sentfeat_vars = [v for v in t_vars if 'sf_' in v.name] # only sent embedding layers
        sharedfeat_vars = [v for v in t_vars if 'se_' in v.name] # shared embedding layers
        attention_vars = [v for v in t_vars if 'att' in v.name] # only attention weights
        
        catclas_vars = [v for v in t_vars if 'cc_' in v.name]
        # modclas_vars = [v for v in t_vars if 'mc_' in v.name]
        
        tf.summary.scalar('metric loss', metric_loss)
        # tf.summary.scalar('modality loss', modal_loss)
        tf.summary.scalar('category loss', class_loss)
        tf.summary.scalar('total loss', total_loss)
        
        global_step_tensor = tf.Variable(0, trainable=False)
        
        # learning_rate = tf.train.exponential_decay(initLR, global_step_tensor, steps_per_epoch, 0.9, staircase=True)
        # optimizer  = tf.train.AdamOptimizer(initLR)
        
        # gvs = optimizer.compute_gradients(total_loss)
        # capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gvs]
        # train_op = optimizer.apply_gradients(capped_gvs, global_step_tensor)

        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
            # train_op = optimizer.minimize(total_loss, global_step_tensor)
        
        # total_train_op = tf.train.AdamOptimizer(learning_rate=params.initLR).minimize(total_loss, global_step=global_step_tensor)
        
        emb_train_op = tf.train.AdamOptimizer(learning_rate=params.initLR).minimize(emb_loss, global_step=global_step_tensor, var_list=visfeat_vars+sentfeat_vars+sharedfeat_vars+catclas_vars+attention_vars)
        # modal_train_op = tf.train.AdamOptimizer(learning_rate=params.initLR).minimize(scaled_modal_loss, var_list=modclas_vars)

        saver = tf.train.Saver(max_to_keep=10)
        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True
        summary_tensor = tf.summary.merge_all()
        
        with tf.Session(config=session_config) as sess:
            summary_writer = tf.summary.FileWriter(params.ExperimentDirectory, graph=tf.get_default_graph())
            sess.run([tf.global_variables_initializer()])
            if params.Restore:
                print('restoring checkpoint')
                saver.restore(sess, tf.train.latest_checkpoint(params.ExperimentDirectory))
            
            # i2s_mapk_all = []
            # s2i_mapk_all = []
            
            for i in range(num_steps):
            
                if i % steps_per_epoch == 0:
                    train_data_loader.shuffle_inds()
                
                im_feats, sent_feats, labels = train_data_loader.get_batch(i % steps_per_epoch, params.batchSize, image_aug = params.image_aug)
                
                feed_dict = {image_placeholder : im_feats, sent_placeholder : sent_feats, label_placeholder : labels}

                _, summary, global_step, loss_total, loss_class, loss_metric = sess.run(
                                                            [emb_train_op, summary_tensor, global_step_tensor, 
                                                            total_loss, class_loss, metric_loss], feed_dict = feed_dict)
                
                
                summary_writer.add_summary(summary, global_step)
                
                if i % params.printEvery == 0:
                    print('Epoch: %d | Step: %d | Total Loss: %f | Class Loss: %f | Metric Loss: %f' % (i // steps_per_epoch, i, loss_total, loss_class, loss_metric))
                
                if (i % (steps_per_epoch * params.saveEvery) == 0 and i > 0) or (i == num_steps-1):
                    print('Saving checkpoint at step %d' % i)
                    saver.save(sess, os.path.join(params.ExperimentDirectory, 'model.ckpt'+str(global_step)))

                # if (i % (steps_per_epoch * 5) == 0 and i > 0):
                    # pdb.set_trace()
                    # (i2s_mapk, s2i_mapk) = eval_every(sess, image_placeholder, sent_placeholder, label_placeholder)
                    # i2s_mapk_all.append(i2s_mapk)
                    # s2i_mapk_all.append(s2i_mapk)

def eval(checkpoint_path, f):
    test_data_loader = DatasetLoader(params.image_feat_path_test, params.sent_feat_path_test, params.label_path_test)
    test_num_samples = test_data_loader.no_samples
    im_feat_dim = test_data_loader.im_feat_shape
    sent_feat_dim = test_data_loader.sent_feat_shape
    params.batchSize = test_num_samples
    
    # steps_per_epoch = test_num_samples // params.batchSize
    steps_per_epoch = test_num_samples # one image at a time
    
    with tf.Graph().as_default():
        # Setup placeholders for input variables.
        image_placeholder = tf.placeholder(dtype=tf.float32, shape=[params.batchSize, im_feat_dim])
        sent_placeholder = tf.placeholder(dtype=tf.float32, shape=[params.batchSize, sent_feat_dim])
        label_placeholder = tf.placeholder(dtype=tf.int32, shape=[params.batchSize])

        # create model
        image_intermediate_embed_tensor = visual_feature_embed(image_placeholder, params.embedDim, is_training=False)
        sent_intermediate_embed_tensor = sent_feature_embed(sent_placeholder, params.embedDim, is_training=False)
        
        if params.attention:
            image_intermediate_embed_tensor, sent_intermediate_embed_tensor, alpha_tensor = aligned_attention(image_intermediate_embed_tensor, sent_intermediate_embed_tensor, params.embedDim, reuse=tf.AUTO_REUSE, is_training=False, skip=params.skip)
        else:
            image_intermediate_embed_tensor = tf.nn.l2_normalize(image_intermediate_embed_tensor, 1, epsilon=1e-10)
            sent_intermediate_embed_tensor = tf.nn.l2_normalize(sent_intermediate_embed_tensor, 1, epsilon=1e-10)

        if params.shared:
            image_embed_tensor = shared_embed(image_intermediate_embed_tensor, params.embedDim, dropout_ratio=params.dropout)
            sent_embed_tensor = shared_embed(sent_intermediate_embed_tensor, params.embedDim, reuse=True, dropout_ratio=params.dropout)
        else:
            image_embed_tensor = image_intermediate_embed_tensor
            sent_embed_tensor = sent_intermediate_embed_tensor

        saver = tf.train.Saver()
        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True
        
        with tf.Session(config=session_config) as sess:
            sess.run([tf.global_variables_initializer()])
            print('restoring checkpoint: '+checkpoint_path)
            # saver.restore(sess, tf.train.latest_checkpoint(params.ExperimentDirectory))
            saver.restore(sess, os.path.join(params.ExperimentDirectory,checkpoint_path))
            
            all_labels = []
            sim_mat = np.zeros((test_num_samples,test_num_samples))
            
            for i in range(steps_per_epoch):
            
                # im_feats, sent_feats, labels = test_data_loader.get_batch(i, params.batchSize, phase = 'test')
                im_feats, sent_feats, labels = test_data_loader.get_batch(i, 1, phase = 'test')
                
                im_feats_rep = np.repeat(im_feats,test_num_samples,0)
                labels_rep = np.repeat(labels,test_num_samples,0)
                
                feed_dict = {image_placeholder : im_feats_rep, sent_placeholder : sent_feats, label_placeholder : labels_rep}

                image_embed, sent_embed = sess.run([image_embed_tensor, sent_embed_tensor], feed_dict = feed_dict)
                sim_mat[i,:] = pairwise_distances(image_embed[0,:].reshape(1,-1),sent_embed)#,'cosine')

                all_labels.extend(labels)
                if i % 100 == 0:
                    print('Done: '+str(i)+' of '+str(steps_per_epoch))
            
            i2s_mapk = compMapScore(sim_mat, all_labels)
            s2i_mapk = compMapScore(sim_mat.T, all_labels)
            
            print(checkpoint_path,'\n',
                  'Image to Sentence mAP@50: ',i2s_mapk,'\n',
                  'Sentence to Image mAP@50: ',s2i_mapk,'\n',
                  file=f)


train()

checkpoints = open(os.path.join(params.ExperimentDirectory,'checkpoint'),'r').readlines()
with open(os.path.join(params.ExperimentDirectory, 'scores.txt'), 'w') as f:
    for checkpoint in checkpoints[1:]:
        checkpoint_path = checkpoint.split(':')[1].strip()[1:-1]
        eval(checkpoint_path,f)

