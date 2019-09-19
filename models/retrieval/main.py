from __future__ import print_function
from __future__ import absolute_import 
from __future__ import division

import numpy as np
import tensorflow as tf
import os
import pickle as pkl
import pdb

from models.retrieval.model import visual_feature_embed, sent_feature_embed, shared_embed
from models.retrieval.model import aligned_attention
from models.retrieval.loss import triplet_loss, category_loss, modality_loss, lifted_structured_loss
from models.retrieval.datautils import DatasetLoader
from models.retrieval.moreutils import tsneVis, compClusterScores, pairwise_distances
from models.retrieval.moreutils import compMapScore
# from moreutils import matlab_mapk as compMapScore
from models.retrieval.moreutils import compute_nmi, recall_at_k
from models.retrieval.moreutils import ModelParameters
from models.retrieval.data_paths import paths


class Retrieval(object):

	def __init__(self, cfg):
		
		# Training dataloader
		self.cfg = cfg
		# self.train_data_loader = DatasetLoader(cfg.RETIEVAL.DATA_PATH)
		self.im_feat_dim = cfg.RETRIEVAL.im_feat_shape
		self.sent_feat_dim = cfg.RETRIEVAL.sent_feat_shape


	def build_model(self):
	# with tf.Graph().as_default():
		# Setup placeholders for training variables.
		self.image_placeholder = tf.placeholder(dtype=tf.float32,\
				shape=[self.cfg.RETRIEVAL.BATCH_SIZE, self.im_feat_dim])
		self.sent_placeholder = tf.placeholder(dtype=tf.float32,\
				shape=[self.cfg.RETRIEVAL.BATCH_SIZE, self.sent_feat_dim])
		# self.lambda_placeholder = tf.placeholder(tf.float32, [])
		self.label_placeholder = tf.placeholder(dtype=tf.int32, shape=[self.cfg.RETRIEVAL.BATCH_SIZE])
		
		# create embedding model
		image_intermediate_embed_tensor = visual_feature_embed(self.image_placeholder, \
				self.cfg.RETRIEVAL.EMBED_DIM, dropout_ratio=self.cfg.RETRIEVAL.DROPOUT)
		sent_intermediate_embed_tensor = sent_feature_embed(self.sent_placeholder, \
				self.cfg.RETRIEVAL.EMBED_DIM, dropout_ratio=self.cfg.RETRIEVAL.DROPOUT)

		if self.cfg.RETRIEVAL.ATTENTION:
			image_intermediate_embed_tensor, sent_intermediate_embed_tensor, _ = \
					aligned_attention(image_intermediate_embed_tensor, sent_intermediate_embed_tensor, \
					self.cfg.RETRIEVAL.EMBED_DIM, skip=self.cfg.RETRIEVAL.SKIP)
		else:
			image_intermediate_embed_tensor = tf.nn.l2_normalize(image_intermediate_embed_tensor,\
					   1, epsilon=1e-10)
			sent_intermediate_embed_tensor = tf.nn.l2_normalize(sent_intermediate_embed_tensor, 1,\
						epsilon=1e-10)
		
		# shared layers
		if self.cfg.RETRIEVAL.SHARED:
			image_embed_tensor = shared_embed(image_intermediate_embed_tensor,\
					self.cfg.RETRIEVAL.EMBED_DIM, dropout_ratio=self.cfg.RETRIEVAL.DROPOUT)
			sent_embed_tensor = shared_embed(sent_intermediate_embed_tensor, \
					self.cfg.RETRIEVAL.EMBED_DIM, reuse=True, dropout_ratio=self.cfg.RETRIEVAL.DROPOUT)
		else:
			image_embed_tensor = image_intermediate_embed_tensor
			sent_embed_tensor = sent_intermediate_embed_tensor
		
		# category loss
		class_loss = category_loss(image_embed_tensor, sent_embed_tensor,\
				self.label_placeholder, self.cfg.RETRIEVAL.NumCategories)
		
		# metric loss
		metric_loss = triplet_loss(image_embed_tensor, sent_embed_tensor, \
				self.label_placeholder, self.cfg.RETRIEVAL.MARGIN)

		emb_loss = self.cfg.RETRIEVAL.CategoryScale*class_loss + self.cfg.RETRIEVAL.MetricScale*metric_loss
		# scaled_modal_loss = self.cfg.RETRIEVAL.modalityScale*modal_loss

		total_loss = self.cfg.RETRIEVAL.CategoryScale*class_loss + \
					self.cfg.RETRIEVAL.MetricScale*metric_loss 

		# scopes for different functions to separate learning
		t_vars = tf.trainable_variables()
		# pdb.set_trace()
		visfeat_vars = [v for v in t_vars if 'vf_' in v.name] # only visual embedding layers
		sentfeat_vars = [v for v in t_vars if 'sf_' in v.name] # only sent embedding layers
		sharedfeat_vars = [v for v in t_vars if 'se_' in v.name] # shared embedding layers
		attention_vars = [v for v in t_vars if 'att' in v.name] # only attention weights	
		catclas_vars = [v for v in t_vars if 'cc_' in v.name]
		self.var_list = visfeat_vars+sentfeat_vars+sharedfeat_vars+catclas_vars+attention_vars
		return image_embed_tensor, sent_embed_tensor, total_loss

	def eval(self, sample_size=None):

		# Setup placeholders for input variables.
		if sample_size is not None:
			sample_size = sample_size
		else:	
			sample_size = self.cfg.RETRIEVAL.SAMPLE_NUM
		self.image_placeholder_test = tf.placeholder(dtype=tf.float32,\
				shape=[sample_size, self.im_feat_dim])
		self.sent_placeholder_test = tf.placeholder(dtype=tf.float32,
				shape=[sample_size, self.sent_feat_dim])
		self.label_placeholder_test = tf.placeholder(dtype=tf.int32, 
				shape=[sample_size])

		# create model
		image_intermediate_embed_tensor = visual_feature_embed(self.image_placeholder_test,\
			self.cfg.RETRIEVAL.EMBED_DIM, is_training=False, reuse=True)
		sent_intermediate_embed_tensor = sent_feature_embed(self.sent_placeholder_test,\
			self.cfg.RETRIEVAL.EMBED_DIM, is_training=False, reuse=True)
		
		if self.cfg.RETRIEVAL.ATTENTION:
			image_intermediate_embed_tensor, sent_intermediate_embed_tensor, alpha_tensor = \
				aligned_attention(image_intermediate_embed_tensor, sent_intermediate_embed_tensor,\
				self.cfg.RETRIEVAL.EMBED_DIM,  reuse=True, is_training=False, skip=self.cfg.RETRIEVAL.SKIP)
		else:
			image_intermediate_embed_tensor = tf.nn.l2_normalize(image_intermediate_embed_tensor, 1, epsilon=1e-10)
			sent_intermediate_embed_tensor = tf.nn.l2_normalize(sent_intermediate_embed_tensor, 1, epsilon=1e-10)

		# shared layers
		if self.cfg.RETRIEVAL.SHARED:
			self.image_embed_tensor = shared_embed(image_intermediate_embed_tensor,\
				self.cfg.RETRIEVAL.EMBED_DIM, reuse=True, is_training=False)
			self.sent_embed_tensor = shared_embed(sent_intermediate_embed_tensor,\
				self.cfg.RETRIEVAL.EMBED_DIM, reuse=True, is_training=False)
		else:
			self.image_embed_tensor = image_intermediate_embed_tensor
			self.sent_embed_tensor = sent_intermediate_embed_tensor	
		# return self.image_embed_tensor, self.sent_embed_tensor
		

def eval():
	test_data_loader = DatasetLoader(self.cfg.RETRIEVAL.image_feat_path_test, self.cfg.RETRIEVAL.sent_feat_path_test, self.cfg.RETRIEVAL.label_path_test)
	test_num_samples = test_data_loader.no_samples
	im_feat_dim = test_data_loader.im_feat_shape
	sent_feat_dim = test_data_loader.sent_feat_shape
	self.cfg.RETRIEVAL.batchSize = test_num_samples

	# steps_per_epoch = test_num_samples // self.cfg.RETRIEVAL.batchSize
	steps_per_epoch = test_num_samples # one image at a time

	with tf.Graph().as_default():
		# Setup placeholders for input variables.
		image_placeholder = tf.placeholder(dtype=tf.float32, shape=[self.cfg.RETRIEVAL.batchSize, im_feat_dim])
		sent_placeholder = tf.placeholder(dtype=tf.float32, shape=[self.cfg.RETRIEVAL.batchSize, sent_feat_dim])
		label_placeholder = tf.placeholder(dtype=tf.int32, shape=[self.cfg.RETRIEVAL.batchSize])

		# create model
		image_intermediate_embed_tensor = visual_feature_embed(image_placeholder, self.cfg.RETRIEVAL.embedDim, is_training=False)
		sent_intermediate_embed_tensor = sent_feature_embed(sent_placeholder, self.cfg.RETRIEVAL.embedDim, is_training=False)
		
		if self.cfg.RETRIEVAL.attention:
			image_intermediate_embed_tensor, sent_intermediate_embed_tensor, alpha_tensor = aligned_attention(image_intermediate_embed_tensor, sent_intermediate_embed_tensor, self.cfg.RETRIEVAL.embedDim, reuse=tf.AUTO_REUSE, is_training=False, skip=self.cfg.RETRIEVAL.skip)
		else:
			image_intermediate_embed_tensor = tf.nn.l2_normalize(image_intermediate_embed_tensor, 1, epsilon=1e-10)
			sent_intermediate_embed_tensor = tf.nn.l2_normalize(sent_intermediate_embed_tensor, 1, epsilon=1e-10)

		# shared layers
		if self.cfg.RETRIEVAL.shared:
			image_embed_tensor = shared_embed(image_intermediate_embed_tensor, self.cfg.RETRIEVAL.embedDim, is_training=False)
			sent_embed_tensor = shared_embed(sent_intermediate_embed_tensor, self.cfg.RETRIEVAL.embedDim, reuse=True, is_training=False)
		else:
			image_embed_tensor = image_intermediate_embed_tensor
			sent_embed_tensor = sent_intermediate_embed_tensor
		
		saver = tf.train.Saver()
		session_config = tf.ConfigProto()
		session_config.gpu_options.allow_growth = True
		
		with tf.Session(config=session_config) as sess:
			sess.run([tf.global_variables_initializer()])
			print('restoring checkpoint')
			saver.restore(sess, tf.train.latest_checkpoint(self.cfg.RETRIEVAL.ExperimentDirectory))
			
			all_labels = []
			sim_mat = np.zeros((test_num_samples,test_num_samples))
			
			for i in range(steps_per_epoch):
			
				# im_feats, sent_feats, labels = test_data_loader.get_batch(i, self.cfg.RETRIEVAL.batchSize, phase = 'test')
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

			print('Image to Sentence mAP@50: ',i2s_mapk,'\n',
				 'Sentence to Image mAP@50: ',s2i_mapk,'\n',)
				 
			with open(os.path.join(self.cfg.RETRIEVAL.ExperimentDirectory, 'scores1.txt'), 'w') as f:
				print('Image to Sentence mAP@50: ',i2s_mapk,'\n',
					 'Sentence to Image mAP@50: ',s2i_mapk,'\n',
					  file=f)

def eval_new():
	train_data_loader = DatasetLoader(self.cfg.RETRIEVAL.image_feat_path_train, self.cfg.RETRIEVAL.sent_feat_path_train, self.cfg.RETRIEVAL.label_path_train)
	test_num_samples = train_data_loader.no_samples
	im_feat_dim = train_data_loader.im_feat_shape
	sent_feat_dim = train_data_loader.sent_feat_shape
	self.cfg.RETRIEVAL.batchSize = 64#test_num_samples

	steps_per_epoch = test_num_samples // self.cfg.RETRIEVAL.batchSize
	# steps_per_epoch = test_num_samples # one image at a time

	with tf.Graph().as_default():
		# Setup placeholders for input variables.
		image_placeholder_val = tf.placeholder(dtype=tf.float32, shape=[self.cfg.RETRIEVAL.batchSize, im_feat_dim])
		sent_placeholder_val = tf.placeholder(dtype=tf.float32, shape=[self.cfg.RETRIEVAL.batchSize, sent_feat_dim])
		label_placeholder_val = tf.placeholder(dtype=tf.int32, shape=[self.cfg.RETRIEVAL.batchSize])

		# create model
		image_intermediate_embed_tensor = visual_feature_embed(image_placeholder_val, self.cfg.RETRIEVAL.embedDim, is_training=False)
		sent_intermediate_embed_tensor = sent_feature_embed(sent_placeholder_val, self.cfg.RETRIEVAL.embedDim, is_training=False)
		
		if self.cfg.RETRIEVAL.attention:
			image_intermediate_embed_tensor, sent_intermediate_embed_tensor, alpha_tensor = aligned_attention(image_intermediate_embed_tensor, sent_intermediate_embed_tensor, self.cfg.RETRIEVAL.embedDim, reuse=tf.AUTO_REUSE, is_training=False, skip=self.cfg.RETRIEVAL.skip)
		else:
			image_intermediate_embed_tensor = tf.nn.l2_normalize(image_intermediate_embed_tensor, 1, epsilon=1e-10)
			sent_intermediate_embed_tensor = tf.nn.l2_normalize(sent_intermediate_embed_tensor, 1, epsilon=1e-10)

		# shared layers
		if self.cfg.RETRIEVAL.shared:
			image_embed_tensor = shared_embed(image_intermediate_embed_tensor, self.cfg.RETRIEVAL.embedDim, is_training=False)
			sent_embed_tensor = shared_embed(sent_intermediate_embed_tensor, self.cfg.RETRIEVAL.embedDim, reuse=True, is_training=False)
		else:
			image_embed_tensor = image_intermediate_embed_tensor
			sent_embed_tensor = sent_intermediate_embed_tensor
		
		saver = tf.train.Saver()
		session_config = tf.ConfigProto()
		session_config.gpu_options.allow_growth = True
		
		with tf.Session(config=session_config) as sess:
			sess.run([tf.global_variables_initializer()])
			print('restoring checkpoint')
			saver.restore(sess, tf.train.latest_checkpoint(self.cfg.RETRIEVAL.ExperimentDirectory))
			
			all_img_emb = np.zeros((test_num_samples, 512))	
			all_cap_emb = np.zeros((test_num_samples, 512))	
			all_labels = np.zeros((test_num_samples))
			sim_mat = np.zeros((test_num_samples,test_num_samples))
			
			for i in range(steps_per_epoch):
			
				im_feats, sent_feats, labels = train_data_loader.get_batch(i, self.cfg.RETRIEVAL.batchSize, phase = 'test')
				# im_feats, sent_feats, labels = test_data_loader.get_batch(i, 1, phase = 'test')

				# im_feats_rep = np.repeat(im_feats,test_num_samples,0)
				# labels_rep = np.repeat(labels,test_num_samples,0)
				
				# feed_dict = {image_placeholder_val : im_feats_rep, sent_placeholder_val : sent_feats, label_placeholder_val : labels_rep}
				feed_dict = {image_placeholder_val : im_feats, sent_placeholder_val : sent_feats, label_placeholder_val : labels}

				image_embed, sent_embed = sess.run([image_embed_tensor, sent_embed_tensor], feed_dict = feed_dict)
				all_labels[i*self.cfg.RETRIEVAL.batchSize:(i+1)*self.cfg.RETRIEVAL.batchSize] = labels
				all_img_emb[i*self.cfg.RETRIEVAL.batchSize:(i+1)*self.cfg.RETRIEVAL.batchSize] = image_embed
				all_cap_emb[i*self.cfg.RETRIEVAL.batchSize:(i+1)*self.cfg.RETRIEVAL.batchSize] = sent_embed
				
				np.save('/shared/kgcoe-research/mil/txt2img/birds/train/cvs_img_emb_512.npy',all_img_emb)
				np.save('/shared/kgcoe-research/mil/txt2img/birds/train/cvs_txt_emb_512.npy',all_cap_emb)
				# import pickle
				# pickle.dump(all_img_emb,open('/shared/kgcoe-research/mil/txt2img/flowers/train/cvs_img_emb_512.pickle','w'))
				# pickle.dump(all_cap_emb,open('/shared/kgcoe-research/mil/txt2img/flowers/train/cvs_txt_emb_512.pickle','w'))
			for j in range(test_num_samples):
				# pdb.set_trace()
				sim_mat[j,:] = pairwise_distances(all_img_emb[j,:].reshape(1, -1) , all_cap_emb)#,'cosine')	
				
				# sim_mat[i,:] = pairwise_distances(image_embed[0,:].reshape(1,-1),sent_embed)#,'cosine')

				# all_labels.extend(labels)
				# if i % 100 == 0:
					# print('Done: '+str(i)+' of '+str(steps_per_epoch))
			
			i2s_mapk = compMapScore(sim_mat, all_labels)
			s2i_mapk = compMapScore(sim_mat.T, all_labels)
			
			print('Image to Sentence mAP@50: ',i2s_mapk,'\n',
					 'Sentence to Image mAP@50: ',s2i_mapk,'\n',)

			with open(os.path.join(self.cfg.RETRIEVAL.ExperimentDirectory, 'scores1.txt'), 'w') as f:
				print('Image to Sentence mAP@50: ',i2s_mapk,'\n',
					 'Sentence to Image mAP@50: ',s2i_mapk,'\n',
					  file=f)	
	
	
# train()
# eval_new()

