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
			self.cfg.RETRIEVAL.EMBED_DIM, is_training=False, reuse=tf.AUTO_REUSE)
		sent_intermediate_embed_tensor = sent_feature_embed(self.sent_placeholder_test,\
			self.cfg.RETRIEVAL.EMBED_DIM, is_training=False, reuse=tf.AUTO_REUSE)
		
		if self.cfg.RETRIEVAL.ATTENTION:
			image_intermediate_embed_tensor, sent_intermediate_embed_tensor, alpha_tensor = \
				aligned_attention(image_intermediate_embed_tensor, sent_intermediate_embed_tensor,\
				self.cfg.RETRIEVAL.EMBED_DIM,  reuse=tf.AUTO_REUSE, is_training=False, skip=self.cfg.RETRIEVAL.SKIP)
		else:
			image_intermediate_embed_tensor = tf.nn.l2_normalize(image_intermediate_embed_tensor, 1, epsilon=1e-10)
			sent_intermediate_embed_tensor = tf.nn.l2_normalize(sent_intermediate_embed_tensor, 1, epsilon=1e-10)

		# shared layers
		if self.cfg.RETRIEVAL.SHARED:
			self.image_embed_tensor = shared_embed(image_intermediate_embed_tensor,\
				self.cfg.RETRIEVAL.EMBED_DIM, reuse=tf.AUTO_REUSE, is_training=False)
			self.sent_embed_tensor = shared_embed(sent_intermediate_embed_tensor,\
				self.cfg.RETRIEVAL.EMBED_DIM, reuse=tf.AUTO_REUSE, is_training=False)
		else:
			self.image_embed_tensor = image_intermediate_embed_tensor
			self.sent_embed_tensor = sent_intermediate_embed_tensor	
		# return self.image_embed_tensor, self.sent_embed_tensor

