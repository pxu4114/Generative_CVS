from __future__ import division
from __future__ import print_function

import numpy as np
import os
import pdb
import pickle as pkl

np.random.seed(999)

class DatasetLoader():
	""" Dataset loader class that loads feature matrices from given paths and
		create shuffled batch for training, unshuffled batch for evaluation.
	"""
	# def __init__(self, im_feat_path, sent_feat_path, label_path):
	def __init__(self, data_path, mode):
		
		im_feat_path = os.path.join(data_path,'image_%s.npy'%mode)
		sent_feat_path = os.path.join(data_path,'text_%s.npy'%mode)
		label_path = os.path.join(data_path,'label_%s.npy'%mode)
		labels = np.load(label_path)
		self.no_samples = labels.shape[0]
		self._epochs_completed = -1		
		self._index_in_epoch = self.no_samples
		self.no_categories = len(np.unique(labels))
		print('No. of unique categories is', self.no_categories)

		print('Loading image features from', im_feat_path)
		data_im = np.load(im_feat_path)
		im_feats = data_im.astype(np.float32)
		if len(im_feats.shape) == 2:
			im_feats = np.reshape(im_feats,(im_feats.shape[0],1,im_feats.shape[1]))
		print('Loaded image feature shape:', im_feats.shape)
		
		print('Loading sentence features from', sent_feat_path)
		data_sent = np.load(sent_feat_path)
		sent_feats = data_sent.astype(np.float32)
		if len(sent_feats.shape) == 2:
			sent_feats = np.reshape(sent_feats,(sent_feats.shape[0],1,sent_feats.shape[1]))
		print('Loaded sentence feature shape:', sent_feats.shape)

		self.im_feat_shape = im_feats.shape[-1]
		self.sent_feat_shape = sent_feats.shape[-1]
		
		# self.sample_inds = range(labels.shape[0]) # we will shuffle this every epoch for training
		self.sample_inds = list(range(labels.shape[0])) # we will shuffle this every epoch for training
		self.im_feats = im_feats
		self.sent_feats = sent_feats
		self.labels = labels
		# Assume the number of sentence per image is a constant.
		self.sent_im_ratio = sent_feats.shape[1]
		self.im_aug_num = im_feats.shape[1]

	def shuffle_inds(self):
		'''
		shuffle the indices in training (run this once per epoch)
		nop for testing and validation
		'''
		np.random.shuffle(self.sample_inds)

	def get_batch(self, batch_index, batch_size, phase = 'train', image_aug = True):

		if phase == 'incep':
			start_ind = self._index_in_epoch
			self._index_in_epoch += batch_size

			if self._index_in_epoch > self.no_samples:
				# Finished epoch
				self._epochs_completed += 1
				# Shuffle the .data
				self._perm = np.arange(self.no_samples)
				# np.random.shuffle(self._perm)
				self.shuffle_inds()

				# Start next epoch
				start_ind = 0
				self._index_in_epoch = batch_size
				assert batch_size <= self.no_samples
			end_ind = self._index_in_epoch
		
		else:
			start_ind = batch_index * batch_size
			end_ind = start_ind + batch_size
			
		sample_inds = self.sample_inds[start_ind : end_ind]
			
		if phase == 'train':
			if image_aug:
				im_feats = self.im_feats[sample_inds,np.random.randint(0, self.im_aug_num),:]
			elif not image_aug:
				im_feats = self.im_feats[sample_inds,0,:]
			sent_feats = self.sent_feats[sample_inds,np.random.randint(0, self.sent_im_ratio),:] 
		elif phase == 'test' or phase == 'incep':
			if len(self.im_feats.shape) == 2:
				im_feats = self.im_feats[sample_inds,:] # only one feature per image (no 10x augmentation)
			else:
				im_feats = self.im_feats[sample_inds,0,:]
			if len(self.sent_feats.shape) == 2:
				# sent_feats = self.sent_feats[sample_inds,:] # test with only one sentence
				sent_feats = self.sent_feats
			else:
				# sent_feats = self.sent_feats[sample_inds,np.random.randint(0, self.sent_im_ratio),:]
				sent_feats = self.sent_feats[sample_inds,0,:]
				# sent_feats = self.sent_feats[:,0,:]
				# sent_feats = np.mean(self.sent_feats[:,:,:],1)
		elif phase=='eval':
			if len(self.im_feats.shape) == 2:
				im_feats = self.im_feats[sample_inds,:] # only one feature per image (no 10x augmentation)
			else:
				im_feats = self.im_feats[sample_inds,0,:]
			if len(self.sent_feats.shape) == 2:
				# sent_feats = self.sent_feats[sample_inds,:] # test with only one sentence
				sent_feats = self.sent_feats
			else:
				# sent_feats = self.sent_feats[sample_inds,np.random.randint(0, self.sent_im_ratio),:]
				# sent_feats = self.sent_feats[sample_inds,0,:]
				sent_feats = self.sent_feats[:,0,:]
				# sent_feats = np.mean(self.sent_feats[:,:,:],1)            
		labels = self.labels[sample_inds]
		labels = labels.astype(np.int32)
		return(im_feats, sent_feats, labels)
        
