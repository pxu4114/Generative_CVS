import torch
import cv2
import glob
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

import pdb

class Data_loader(Dataset):
	"""
	Dataset: extracting labels and images
	"""

	def __init__(self, data_path, data_name, num, phase='train', transform=None):
		self.phase = phase
		self.data_name = data_name
		self.img_path = np.load(os.path.join(data_path, 'image_%s.npy'%self.phase))
		self.text_path = np.load(os.path.join(data_path, 'text_%s.npy'%self.phase))
		self.label_path = np.load(os.path.join(data_path, 'label_%s.npy'%self.phase))
		self.transform = transform
		self.num = num
		
	def __getitem__(self, index):
		image_aug=False
		if self.phase == 'train':
			if image_aug:
				# pdb.set_trace()
				img_path = self.img_path[index,np.random.randint(0, 10),:]
			elif not image_aug:
				img_path = self.img_path[index,0,:]
			if self.data_name == 'pascal':
				text_path = self.text_path[index,np.random.randint(0, 5),:] 
			elif self.data_name == 'birds' or 'flowers':    
				text_path = self.text_path[index,np.random.randint(0, 10),:]
			elif self.data_name == 'nuswide':
				text_path = self.text_path[index,0,:]
		else:
			# pdb.set_trace()
			if len(self.img_path.shape) == 2:
				img_path = self.img_path[index,:] # only one feature per image (no 10x augmentation)
			else:
				img_path = self.img_path[index,0,:]
			if len(self.text_path.shape) == 2:
				text_path = self.text_path[index,:] # test with only one sentence
				# text_path = self.text_path
			else:
				# text_path = self.text_path[index,np.random.randint(0, self.sent_im_ratio),:]
				text_path = self.text_path[index,0,:]
				# text_path = self.text_path[:,0,:]
				# text_path = np.mean(self.text_path[:,:,:],1)
		
		labels = self.label_path[index]
		labels = labels.astype(np.int32)
		return(img_path, text_path, labels)	
		
		return img, text, label

	def __len__(self):
		if self.num is None:
			return self.label_path.shape[0]
		else: return self.num	