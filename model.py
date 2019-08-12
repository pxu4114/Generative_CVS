import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.functional import softmax
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from itertools import combinations

import pdb

def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X

class EncodeImage(nn.Module):

    def __init__(self,image_dim, embed_size, use_abs=False, no_imgnorm=False):
        super(EncodeImage, self).__init__()
        self.embed_size = embed_size
        self.use_abs = use_abs
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(image_dim, embed_size)
        self.fc1 = nn.Linear(embed_size, embed_size)
        self.fc2 = nn.Linear(embed_size, embed_size)
        self.tanh = nn.Tanh() 
        self.dropout = nn.Dropout(p=0.05)
        self.init_weights()
        
    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)         
        self.fc1.weight.data.uniform_(-r, r)
        self.fc1.bias.data.fill_(0)  		
        self.fc2.weight.data.uniform_(-r, r)
        self.fc2.bias.data.fill_(0)        
        
    def forward(self, image):
        # pdb.set_trace()
        features = self.fc(image)
        features = self.tanh(self.dropout(features))
        features = self.fc1(features)		
        features = self.tanh(self.dropout(features))
        features = self.fc2(features)     
        return features
        
class EncodeText(nn.Module):

	def __init__(self,text_dim, embed_size, use_abs=False, no_imgnorm=False):
		super(EncodeText, self).__init__()
		self.embed_size = embed_size
		self.use_abs = use_abs
		self.no_imgnorm = no_imgnorm
		self.fc = nn.Linear(text_dim, embed_size)
		self.fc1 = nn.Linear(embed_size, embed_size)
		self.fc2 = nn.Linear(embed_size, embed_size)
		self.tanh = nn.Tanh()     
		self.dropout = nn.Dropout(p=0.05)        
		self.init_weights()
		
	def init_weights(self):
		"""Xavier initialization for the fully connected layer
		"""
		r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
								  self.fc.out_features)
		self.fc.weight.data.uniform_(-r, r)
		self.fc.bias.data.fill_(0)         
		self.fc1.weight.data.uniform_(-r, r)
		self.fc1.bias.data.fill_(0)  		
		self.fc2.weight.data.uniform_(-r, r)
		self.fc2.bias.data.fill_(0)        
		
	def forward(self, captions):
		# pdb.set_trace()
		captions = captions.type(torch.float)
		features = self.fc(captions)
		features = self.tanh(self.dropout(features))
		features = self.fc1(features)		
		features = self.tanh(self.dropout(features))
		features = self.fc2(features)     
		return features
        
class CategoryClassifier(nn.Module):

	def __init__(self, embed_size, num_categories):
		super(CategoryClassifier, self).__init__()
		self.fc = nn.Linear(embed_size, 256)
		self.features = nn.Linear(256, num_categories)
		self.tanh = nn.Tanh()        
		self.init_weights()
		
	def init_weights(self):
		"""Xavier initialization for the fully connected layer
		"""
		r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
								  self.fc.out_features)
		self.fc.weight.data.uniform_(-r, r)
		self.fc.bias.data.fill_(0) 		
		self.features.weight.data.uniform_(-r, r)
		self.features.bias.data.fill_(0)        
		
	def forward(self, img_feat, cap_feat):
		img_features = self.tanh(self.fc(img_feat))
		cap_features = self.tanh(self.fc(cap_feat))
		img_pred = self.features(img_features)		  
		cap_pred = self.features(cap_features)		  
		return img_pred, cap_pred     
	

class TripletLoss(nn.Module):
    '''
    Compute triplet loss
    '''
    def __init__(self, margin, squared=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.squared = squared

    def _pairwise_distances(self, embeddings, squared=False):
        """Compute the 2D matrix of distances between all the embeddings.
        Args:
            embeddings: tensor of shape (batch_size, embed_dim)
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                     If false, output is the pairwise euclidean distance matrix.
        Returns:
            pairwise_distances: tensor of shape (batch_size, batch_size)
        """
        dot_product = torch.matmul(embeddings, embeddings.t())

        # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
        # This also provides more numerical stability (the diagonal of the result will be exactly 0).
        # shape (batch_size,)
        square_norm = torch.diag(dot_product)

        # Compute the pairwise distance matrix as we have:
        # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
        # shape (batch_size, batch_size)
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)

        # Because of computation errors, some distances might be negative so we put everything >= 0.0
        distances[distances < 0] = 0

        if not squared:
            # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
            # we need to add a small epsilon where distances == 0.0
            mask = distances.eq(0).float()
            distances = distances + mask * 1e-16

            distances = (1.0 -mask) * torch.sqrt(distances)

        return distances
        
    def _get_triplet_mask(self, labels):
        """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
        A triplet (i, j, k) is valid if:
            - i, j, k are distinct
            - labels[i] == labels[j] and labels[i] != labels[k]
        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]
        """
        # Check that i, j and k are distinct
        # pdb.set_trace()
        indices_equal = torch.eye(labels.size(0)).byte()
        indices_not_equal = ~indices_equal
        i_not_equal_j = indices_not_equal.unsqueeze(2)
        i_not_equal_k = indices_not_equal.unsqueeze(1)
        j_not_equal_k = indices_not_equal.unsqueeze(0)

        distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k


        label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        i_equal_j = label_equal.unsqueeze(2)
        i_equal_k = label_equal.unsqueeze(1)

        valid_labels = ~i_equal_k & i_equal_j

        return valid_labels.cuda() & distinct_indices.cuda()           

    def forward(self, labels, embeddings):	
        # Get the pairwise distance matrix
        pairwise_dist = self._pairwise_distances(embeddings, squared=self.squared)

        anchor_positive_dist = pairwise_dist.unsqueeze(2)
        anchor_negative_dist = pairwise_dist.unsqueeze(1)

        # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
        # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
        # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
        # and the 2nd (batch_size, 1, batch_size)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.margin



        # Put to zero the invalid triplets
        # (where label(a) != label(p) or label(n) == label(a) or a == p)
        mask = self._get_triplet_mask(labels)
        triplet_loss = mask.cuda().float() * triplet_loss

        # Remove negative losses (i.e. the easy triplets)
        triplet_loss[triplet_loss < 0] = 0

        # Count number of positive triplets (where triplet_loss > 0)
        valid_triplets = triplet_loss[triplet_loss > 1e-16]
        num_positive_triplets = valid_triplets.size(0)
        num_valid_triplets = mask.sum()

        fraction_positive_triplets = num_positive_triplets / (num_valid_triplets.float() + 1e-16)

        # Get final mean triplet loss over the positive valid triplets
        triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)
        
        return triplet_loss, fraction_positive_triplets     		

def attention(embed_size, emb_one, emb_two):
    if len(emb_two.shape) == 3:
        emb_two = emb_two[0,:,:]
        # pdb.set_trace()
    # pdb.set_trace()    
    w11 = Variable(torch.FloatTensor(embed_size, embed_size).uniform_(0.01, -0.01), requires_grad=True).cuda()
    w12 = Variable(torch.FloatTensor(embed_size, embed_size).uniform_(0.01, -0.01), requires_grad=True).cuda()
    w13 = Variable(torch.FloatTensor(embed_size, embed_size).uniform_(0.01, -0.01), requires_grad=True).cuda()
    w14 = Variable(torch.FloatTensor(embed_size, embed_size).uniform_(0.01, -0.01), requires_grad=True).cuda()
    # w15 = Variable(torch.FloatTensor(embed_size, embed_size).uniform_(0.01, -0.01), requires_grad=True).cuda()
    w_one = torch.mm(emb_one, w11)
    w_two = torch.mm(emb_two, w12)
    w_ot = torch.mm(emb_one * emb_two, w13)
    w_add = w_one + w_two + w_ot
    # pdb.set_trace()
    w_add = F.tanh(w_add)
    w_add = torch.mm(w_add, w14)
    w_one_norm = softmax(w_add) * emb_one + emb_one
    w_two_norm = softmax(w_add) * emb_two + emb_two
    w_one_norm = l2norm(w_one_norm)
    w_two_norm = l2norm(w_two_norm)
    return w_one_norm, w_two_norm
   
    
    
class CVS(object):

	def __init__(self, opt):
		self.grad_clip = opt.grad_clip    
		self.embed_size = opt.embed_size
		self.num_categories = opt.num_categories
		self.img_enc = EncodeImage(opt.img_dim, opt.embed_size,
									opt.finetune, opt.cnn_type)
		self.txt_enc = EncodeText(opt.word_dim,opt.embed_size, 
									opt.num_layers)
		self.cat_class = CategoryClassifier(opt.embed_size, self.num_categories)                                                       
									
		if torch.cuda.is_available():
			self.img_enc.cuda()
			self.txt_enc.cuda()
			self.cat_class.cuda()
			cudnn.benchmark = True 
			
		self.sim_criterion = TripletLoss(opt.margin)      
		self.cat_criterion = nn.CrossEntropyLoss() 
		# self.mse = criterion = nn.MSELoss()
		self.params = list(self.img_enc.fc.parameters()) + list(self.txt_enc.parameters())  
		self.optimizer = torch.optim.Adam(self.params, lr=opt.learning_rate)

	def to_one_hot(self, labels, num_categories):
		y = torch.eye(num_categories) 
		return y[labels.type(torch.LongTensor)] 
	 

	def category_loss(self, logits_image, logits_sent, labels, num_categories):
		labels_onehot = self.to_one_hot(labels, num_categories)
		# pdb.set_trace()	
		category_loss = self.cat_criterion(logits_image, labels.type(torch.LongTensor).cuda()) + \
						self.cat_criterion(logits_sent, labels.type(torch.LongTensor).cuda())
		# loss = category_loss.sum()	
		return category_loss	

	def invariance_loss(self, img_emb, cap_emb):
		'''
		loss for positive pairs
		'''
		num_samples = img_emb.shape[0]
		loss = torch.sum((torch.sqrt(torch.sum((img_emb - cap_emb)**2))))
		loss = loss/num_samples
		# pdb.set_trace()
		return loss
		
	def lr_scheduler(self):
		# Learning rate scheduler
		scheduler = StepLR(self.optimizer, step_size=250, gamma=0.5)
		scheduler.step()
		return scheduler.get_lr()[0]

	def state_dict(self):
		state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
		return state_dict

	def train_start(self):
		"""switch to train mode
		"""
		self.img_enc.train()
		self.txt_enc.train()

	def val_start(self):
		"""switch to evaluate mode
		"""
		self.img_enc.eval()
		self.txt_enc.eval()        
								   
	def forward_emb(self, images, captions, volatile=False):
		"""Compute the image and caption embeddings
		"""
		# Set mini-batch dataset
		images = Variable(images, volatile=volatile)
		captions = Variable(captions, volatile=volatile)		
		
		if torch.cuda.is_available():
			images = images.cuda()
			captions = captions.cuda()
		
		# Forward
		img_emb = self.img_enc(images)
		cap_emb = self.txt_enc(captions)
		img_emb, cap_emb = attention(512, img_emb, cap_emb)
		# pdb.set_trace()
		return img_emb, cap_emb

	def forward_loss(self, labels, img_emb, cap_emb, num_categories):
		"""Compute the loss given pairs of image and caption embeddings
		"""
		# classifier network and loss
		# pdb.set_trace()
		labels = labels.cuda()
		img_pred, cap_pred = self.cat_class(img_emb, cap_emb)
		cat_loss = self.category_loss(img_pred, cap_pred, labels, num_categories)
		
		mm_emb = torch.cat((img_emb,cap_emb), dim=0)
		mm_label = torch.cat((labels,labels), dim=0)
		
		img_loss,_ = self.sim_criterion(labels.cuda(), img_emb)
		cap_loss,_ = self.sim_criterion(labels, cap_emb)
		mm_loss,_ = self.sim_criterion(mm_label, mm_emb)
		metric_loss = 0.5*mm_loss + 0.1*img_loss + 0.5*cap_loss 

		invariance_loss = self.invariance_loss(img_emb, cap_emb)
		total_loss = 1.0*metric_loss + 1.0*cat_loss #+ 0.5*invariance_loss
		# total_loss = 1.0*invariance_loss + 1.0*cat_loss
		
		# Optimize model
		total_loss.backward()
		if self.grad_clip > 0:
			clip_grad_norm(self.params, self.grad_clip)
		self.optimizer.step()        
		return img_loss, cap_loss, mm_loss, metric_loss, total_loss

	def train_emb(self, images, captions, labels, ids=None, *args):
		"""One training step given images and captions.
		"""
		# compute the embeddings
		img_emb, cap_emb = self.forward_emb(images, captions)

		return img_emb, cap_emb