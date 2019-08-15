from __future__ import print_function
import os
import pickle
import time
import numpy as np
import torch

from sklearn.metrics.pairwise import pairwise_distances

import pdb

def encode_data(val_dataset, val_loader, model, batch_size):
	"""Encode all images and captions loadable by `data_loader`
	"""

	# switch to evaluate mode
	model.val_start()
	# pdb.set_trace()
	test_num_samples = val_dataset.__len__()
	# all_labels = []
	sim_mat = np.zeros((test_num_samples,test_num_samples))	
	all_img_emb = np.zeros((test_num_samples, 512))	
	all_cap_emb = np.zeros((test_num_samples, 512))	
	all_labels = np.zeros((test_num_samples))
	for i, val_data in enumerate(val_loader):
		if i%500==0 and i!=0: print('done:{}'.format(i))
		# images = np.repeat(val_data[0],test_num_samples,0)
		images = val_data[0]
		labels = val_data[2]
		captions = val_data[1]	
		img_emb, cap_emb = model.forward_emb(images, captions)
		# all_labels.append(labels)
		# pdb.set_trace()
		all_labels[i*batch_size:(i+1)*batch_size] = labels.cpu().detach().numpy()
		all_img_emb[i*batch_size:(i+1)*batch_size] = img_emb.cpu().detach().numpy()
		all_cap_emb[i*batch_size:(i+1)*batch_size] = cap_emb.cpu().detach().numpy()

	for j in range(test_num_samples):
		# pdb.set_trace()
		sim_mat[j,:] = pairwise_distances(all_img_emb[j,:].reshape(1, -1) , all_cap_emb)#,'cosine')
		# sim_mat[i,:] = pairwise_distances(img_emb[i,:].cpu().detach().numpy().reshape(1,-1),cap_emb.cpu().detach().numpy())#,'cosine')
		# pdb.set_trace()

	return sim_mat, all_labels

	
        
def compMapScore(distance_matrix, gt_test, k = 50, same = False):
    # pdb.set_trace()
    
    num_samples = distance_matrix.shape[0]
    # distance_matrix = pairwise_distances(emb1,emb2)
    # num_samples = len(emb1)

    # id_matrix = np.identity(distance_matrix.shape[0]) # Assuming emb1 and emb2 shapes are equal

    # Set the diagonal elements to a very high value when same modality
    if same:
        for row in range(num_samples):
            for col in range(num_samples):
                if row==col:
                    distance_matrix[row, col] = 1e10
    #same_modality_check = np.multiply(distance_matrix,id_matrix)
    # if (np.all(same_modality_check == 0)):
        # np.fill_diadonal(distance_matrix, 1e10)
        
    sorted_index = np.argsort(distance_matrix)
    # gt_test = np.asarray(gt_test)
    predicted_k_labels = gt_test[sorted_index]
    predicted_k_labels = predicted_k_labels.tolist()
    
    mapk_score = mapk([[i] for i in gt_test.tolist()], predicted_k_labels, k)
    return mapk_score	
	
def apk(actual, predicted, k=50):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        # if p in actual:# and p not in predicted[:i]:
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual or num_hits==0:
        return 0.0

    # return score / num_hits#min(len(actual), k)
    return score / min(len(actual), k)	
	
def mapk(actual, predicted, k=50):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])	
    
    
    