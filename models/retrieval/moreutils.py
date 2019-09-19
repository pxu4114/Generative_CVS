from __future__ import division
from __future__ import print_function

import matplotlib
matplotlib.use('Agg')

import os
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import recall_score
import pdb
from matplotlib import pyplot as plt
# from tsne import bh_sne
import random
from sklearn.metrics import calinski_harabaz_score
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from models.retrieval.mapk import mapk, apk


class ModelParameters():
    def __init__(self):
        self.image_feat_path_train = '/shared/kgcoe-research/mil/video_project/cvs/flowers/np/image_train.npy'
        self.sent_feat_path_train = '/shared/kgcoe-research/mil/video_project/cvs/flowers/np/text_train.npy'
        self.label_path_train = '/shared/kgcoe-research/mil/video_project/cvs/flowers/np/label_train.npy'
        self.image_feat_path_test = '/shared/kgcoe-research/mil/video_project/cvs/flowers/np/image_test.npy'
        self.sent_feat_path_test = '/shared/kgcoe-research/mil/video_project/cvs/flowers/np/text_test.npy'
        self.label_path_test = '/shared/kgcoe-research/mil/video_project/cvs/flowers/np/label_test.npy'

        self.ExperimentDirectory = './experiment_test'
        self.batchSize = 64
        self.maxEpoch = 100
        self.initLR = 0.001
        self.Restore = False
        self.printEvery = 10
        self.embedDim = 512
        self.categoryScale = 1.0
        self.metricScale = 1.0
        self.modalityScale = 1.0
        self.margin = 1.0
        self.numCategories = 82
        self.saveEvery = 1
        
    def createDir(self):
        if not os.path.exists(self.ExperimentDirectory):
            os.makedirs(self.ExperimentDirectory)
    
    def saveParams(self):
        data = vars(self)
        with open(os.path.join(self.ExperimentDirectory,'parameters.txt'),'w') as f:
            for key in sorted(data.keys()):
                f.write(key+' : '+str(data[key])+'\n')
    
    
def tsneVis(X1, X2, labels, save_dir, max_points = 10000):
    # perform t-SNE embedding
    modal = np.concatenate((np.ones(np.array(labels).shape), np.zeros(np.array(labels).shape)))
    labels = np.concatenate((np.array(labels), np.array(labels)))
    X = np.concatenate((np.array(X1), np.array(X2)))
    X = X.astype('float64')
    vis_data = bh_sne(X)

    if vis_data.shape[0] > max_points:
        ind = random.sample(range(1, vis_data.shape[0]), max_points)
        vis_data = vis_data[ind,:]
        labels = labels[ind]
        modal = modal[ind]
    # plot the result
    vis_x = vis_data[:, 0]
    vis_y = vis_data[:, 1]

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_figheight(50)
    fig.set_figwidth(50)
    
    ax.plot(vis_x, vis_y, marker='', linestyle='')
    # Text labels:
    for i in range(labels.shape[0]):
        if modal[i] == 0:
            cr = 'red'
        else:
            cr = 'blue'
        ax.annotate(str(labels[i]), (vis_x[i], vis_y[i]), fontsize=14, color=cr)

    # for i in range(labels.shape[0]):

        # plt.scatter(vis_x[i], vis_y[i], marker=r"$ {} $".format(str(labels[i,0])), s = 200, c = cr)
        # ax.annotate(labels[i], (vis_x[i], vis_y[i]))
    # pdb.set_trace()
    # plt.scatter(vis_x, vis_y, c=labels, cmap=plt.cm.get_cmap("jet", len(np.unique(labels))))
    # plt.colorbar(ticks=range(len(np.unique(labels))))
    # plt.clim(-0.5, 9.5)
    plt.savefig(os.path.join(save_dir,'tSNE.png'), bbox_inches='tight')
    
def compClusterScores(X1, X2, labels):
    # only image
    # img_ch_score = calinski_harabaz_score(np.array(X1), np.array(labels))
    # img_s_score = silhouette_score(np.array(X1), np.array(labels))
    # only sentence
    # sent_ch_score = calinski_harabaz_score(np.array(X2), np.array(labels))
    # sent_s_score = silhouette_score(np.array(X2), np.array(labels))    
    # cvs
    Xlabels = np.concatenate((np.array(labels), np.array(labels)))
    X = np.concatenate((np.array(X1), np.array(X2)))
    cvs_ch_score = calinski_harabaz_score(np.array(X), np.array(Xlabels))
    cvs_s_score = silhouette_score(np.array(X), np.array(Xlabels))
    
    return cvs_ch_score, cvs_s_score

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
    gt_test = np.asarray(gt_test)
    predicted_k_labels = gt_test[sorted_index]
    predicted_k_labels = predicted_k_labels.tolist()
    
    mapk_score = mapk([[i] for i in gt_test.tolist()], predicted_k_labels, k)
    return mapk_score

def compute_nmi(X1, X2, labels):
    n_cluster = len(set(labels))
    Xlabels = np.concatenate((np.array(labels), np.array(labels)))
    X = np.concatenate((np.array(X1), np.array(X2)))
    kmeans= KMeans(n_clusters=n_cluster, n_jobs=-1, random_state=1, max_iter=1000).fit(X)
    kmeans_nmi = normalized_mutual_info_score(Xlabels, kmeans.labels_)  # K-means NMI
    return kmeans_nmi

def recall_at_k(X1, X2, labels, recall_scales=[1,2,4,8,10], same = False):
    """
    Computes Recall at k
    Args: 
        X1 : [num_samples, embedding_dim] 
        X2 : [num_samples, embedding_dim] 
        labels : [num_samples] -- Groundtruth class
        same: True if X1, X2 are same modality
        recall_scales : list of recall factors [1, 2, 4, 8, 10]
    """
    X = np.concatenate((np.array(X1), np.array(X2)))
    num_samples = len(X1)
    pdist_matrix = pairwise_distances(X1, X2)

    # Set the diagonal elements to a very high value when same modality
    if same:
        for row in range(num_samples):
            for col in range(num_samples):
                if row==col:
                    pdist_matrix[row, col] = 1e10
    # For each sample, sort the distances to the neighbouring samples
    # Get the sorted topK indices( distances ascending order sorted)
    # Increment if the groundtruth class id is in list of topK indices. 
    recall_k = []
    for k in recall_scales:
        num_correct=0
        for i in range(num_samples):
            this_class_index = labels[i]
            sorted_indices = np.argsort(pdist_matrix[i, :])
            knn_indices = sorted_indices[:k]
            knn_class_indices = [labels[x] for x in list(knn_indices)]
            if this_class_index in knn_class_indices:
                num_correct+=1
        recall = float(num_correct)/num_samples
        recall_k.append(recall)
    return recall_k

# def matlab_mapk(emb1,emb2,labels1,labels2,same = False,k=50):
def matlab_mapk(distance_matrix,gt_test,k=50):
    
    labels1 = gt_test
    labels2 = gt_test
	#Calculate distances
    #pdb.set_trace()
    # distance_matrix = pairwise_distances(emb1,emb2)
    
    num_samples = distance_matrix.shape[0]
	# Set the diagonal elements to a very high value when same modality
    sorted_distance_matrix = np.argsort(distance_matrix)
    
    AP = np.zeros((num_samples,1))
    for i in range(num_samples):

        resultList = sorted_distance_matrix[i,:k]	
        relCount = 0.0
        for j in range(k):
            if labels2[resultList[j]] == labels1[i]:
                relCount = relCount + 1
                AP[i] = AP[i] + relCount/(j+1)
        AP[i] = AP[i]/(relCount+1)
    mapk = np.mean(AP)
    
    return mapk

def compMapScore_ACMR(emb1, emb2, labels1, labels2):
    
    avg_precs = []
    all_precs = []
    test_labels = labels1
    all_k = [50]
    for k in all_k: 
        for i in range(len(emb1)):
            query_label = test_labels[i]

            # distances and sort by distances
            wv = emb1[i]
            diffs = emb2 - wv
            dists = np.linalg.norm(diffs, axis=1)
            sorted_idx = np.argsort(dists)

            #for each k do top-k
            precs = []
            for topk in range(1, k + 1):
                hits = 0
                top_k = sorted_idx[0 : topk]                    
                if query_label != test_labels[top_k[-1]]:
                    continue
                for ii in top_k:
                    retrieved_label = test_labels[ii]
                    if retrieved_label == query_label:
                        hits += 1
                precs.append(float(hits) / float(topk))
            if len(precs) == 0:
                precs.append(0)
            avg_precs.append(np.average(precs))
        mean_avg_prec = np.mean(avg_precs)
        all_precs.append(mean_avg_prec)
    
    return all_precs[0]
    
def top_acc(distance_matrix, gt_test):
    # gt_test = np.asarray(gt_test)
    # emb1 = np.asarray(emb1)
    # emb2 = np.asarray(emb2)
    # avg_emb2 = []
    # for i in range(0,len(np.unique(gt_test))):
        # # pdb.set_trace()
        # avg_emb2.append(np.mean(emb2[np.where(gt_test==i)],axis=0))
        # # avg_emb2.append(np.mean([emb2[x] for x in np.where(np.asarray(gt_test)==i)[0]],axis=0))
    # # pdb.set_trace()
    # distance_matrix = pairwise_distances(emb1,np.asarray(avg_emb2))
    # pdb.set_trace()
    sorted_index = np.argsort(distance_matrix)
    gt_test = np.asarray(gt_test)
    # predicted_k_labels = np.unique(gt_test)[sorted_index]
    predicted_k_labels = gt_test[sorted_index]
    top1 = np.mean(predicted_k_labels[:,0]==gt_test)
    # hits=0
    # for i in range(gt_test.shape[0]):
        # if gt_test[i] in predicted_k_labels[i,:5]:
            # hits = hits+1.0 
    # top5 = hits / gt_test.shape[0]
    return top1#, top5

def mapk_zs(emb1, emb2, gt_test, k = 50):
    gt_test = np.asarray(gt_test)
    emb1 = np.asarray(emb1)
    emb2 = np.asarray(emb2)    
    avg_emb2 = []
    for i in range(0,len(np.unique(gt_test))):
        avg_emb2.append(np.mean(emb2[np.where(gt_test==i)],axis=0))
        # avg_emb2.append(np.mean([emb2[x] for x in np.where(np.asarray(gt_test)==i)[0]],axis=0))
    # pdb.set_trace()
    distance_matrix = pairwise_distances(emb1,np.asarray(avg_emb2))
    sorted_index = np.argsort(distance_matrix)
    gt_test = np.asarray(gt_test)
    predicted_k_labels = np.unique(gt_test)[sorted_index]#gt_test[sorted_index]
    predicted_k_labels = predicted_k_labels.tolist()
    
    actual = [[i] for i in gt_test.tolist()] #actual 
    #actual=range(1,len(np.unique(gt_test))+1)
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted_k_labels)])
