def paths(params, dataset):

    if dataset == 'flowers_zs':
        params.numCategories = 82
        params.image_feat_path_train = '/shared/kgcoe-research/mil/video_project/cvs/flowers/np/image_train.npy'
        params.sent_feat_path_train = '/shared/kgcoe-research/mil/video_project/cvs/flowers/np/text_train.npy'
        params.label_path_train = '/shared/kgcoe-research/mil/video_project/cvs/flowers/np/label_train.npy'
        params.image_feat_path_test = '/shared/kgcoe-research/mil/video_project/cvs/flowers/np/image_test.npy'
        params.sent_feat_path_test = '/shared/kgcoe-research/mil/video_project/cvs/flowers/np/text_test.npy'
        params.label_path_test = '/shared/kgcoe-research/mil/video_project/cvs/flowers/np/label_test_0.npy'
        
    elif dataset == 'flowers_simple': # 
        params.numCategories = 82
        params.image_feat_path_train = '/shared/kgcoe-research/mil/video_project/cvs/flowers/np/onlyTrain/image_train.npy'
        params.sent_feat_path_train = '/shared/kgcoe-research/mil/video_project/cvs/flowers/np/onlyTrain/text_train.npy'
        params.label_path_train = '/shared/kgcoe-research/mil/video_project/cvs/flowers/np/onlyTrain/label_train.npy'
        params.image_feat_path_test = '/shared/kgcoe-research/mil/video_project/cvs/flowers/np/onlyTrain/image_test.npy'
        params.sent_feat_path_test = '/shared/kgcoe-research/mil/video_project/cvs/flowers/np/onlyTrain/text_test.npy'
        params.label_path_test = '/shared/kgcoe-research/mil/video_project/cvs/flowers/np/onlyTrain/label_test.npy'
        
    elif dataset == 'nuswide':
        params.numCategories = 10
        params.image_feat_path_train = '/shared/kgcoe-research/mil/video_project/cvs/nuswide/np/image_train.npy'
        params.sent_feat_path_train = '/shared/kgcoe-research/mil/video_project/cvs/nuswide/np/text_train.npy'
        params.label_path_train = '/shared/kgcoe-research/mil/video_project/cvs/nuswide/np/label_train.npy'
        params.image_feat_path_test = '/shared/kgcoe-research/mil/video_project/cvs/nuswide/np/image_test.npy'
        params.sent_feat_path_test = '/shared/kgcoe-research/mil/video_project/cvs/nuswide/np/text_test.npy'
        params.label_path_test = '/shared/kgcoe-research/mil/video_project/cvs/nuswide/np/label_test.npy'
    
    elif dataset == 'pascal':
        params.numCategories = 20 
        params.image_feat_path_train = '/shared/kgcoe-research/mil/video_project/cvs/pascal/np/image_train.npy'
        params.sent_feat_path_train = '/shared/kgcoe-research/mil/video_project/cvs/pascal/np/text_train.npy'
        params.label_path_train = '/shared/kgcoe-research/mil/video_project/cvs/pascal/np/label_train.npy'
        params.image_feat_path_test = '/shared/kgcoe-research/mil/video_project/cvs/pascal/np/image_test.npy'
        params.sent_feat_path_test = '/shared/kgcoe-research/mil/video_project/cvs/pascal/np/text_test.npy'
        params.label_path_test = '/shared/kgcoe-research/mil/video_project/cvs/pascal/np/label_test.npy'        

    elif dataset == 'pascal_loc':
        params.numCategories = 20 
        params.image_feat_path_test = '/shared/kgcoe-research/mil/video_project/cvs/pascal/np_localize/image_test.npy'
        params.sent_feat_path_test = '/shared/kgcoe-research/mil/video_project/cvs/pascal/np_localize/text_test.npy'
        params.label_path_test = '/shared/kgcoe-research/mil/video_project/cvs/pascal/np_localize/label_test.npy'    
    
    elif dataset == 'birds_zs':
        params.numCategories = 150 
        params.image_feat_path_train = '/shared/kgcoe-research/mil/video_project/cvs/birds/np/zs/image_train.npy'
        params.sent_feat_path_train = '/shared/kgcoe-research/mil/video_project/cvs/birds/np/zs/text_train.npy'
        params.label_path_train = '/shared/kgcoe-research/mil/video_project/cvs/birds/np/zs/label_train.npy'
        params.image_feat_path_test = '/shared/kgcoe-research/mil/video_project/cvs/birds/np/zs/image_test.npy'
        params.sent_feat_path_test = '/shared/kgcoe-research/mil/video_project/cvs/birds/np/zs/text_test.npy'
        params.label_path_test = '/shared/kgcoe-research/mil/video_project/cvs/birds/np/zs/label_test.npy' 
        
    elif dataset == 'wiki_orig':
        params.numCategories = 10
        params.image_feat_path_train = '/shared/kgcoe-research/mil/video_project/cvs/wiki/orig/I_tr.npy'
        params.sent_feat_path_train = '/shared/kgcoe-research/mil/video_project/cvs/wiki/orig/T_tr.npy'
        params.label_path_train = '/shared/kgcoe-research/mil/video_project/cvs/wiki/train_labels.npy'
        params.image_feat_path_test = '/shared/kgcoe-research/mil/video_project/cvs/wiki/orig/I_te.npy'
        params.sent_feat_path_test = '/shared/kgcoe-research/mil/video_project/cvs/wiki/orig/T_te.npy'
        params.label_path_test = '/shared/kgcoe-research/mil/video_project/cvs/wiki/test_labels.npy' 

    elif dataset == 'wiki':
        params.numCategories = 10
        params.image_feat_path_train = '/shared/kgcoe-research/mil/video_project/cvs/wiki/wiki_image_train.npy'
        params.sent_feat_path_train = '/shared/kgcoe-research/mil/video_project/cvs/wiki/train_text_feats.npy'
        params.label_path_train = '/shared/kgcoe-research/mil/video_project/cvs/wiki/train_labels.npy'
        params.image_feat_path_test = '/shared/kgcoe-research/mil/video_project/cvs/wiki/wiki_image_test.npy'
        params.sent_feat_path_test = '/shared/kgcoe-research/mil/video_project/cvs/wiki/test_text_feats.npy'
        params.label_path_test = '/shared/kgcoe-research/mil/video_project/cvs/wiki/test_labels.npy' 

    elif dataset == 'xmedianet':
        params.numCategories = 200
        params.image_feat_path_train = '/shared/kgcoe-research/mil/video_project/cvs/xmedianet/train_image_feat.npy'
        params.sent_feat_path_train = '/shared/kgcoe-research/mil/video_project/cvs/xmedianet/train_text_feat.npy'
        params.label_path_train = '/shared/kgcoe-research/mil/video_project/cvs/xmedianet/train_labels.npy'
        params.image_feat_path_test = '/shared/kgcoe-research/mil/video_project/cvs/xmedianet/test_image_feat.npy'
        params.sent_feat_path_test = '/shared/kgcoe-research/mil/video_project/cvs/xmedianet/test_text_feat.npy'
        params.label_path_test = '/shared/kgcoe-research/mil/video_project/cvs/xmedianet/test_labels.npy' 

    else:
        print('Unknown Dataset Identifier!!!')
        
    return params
