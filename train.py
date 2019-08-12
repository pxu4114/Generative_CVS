import pickle
import os
import time
import shutil
import argparse
import datetime
import numpy as np

import torch
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from tensorboard_logger import configure, log_value, Logger
import pdb

from data_loader import Data_loader
from eval import encode_data, compMapScore
from model import CVS 


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/shared/kgcoe-research/mil/video_project/cvs/birds/np/zs/',
                        help='path to datasets')
    parser.add_argument('--data_name', default='birds',
                        help='birds,flowers,pascal,wiki,nuswide')
    parser.add_argument('--logs_path', default='/shared/kgcoe-research/mil/generative_cvs',
                        help='Path to saved vocabulary pickle files.')
    parser.add_argument('--margin', default=1.0, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=500, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--word_dim', default=4800, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', default=512, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--grad_clip', default=0, type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--num', default=None, type=int,
                        help='Size of an image crop as the CNN input.')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--learning_rate', default=.0001, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_update', default=15, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=15, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=100, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=500, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--logger_name', default='/shared/kgcoe-research/mil/multi_modal_instance/runs/f8k_mm',
                        help='Path to save the model and Tensorboard log.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--num_categories', default=150,
                        help='Use max instead of sum in the rank loss.')
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')    
    parser.add_argument('--finetune', action='store_true',
                        help='Fine-tune the image encoder.')
    parser.add_argument('--cnn_type', default='resnet152',
                        help="""The CNN used for image encoder
                        (e.g. vgg19, resnet152)""")
    parser.add_argument('--use_restval', action='store_true',
                        help='Use the restval data for training on MSCOCO.')
    parser.add_argument('--measure', default='cosine',
                        help='Similarity measure used (cosine|order)')
    parser.add_argument('--use_abs', action='store_true',
                        help='Take the absolute value of embedding vectors.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--reset_train', action='store_false',
                        help='Ensure the training is always done in '
                        'train mode (Not recommended).')
    parser.add_argument('--phase', default='train',
                        help='Phase of model')
                        

    opt = parser.parse_args()
    print(opt)

    # Dataloader
    train_dataset = Data_loader(opt.data_path, opt.data_name, opt.num, phase='train')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=opt.batch_size, 
                                                   shuffle=True,
                                                   num_workers=20) 

    val_dataset = Data_loader(opt.data_path, opt.data_name, opt.num, phase='test')	
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                   batch_size=128, 
                                                   shuffle=False,
                                                   num_workers=20) 												   

    # Build model
    model = CVS(opt)

    # Creating checkpoint saver and directory
    now = datetime.datetime.now()
    checkpoint_dir = opt.logs_path + '/checkpoint/' + now.strftime("%Y-%m-%d_%H_%M")
    Log = Logger(opt.logs_path + '/summary/' + now.strftime("%Y-%m-%d_%H_%M"), flush_secs=5)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)		

    # Write the parameters of the experiment in checkpoint_dir
    param_file = open(os.path.join(checkpoint_dir, 'exp_params.txt'), 'w')
    for key, value in vars(opt).items():
        param_file.write(str(key)+' : '+ str(value)+'\n')
    param_file.close()    

    # Train the model
    total_step = len(train_loader)
    print('number of samples to train: {}'.format(train_dataset.__len__()))
    count = 0
    for epoch in range(opt.num_epochs):
        lr = model.lr_scheduler
        # scheduler.step()
        # print(lr)
        model.train_start()
        for i, train_data in enumerate(train_loader): 
            count +=1
            # Update the model       
            img_emb, cap_emb = model.train_emb(*train_data)
            
            # compute loss
            img_loss, cap_loss, mm_loss, metric_loss, total_loss = model.forward_loss(train_data[2], 
                                                img_emb, cap_emb, opt.num_categories)

            if (i+1) % 20 == 0:				
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch+1, opt.num_epochs, i+1, total_step, total_loss.item()))
                           
                # Log scalar values (scalar summary)
                Log.log_value('loss', total_loss.item(), count)

        # evaluate the performance of model on test set
        i2s, s2i = eval(val_loader, model, opt.batch_size)
        avg_score = (i2s+s2i)/2
        Log.log_value('image2text', i2s, count)
        Log.log_value('text2image', s2i, count)
        Log.log_value('average score', avg_score, count)
        

        if (epoch+1)%10==0:	
            # Save the model checkpoints 
            torch.save(model.state_dict(), os.path.join(
                checkpoint_dir, 'model-{}.ckpt'.format(epoch+1)))
            print('checkpoint saved')	
		

def eval(val_loader, model, batch_size):
	sim_mat, all_labels = encode_data(val_loader, model, batch_size)
	
	# compute scores
	i2s_mapk = compMapScore(sim_mat, all_labels)
	s2i_mapk = compMapScore(sim_mat.T, all_labels)	

	print('image to text: {}'.format(i2s_mapk))
	print('text to image: {}'.format(s2i_mapk))
	return i2s_mapk, s2i_mapk 
	
    
if __name__ == '__main__':
    main()                        