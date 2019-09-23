import tensorflow as tf

from models.stackgan.stageI.model import ConditionalGan
from utils.utils import save_images, get_balanced_factorization, initialize_uninitialized, save_captions, save_scores
from utils.saver import save, load
from preprocess.dataset import TextDataset

from models.retrieval.datautils import DatasetLoader
from models.retrieval.main import Retrieval
from models.retrieval.moreutils import tsneVis, compClusterScores, pairwise_distances
from models.retrieval.moreutils import compMapScore
from models.retrieval.moreutils import compute_nmi, recall_at_k
from models.retrieval.moreutils import ModelParameters

import numpy as np
import time
import pdb


class ConditionalGanTrainer(object):
    def __init__(self, sess: tf.Session, model: ConditionalGan, dataset: TextDataset, cfg):
        self.sess = sess
        self.model = model
        self.dataset = dataset
        self.R_loader = DatasetLoader(cfg.RETRIEVAL.DATA_PATH, mode='train')
        self.test_data_loader = DatasetLoader(cfg.RETRIEVAL.DATA_PATH, mode='test')
        self.Retrieval = Retrieval(cfg)
        self.cfg = cfg
        self.lr = self.cfg.TRAIN.D_LR

    def define_losses(self):
        self.img_emb, self.txt_emb, self.R_loss = self.Retrieval.build_model()
        self.model.build_model(self.txt_emb)
        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=None, name='lr')
        self.D_synthetic_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.model.D_synthetic_logits,
                                                    labels=tf.zeros_like(self.model.D_synthetic)))
        self.D_real_match_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.model.D_real_match_logits,
                                                    labels=tf.fill(self.model.D_real_match.get_shape(), 0.9)))
        self.D_real_mismatch_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.model.D_real_mismatch_logits,
                                                    labels=tf.zeros_like(self.model.D_real_mismatch)))

        self.G_kl_loss = self.kl_loss(self.model.embed_mean, self.model.embed_log_sigma)
        self.G_gan_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.model.D_synthetic_logits,
                                                    labels=tf.ones_like(self.model.D_synthetic)))

        # Define the final losses
        alpha_coeff = self.cfg.TRAIN.COEFF.ALPHA_MISMATCH_LOSS
        kl_coeff = self.cfg.TRAIN.COEFF.KL
        self.D_loss = self.D_real_match_loss + alpha_coeff * self.D_real_mismatch_loss \
            + (1.0 - alpha_coeff) * self.D_synthetic_loss
        self.G_loss = self.G_gan_loss + kl_coeff * self.G_kl_loss
        self.R_loss = self.R_loss + 0.5*self.G_loss + 0.5*self.D_loss
        # pdb.set_trace()


        self.G_loss_summ = tf.summary.scalar("g_loss", self.G_loss)
        self.D_loss_summ = tf.summary.scalar("d_loss", self.D_loss)
        self.R_loss_summ = tf.summary.scalar("R_loss", self.R_loss)
        

        self.saver = tf.train.Saver(max_to_keep=self.cfg.TRAIN.CHECKPOINTS_TO_KEEP)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.D_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.cfg.TRAIN.D_BETA_DECAY) \
                .minimize(self.D_loss, var_list=self.model.d_vars)
            self.G_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.cfg.TRAIN.G_BETA_DECAY) \
                .minimize(self.G_loss, var_list=self.model.g_vars)
            self.R_optim = tf.train.AdamOptimizer(learning_rate=self.cfg.RETRIEVAL.R_LR)\
                .minimize(self.R_loss, var_list=self.Retrieval.var_list)    

    def kl_loss(self, mean, log_sigma):
        loss = -log_sigma + .5 * (-1 + tf.exp(2. * log_sigma) + tf.square(mean))
        loss = tf.reduce_mean(loss)
        return loss

    def define_summaries(self):
        self.D_synthetic_summ = tf.summary.histogram('d_synthetic_sum', self.model.D_synthetic)
        self.D_real_match_summ = tf.summary.histogram('d_real_match_sum', self.model.D_real_match)
        self.D_real_mismatch_summ = tf.summary.histogram('d_real_mismatch_sum', self.model.D_real_mismatch)
        self.G_img_summ = tf.summary.image("g_sum", self.model.G)
        self.z_sum = tf.summary.histogram("z", self.model.z)

        self.D_synthetic_loss_summ = tf.summary.scalar('d_synthetic_sum_loss', self.D_synthetic_loss)
        self.D_real_match_loss_summ = tf.summary.scalar('d_real_match_sum_loss', self.D_real_match_loss)
        self.D_real_mismatch_loss_summ = tf.summary.scalar('d_real_mismatch_sum_loss', self.D_real_mismatch_loss)
        self.D_loss_summ = tf.summary.scalar("d_loss", self.D_loss)

        self.G_gan_loss_summ = tf.summary.scalar("g_gan_loss", self.G_gan_loss)
        self.G_kl_loss_summ = tf.summary.scalar("g_kl_loss", self.G_kl_loss)
        self.G_loss_summ = tf.summary.scalar("g_loss", self.G_loss)

        self.G_merged_summ = tf.summary.merge([self.G_img_summ,
                                               self.G_loss_summ,
                                               self.G_gan_loss_summ,
                                               self.G_kl_loss_summ])

        self.D_merged_summ = tf.summary.merge([self.D_real_mismatch_summ,
                                               self.D_real_match_summ,
                                               self.D_synthetic_summ,
                                               self.D_synthetic_loss_summ,
                                               self.D_real_mismatch_loss_summ,
                                               self.D_real_match_loss_summ,
                                               self.D_loss_summ])

        self.writer = tf.summary.FileWriter(self.cfg.LOGS_DIR, self.sess.graph)

    def train(self):
        self.define_losses()
        self.define_summaries() 

        sample_z = np.random.normal(0, 1, (self.model.sample_num, self.model.z_dim))
        _, sample_embed, _, captions = self.dataset.test.next_batch_test(self.model.sample_num, 0, 1)
        im_feats_test, sent_feats_test, labels_test = self.test_data_loader.get_batch(0,self.cfg.RETRIEVAL.SAMPLE_NUM,\
                                                        image_aug = self.cfg.RETRIEVAL.IMAGE_AUG, phase='test') 		
        sample_embed = np.squeeze(sample_embed, axis=0)
        print(sample_embed.shape)

        save_captions(self.cfg.SAMPLE_DIR, captions)

        counter = 1
        start_time = time.time()

        could_load, checkpoint_counter = load(self.saver, self.sess, self.cfg.CHECKPOINT_DIR)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        initialize_uninitialized(self.sess)
        
        # Updates per epoch are given by the training data size / batch size
        updates_per_epoch = self.dataset.train.num_examples // self.model.batch_size
        epoch_start = counter // updates_per_epoch

        for epoch in range(epoch_start, self.cfg.TRAIN.EPOCH):
            cen_epoch = epoch // 100

            for idx in range(0, updates_per_epoch):
                images, wrong_images, embed, _, _ = self.dataset.train.next_batch(self.model.batch_size, 1,
                                                                                  embeddings=True, wrong_img=True)#4
                                                                                  
                # test_data_loader = DatasetLoader()																  
                batch_z = np.random.normal(0, 1, (self.model.batch_size, self.model.z_dim))
                
                # Retrieval data loader
                if idx % updates_per_epoch == 0:
                    self.R_loader.shuffle_inds()
                
                im_feats, sent_feats, labels = self.R_loader.get_batch(idx % updates_per_epoch,\
                                self.cfg.RETRIEVAL.BATCH_SIZE, image_aug = self.cfg.RETRIEVAL.IMAGE_AUG)                

                feed_dict = {
                    self.learning_rate: self.lr * (0.5**cen_epoch),
                    self.model.inputs: images,
                    self.model.wrong_inputs: wrong_images,
                    # self.model.embed_inputs: embed,
                    # self.model.embed_inputs: self.txt_emb,
                    self.model.z: batch_z,
                    self.Retrieval.image_placeholder : im_feats, 
                    self.Retrieval.sent_placeholder : sent_feats,
                    self.Retrieval.label_placeholder : labels
                }
                

                
                # R_feed_dict = {self.Retrieval.image_placeholder : im_feats, self.Retrieval.sent_placeholder : sent_feats,\
                        # self.Retrieval.label_placeholder : labels}                

                # Update D network
                _, err_d, summary_str = self.sess.run([self.D_optim, self.D_loss, self.D_merged_summ],
                                                      feed_dict=feed_dict)
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, err_g, summary_str = self.sess.run([self.G_optim, self.G_loss, self.G_merged_summ],
                                                      feed_dict=feed_dict)
                self.writer.add_summary(summary_str, counter)

                # Update R network
                _, err_r, summary_str = self.sess.run([self.R_optim, self.R_loss, self.R_loss_summ],
                                                      feed_dict=feed_dict)
                self.writer.add_summary(summary_str, counter) 
               
                
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, r_loss: %.8f"
                      % (epoch, idx, updates_per_epoch,
                         time.time() - start_time, err_d, err_g, err_r))

                if np.mod(counter, 1000) == 0:
                    try:
                        self.Retrieval.eval()
                        sent_emb = self.sess.run(self.Retrieval.sent_embed_tensor,
                                                feed_dict={
                                                            self.Retrieval.image_placeholder_test: im_feats_test,
                                                            self.Retrieval.sent_placeholder_test: sent_feats_test,
                                                          })
                        self.model.eval(sent_emb)								  
                        samples = self.sess.run(self.model.sampler,
                                                feed_dict={
                                                            self.model.z_sample: sample_z,
                                                            # self.model.embed_sample: sample_embed,
                                                            self.model.embed_sample: sent_emb,
                                                          })
                        # Retrieval validation
                        # self.Retrieval.eval()
                        # ret = self.sess.run(self.Retrieval.image_embed_tensor, self.Retrieval.sent_embed_tensor,
                                            # feed_dict={self.Retrieval.image_placeholder_test=
                                                       # self.Retrieval.sent_placeholder_test=})
                        
                        save_images(samples, get_balanced_factorization(samples.shape[0]),
                                    '{}train_{:02d}_{:04d}.png'.format(self.cfg.SAMPLE_DIR, epoch, idx))
                    except Exception as e:
                        print("Failed to generate sample image")
                        print(type(e))
                        print(e.args)
                        print(e)

                if np.mod(counter, 500) == 0:
                    save(self.saver, self.sess, self.cfg.CHECKPOINT_DIR, counter)
                
            if np.mod(epoch, 50) == 0 and epoch!=0:
                self.ret_eval(epoch)
        
        
    def ret_eval(self, epoch):
        test_num_samples = self.test_data_loader.no_samples      
        all_labels = []
        sim_mat = np.zeros((test_num_samples,test_num_samples))
        self.Retrieval.eval(sample_size=test_num_samples)
        
        for i in range(test_num_samples):
        
            # im_feats, sent_feats, labels = test_data_loader.get_batch(i, params.batchSize, phase = 'test')
            im_feats, sent_feats, labels = self.test_data_loader.get_batch(i, 1, phase = 'eval')

            im_feats_rep = np.repeat(im_feats,test_num_samples,0)
            labels_rep = np.repeat(labels,test_num_samples,0)
            
            image_embed, sent_embed = self.sess.run((self.Retrieval.image_embed_tensor, self.Retrieval.sent_embed_tensor),
                                    feed_dict={
                                                self.Retrieval.image_placeholder_test: im_feats_rep,
                                                self.Retrieval.sent_placeholder_test: sent_feats,
                                              })     
            sim_mat[i,:] = pairwise_distances(image_embed[0,:].reshape(1,-1),sent_embed)#,'cosine')
            all_labels.extend(labels)
            if i % 100 == 0:
                print('Done: '+str(i)+' of '+str(test_num_samples))
        
        i2s_mapk = compMapScore(sim_mat, all_labels)
        s2i_mapk = compMapScore(sim_mat.T, all_labels)

        print('Image to Sentence mAP@50: ',i2s_mapk,'\n',
             'Sentence to Image mAP@50: ',s2i_mapk,'\n',)
        save_scores(self.cfg.SCORE_DIR, i2s_mapk, s2i_mapk, epoch)	 
             
   