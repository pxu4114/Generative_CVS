from random import randint

from models.stackgan.stageI.model import ConditionalGan
from utils.saver import load
from utils.utils import denormalize_images
from preprocess.dataset import TextDataset
import tensorflow as tf
import numpy as np
from evaluation import fid, inception_score
from models.inception.model import load_inception_inference
import os

from models.retrieval.datautils import DatasetLoader
from models.retrieval.main import Retrieval
from models.retrieval.moreutils import tsneVis, compClusterScores, pairwise_distances
from models.retrieval.moreutils import compMapScore
# from moreutils import matlab_mapk as compMapScore
from models.retrieval.moreutils import compute_nmi, recall_at_k
from models.retrieval.moreutils import ModelParameters


class StageIEval(object):
    def __init__(self, sess: tf.Session, model: ConditionalGan, dataset: TextDataset, cfg):
        self.sess = sess
        self.model = model
        self.Retrieval = Retrieval(cfg)
        self.dataset = dataset
        self.cfg = cfg
        self.test_data_loader = DatasetLoader(cfg.RETRIEVAL.DATA_PATH, mode='test')		
        self.classes = self.cfg.EVAL.NUM_CLASSES
        self.bs = self.cfg.EVAL.SAMPLE_SIZE

    def evaluate_fid(self):
        incep_batch_size = self.cfg.EVAL.INCEP_BATCH_SIZE
        _, layers = load_inception_inference(self.sess, self.classes, incep_batch_size,
                                             self.cfg.EVAL.INCEP_CHECKPOINT_DIR)
        pool3 = layers['PreLogits']
        act_op = tf.reshape(pool3, shape=[incep_batch_size, -1])

        if not os.path.exists(self.cfg.EVAL.ACT_STAT_PATH):
            print('Computing activation statistics for real x')
            fid.compute_and_save_activation_statistics(self.cfg.EVAL.R_IMG_PATH, self.sess, incep_batch_size, act_op,
                                                       self.cfg.EVAL.ACT_STAT_PATH, verbose=True)

        print('Loading activation statistics for the real x')
        stats = np.load(self.cfg.EVAL.ACT_STAT_PATH)
        mu_real = stats['mu']
        sigma_real = stats['sigma']

        z = tf.placeholder(tf.float32, [self.bs, self.model.z_dim], name='real_images')
        cond = tf.placeholder(tf.float32, [self.bs] + [self.model.embed_dim], name='cond')
        eval_gen, _, _ = self.model.generator(z, cond, reuse=False)

        saver = tf.train.Saver(tf.global_variables('g_net'))
        could_load, _ = load(saver, self.sess, self.cfg.CHECKPOINT_DIR)
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            raise RuntimeError('Could not load the checkpoints of the generator')

        print('Generating x...')

        fid_size = self.cfg.EVAL.SIZE
        n_batches = fid_size // self.bs

        w, h, c = self.model.image_dims[0], self.model.image_dims[1], self.model.image_dims[2]
        samples = np.zeros((n_batches * self.bs, w, h, c))
        for i in range(n_batches):
            start = i * self.bs
            end = start + self.bs
            
            im_feats, sent_feats, labels = self.test_data_loader.get_batch(i, self.cfg.RETRIEVAL.BATCH_SIZE\
                                            , phase = 'test')
            im_feats_rep = np.repeat(im_feats,test_num_samples,0)
            labels_rep = np.repeat(labels,test_num_samples,0)

            #########
            image_embed, sent_embed = self.sess.run((self.Retrieval.image_embed_tensor, self.Retrieval.sent_embed_tensor),
                                    feed_dict={
                                                self.Retrieval.image_placeholder_test: im_feats_rep,
                                                self.Retrieval.sent_placeholder_test: sent_feats,
                                              })  			

            ######
            
            sample_z = np.random.normal(0, 1, size=(self.bs, self.model.z_dim))
            images, _, embed, _, _ = self.dataset.test.next_batch(self.bs, 4, embeddings=True)

            samples[start: end] = denormalize_images(self.sess.run(eval_gen, feed_dict={z: sample_z, cond: embed}))

        print('Computing activation statistics for generated x...')
        mu_gen, sigma_gen = fid.calculate_activation_statistics(samples, self.sess, incep_batch_size, act_op,
                                                                verbose=True)
        print("calculate FID:", end=" ", flush=True)
        try:
            FID = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
        except Exception as e:
            print(e)
            FID = 500

        print(FID)

    def evaluate_inception(self):
        incep_batch_size = self.cfg.EVAL.INCEP_BATCH_SIZE
        logits, _ = load_inception_inference(self.sess, self.classes, incep_batch_size,
                                             self.cfg.EVAL.INCEP_CHECKPOINT_DIR)
        pred_op = tf.nn.softmax(logits)

        z = tf.placeholder(tf.float32, [self.bs, self.model.z_dim], name='z')
        cond = tf.placeholder(tf.float32, [self.bs] + [self.model.embed_dim], name='cond')
        eval_gen, _, _ = self.model.generator(z, cond, reuse=False, is_training=False)
        self.Retrieval.eval(self.bs)
        import pdb
       
        saver = tf.train.Saver(tf.global_variables('g_net')) 
        # saver = tf.train.Saver(tf.global_variables())
        could_load, _ = load(saver, self.sess, self.cfg.CHECKPOINT_DIR)
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            raise RuntimeError('Could not load the checkpoints of the generator')

        print('Generating x...')

        size = self.cfg.EVAL.SIZE
        n_batches = size // self.bs

        w, h, c = self.model.image_dims[0], self.model.image_dims[1], self.model.image_dims[2]
        samples = np.zeros((n_batches * self.bs, w, h, c))
        for i in range(n_batches):
            print("\rGenerating batch %d/%d" % (i + 1, n_batches), end="", flush=True)

            sample_z = np.random.normal(0, 1, size=(self.bs, self.model.z_dim))
            # _, _, embed, _, _ = self.dataset.test.next_batch(self.bs, 4, embeddings=True)
            pdb.set_trace()
            _, _, embed, _, _ = self.dataset.test.next_batch(self.bs, 1, embeddings=True)
            start = i * self.bs
            end = start + self.bs
            
            # self.Retrieval.eval()
            sent_emb = self.sess.run(self.Retrieval.sent_embed_tensor,
                                    feed_dict={
                                                self.Retrieval.image_placeholder_test: im_feats_test,
                                                self.Retrieval.sent_placeholder_test: sent_feats_test,
                                              })			

            gen_batch = self.sess.run(eval_gen, feed_dict={z: sample_z, cond: sent_emb})
            samples[start: end] = denormalize_images(gen_batch)

        print('\nComputing inception score...')
        mean, std = inception_score.get_inception_score(samples, self.sess, incep_batch_size, 10, pred_op, verbose=True)
        print('Inception Score | mean:', "%.2f" % mean, 'std:', "%.2f" % std)





