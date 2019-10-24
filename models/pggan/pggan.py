import tensorflow as tf
import time

from models.retrieval.datautils import DatasetLoader
from models.retrieval.main import Retrieval
from models.retrieval.moreutils import tsneVis, compClusterScores, pairwise_distances
from models.retrieval.moreutils import compMapScore
from models.retrieval.moreutils import compute_nmi, recall_at_k
from models.retrieval.moreutils import ModelParameters

from utils.ops import lrelu_act, conv2d, fc, upscale, pool, layer_norm
from utils.utils import save_images, get_balanced_factorization, show_all_variables, save_captions, print_vars, \
    initialize_uninitialized, save_scores
from utils.saver import load, save
import numpy as np
import sys


class PGGAN(object):

	# build model
	def __init__(self, cfg, batch_size, steps, check_dir_write, check_dir_read, dataset, sample_path, log_dir, stage, trans,
				 build_model=True):
		
		self.cfg = cfg
		self.R_loader = DatasetLoader(cfg.RETRIEVAL.DATA_PATH, mode='train')
		self.test_data_loader = DatasetLoader(cfg.RETRIEVAL.DATA_PATH, mode='test')
		self.Retrieval = Retrieval(cfg) 
		
		self.batch_size = batch_size
		self.steps = steps
		self.check_dir_write = check_dir_write
		self.check_dir_read = check_dir_read
		self.dataset = dataset
		self.sample_path = sample_path
		self.log_dir = log_dir
		self.stage = stage
		self.trans = trans

		self.z_dim = 128
		# self.embed_dim = 1024
		self.embed_dim = 512
		self.out_size = 4 * pow(2, stage - 1)
		self.channel = 3
		self.sample_num = 64
		self.compr_embed_dim = 128
		self.lr = 0.00005
		self.lr_inp = self.lr
		self.output_size = 4 * pow(2, stage - 1)

		self.dt = tf.Variable(0.0, trainable=False)
		self.alpha_tra = tf.Variable(initial_value=0.0, trainable=False, name='alpha_tra')

		if build_model:
			# self.build_model()
			self.define_losses()
			self.define_summaries()

	def build_model(self, emb):
		# Define the input tensor by appending the batch size dimension to the image dimension
		self.iter = tf.placeholder(tf.int32, shape=None)
		self.learning_rate = tf.placeholder(tf.float32, shape=None)
		self.x = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, self.channel], name='x')
		self.x_mismatch = tf.placeholder(tf.float32,
										 [self.batch_size, self.output_size, self.output_size, self.channel],
										 name='x_mismatch')
		# self.cond = tf.placeholder(tf.float32, [self.batch_size, self.embed_dim], name='cond')
		self.cond = emb
		self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')
		self.epsilon = tf.placeholder(tf.float32, [self.batch_size, 1, 1, 1], name='eps')

		self.z_sample = tf.placeholder(tf.float32, [self.sample_num] + [self.z_dim], name='z_sample')
		self.cond_sample = tf.placeholder(tf.float32, [self.sample_num] + [self.embed_dim], name='cond_sample')

		self.G, self.mean, self.log_sigma = self.generator(self.z, self.cond, stages=self.stage, t=self.trans)

		self.Dg_logit = self.discriminator(self.G, self.cond, reuse=False, stages=self.stage, t=self.trans)
		self.Dx_logit = self.discriminator(self.x, self.cond, reuse=True, stages=self.stage, t=self.trans)
		self.Dxmi_logit = self.discriminator(self.x_mismatch, self.cond, reuse=True, stages=self.stage, t=self.trans)
		
		self.epsilon = tf.random_uniform([self.batch_size, 1, 1, 1], 0., 1.)
		self.x_hat = self.epsilon * self.G + (1. - self.epsilon) * self.x
		self.cond_inp = self.cond + 0.0
		self.Dx_hat_logit = self.discriminator(self.x_hat, self.cond_inp, reuse=True, stages=self.stage, t=self.trans)

		# self.sampler, _, _ = self.generator(self.z_sample, self.cond_sample, reuse=True, stages=self.stage,
											# t=self.trans)

		self.alpha_assign = tf.assign(self.alpha_tra,
									  (tf.cast(tf.cast(self.iter, tf.float32) / self.steps, tf.float32)))

		self.d_vars = tf.trainable_variables('d_net')
		self.g_vars = tf.trainable_variables('g_net')

		show_all_variables()
		
	def eval(self, emb):
		self.sampler, _, _ = self.generator(self.z_sample, emb, reuse=True, stages=self.stage,
											t=self.trans)	

	def get_gradient_penalty(self, x, y):
		grad_y = tf.gradients(y, [x])[0]
		slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_y), reduction_indices=[1, 2, 3]))
		return tf.reduce_mean(tf.maximum(0.0, slopes - 1.) ** 2)

	def get_gradient_penalty2(self, x, y):
		grad_y = tf.gradients(y, [x])[0]
		slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_y), reduction_indices=[1]))
		return tf.reduce_mean(tf.maximum(0.0, slopes - 1.) ** 2)

	def define_losses(self):
		self.img_emb, self.txt_emb, self.R_loss = self.Retrieval.build_model()
		self.build_model(self.txt_emb)
		self.D_loss_real = tf.reduce_mean(self.Dx_logit)
		self.D_loss_fake = tf.reduce_mean(self.Dg_logit)
		self.D_loss_mismatch = tf.reduce_mean(self.Dxmi_logit)
		self.wdist = self.D_loss_real - self.D_loss_fake
		self.wdist2 = self.D_loss_real - self.D_loss_mismatch
		self.reg_loss = tf.reduce_mean(tf.square(self.Dxmi_logit))

		self.G_kl_loss = self.kl_std_normal_loss(self.mean, self.log_sigma)
		self.real_gp = self.get_gradient_penalty(self.x_hat, self.Dx_hat_logit)
		self.real_gp2 = self.get_gradient_penalty2(self.cond_inp, self.Dx_hat_logit)

		self.D_loss = -self.wdist - self.wdist2 + 200.0 * (self.real_gp + self.real_gp2)
		# self.D_loss = -self.wdist - self.wdist2 + 50000000.0 * (self.real_gp + self.real_gp2)
		self.G_loss = -self.D_loss_fake + 5.0 * self.G_kl_loss
		# self.G_loss = -self.D_loss_fake + 5.0 * self.G_kl_loss
		self.R_loss = self.R_loss + 0.5*self.G_loss + 0.5*self.D_loss  

		self.D_optimizer = tf.train.AdamOptimizer(0.000002, beta1=0.0, beta2=0.99)
		self.G_optimizer = tf.train.AdamOptimizer(0.000002, beta1=0.0, beta2=0.99)
		self.R_optimizer = tf.train.AdamOptimizer(0.000002, beta1=0.0, beta2=0.99)

		with tf.control_dependencies([self.alpha_assign]):
			self.D_optim = self.D_optimizer.minimize(self.D_loss, var_list=self.d_vars)
		self.G_optim = self.G_optimizer.minimize(self.G_loss, var_list=self.g_vars)
		self.R_optim = self.R_optimizer.minimize(self.R_loss, var_list=self.Retrieval.var_list)

		# variables to save
		vars_to_save = self.get_variables_up_to_stage(self.stage)
		print('Length of the vars to save: %d' % len(vars_to_save))
		print('\n\nVariables to save:')
		print_vars(vars_to_save)
		self.saver = tf.train.Saver(vars_to_save, max_to_keep=2)

		# variables to restore
		self.restore = None
		if self.stage > 1 and self.trans:
			vars_to_restore = self.get_variables_up_to_stage(self.stage - 1)
			print('Length of the vars to restore: %d' % len(vars_to_restore))
			print('\n\nVariables to restore:')
			print_vars(vars_to_restore)
			self.restore = tf.train.Saver(vars_to_restore)

	def define_summaries(self):
		summaries = [
			tf.summary.image('x', self.x),
			tf.summary.image('G_img', self.G),

			tf.summary.histogram('z', self.z),
			tf.summary.histogram('z_sample', self.z_sample),

			tf.summary.scalar('G_loss_wass', -self.D_loss_fake),
			tf.summary.scalar('kl_loss', self.G_kl_loss),
			tf.summary.scalar('G_loss', self.G_loss),
			# tf.summary.scalar('d_lr', self.d_lr),
			# tf.summary.scalar('g_lr', self.g_lr),

			tf.summary.scalar('D_loss_real', self.D_loss_real),
			tf.summary.scalar('D_loss_fake', self.D_loss_fake),
			tf.summary.scalar('real_gp', self.real_gp),
			tf.summary.scalar('D_loss', self.D_loss),
			tf.summary.scalar('reg_loss', self.reg_loss),
			tf.summary.scalar('wdist', self.wdist),
			tf.summary.scalar('wdist2', self.wdist2),
			tf.summary.scalar('d_loss_mismatch', self.D_loss_mismatch),
			tf.summary.scalar('real_gp2', self.real_gp2),
		]
		self.summary_op = tf.summary.merge(summaries)

	# do train
	def train(self):
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True

		with tf.Session(config=config) as self.sess:

			summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
			start_point = 0

			if self.stage != 1:
				if self.trans:
					could_load, _ = load(self.restore, self.sess, self.check_dir_read)
					if not could_load:
						raise RuntimeError('Could not load previous stage during transition')
				else:
					could_load, _ = load(self.saver, self.sess, self.check_dir_read)
					if not could_load:
						raise RuntimeError('Could not load current stage')

			# variables to init
			vars_to_init = initialize_uninitialized(self.sess)
			self.sess.run(tf.variables_initializer(vars_to_init))

			sample_z = np.random.normal(0, 1, (self.sample_num, self.z_dim))
			_, sample_cond, _, captions = self.dataset.test.next_batch_test(self.sample_num, 0, 1)
			sample_cond = np.squeeze(sample_cond, axis=0)
			print('Conditionals sampler shape: {}'.format(sample_cond.shape))

			save_captions(self.sample_path, captions)
			start_time = time.time()

			for idx in range(start_point + 1, self.steps):
				if self.trans:
					# Reduce the learning rate during the transition period and slowly increase it
					p = idx / self.steps
					self.lr_inp = self.lr  # * np.exp(-2 * np.square(1 - p))

				epoch_size = self.dataset.train.num_examples // self.batch_size
				epoch = idx // epoch_size

				images, wrong_images, embed, _, _ = self.dataset.train.next_batch(self.batch_size, 1,
																				  wrong_img=True,
																				  embeddings=True)
																				  
				im_feats_test, sent_feats_test, labels_test = self.test_data_loader.get_batch(0,self.cfg.RETRIEVAL.SAMPLE_NUM,\
																image_aug = self.cfg.RETRIEVAL.IMAGE_AUG, phase='test')
														
				batch_z = np.random.normal(0, 1, (self.batch_size, self.z_dim))
				eps = np.random.uniform(0., 1., size=(self.batch_size, 1, 1, 1))
				import pdb
				# pdb.set_trace()
				# Retrieval data loader
				# epoch_size = 120
				# if idx % epoch_size == 0:
					# self.R_loader.shuffle_inds()
				
				im_feats, sent_feats, labels = self.R_loader.get_batch(idx % epoch_size,\
								self.cfg.RETRIEVAL.BATCH_SIZE, image_aug = self.cfg.RETRIEVAL.IMAGE_AUG) 				

				feed_dict = {
					self.x: images,
					self.learning_rate: self.lr_inp,
					self.x_mismatch: wrong_images,
					# self.cond: embed,
					self.z: batch_z,
					self.epsilon: eps,
					self.z_sample: sample_z,
					# self.cond_sample: sample_cond,
					self.iter: idx,
					self.Retrieval.image_placeholder : im_feats, 
					self.Retrieval.sent_placeholder : sent_feats,
					self.Retrieval.label_placeholder : labels					
				}

				_, err_d = self.sess.run([self.D_optim, self.D_loss], feed_dict=feed_dict)
				_, err_g = self.sess.run([self.G_optim, self.G_loss], feed_dict=feed_dict)
				_, err_r = self.sess.run([self.R_optim, self.R_loss], feed_dict=feed_dict)

				if np.mod(idx, 20) == 0:
					summary_str = self.sess.run(self.summary_op, feed_dict=feed_dict)
					summary_writer.add_summary(summary_str, idx)

					print("Epoch: [%2d] [%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, r_loss: %.8f"
						  % (epoch, idx, time.time() - start_time, err_d, err_g, err_r))

				if np.mod(idx, 2000) == 0:
					try:
						self.Retrieval.eval()
						sent_emb = self.sess.run(self.Retrieval.sent_embed_tensor,
												feed_dict={
															self.Retrieval.image_placeholder_test: im_feats_test,
															self.Retrieval.sent_placeholder_test: sent_feats_test,
														  })
						self.eval(sent_emb)	
						samples = self.sess.run(self.sampler,
												feed_dict={
															self.z_sample: sample_z,
															# self.model.embed_sample: sample_embed,
															self.cond_sample: sent_emb,
														  })
						
						# samples = sess.run(self.sampler, feed_dict={
													# self.z_sample: sample_z,
													# self.cond_sample: sample_cond})
						samples = np.clip(samples, -1., 1.)
						if self.out_size > 256:
							samples = samples[:4]

						save_images(samples, get_balanced_factorization(samples.shape[0]),
									'{}train_{:02d}_{:04d}.png'.format(self.sample_path, epoch, idx))

					except Exception as e:
						print("Failed to generate sample image")
						print(type(e))
						print(e.args)
						print(e)

				if np.mod(idx, 2000) == 0 or idx == self.steps - 1:
					save(self.saver, self.sess, self.check_dir_write, idx)
				sys.stdout.flush()

				if np.mod(idx, 20000) == 0:
					print('yes')
					self.ret_eval(idx)				

		tf.reset_default_graph()

	def discriminator(self, inp, cond, stages, t, reuse=False):
		alpha_trans = self.alpha_tra
		with tf.variable_scope("d_net", reuse=reuse):
			x_iden = None
			if t:
				x_iden = pool(inp, 2)
				x_iden = self.from_rgb(x_iden, stages - 2)

			x = self.from_rgb(inp, stages - 1)

			for i in range(stages - 1, 0, -1):
				with tf.variable_scope(self.get_conv_scope_name(i), reuse=reuse):
					x = conv2d(x, f=self.get_dnf(i), ks=(3, 3), s=(1, 1), act=lrelu_act())
					x = conv2d(x, f=self.get_dnf(i-1), ks=(3, 3), s=(1, 1), act=lrelu_act())
					x = pool(x, 2)
				if i == stages - 1 and t:
					x = tf.multiply(alpha_trans, x) + tf.multiply(tf.subtract(1., alpha_trans), x_iden)

			with tf.variable_scope(self.get_conv_scope_name(0), reuse=reuse):
				# Real/False branch
				cond_compress = fc(cond, units=128, act=lrelu_act())
				concat = self.concat_cond4(x, cond_compress)
				x_b1 = conv2d(concat, f=self.get_dnf(0), ks=(3, 3), s=(1, 1), act=lrelu_act())
				x_b1 = conv2d(x_b1, f=self.get_dnf(0), ks=(4, 4), s=(1, 1), padding='VALID', act=lrelu_act())
				output_b1 = fc(x_b1, units=1)

			return output_b1

	def generator(self, z_var, cond_inp, stages, t, reuse=False, cond_noise=True):
		alpha_trans = self.alpha_tra
		with tf.variable_scope('g_net', reuse=reuse):

			with tf.variable_scope(self.get_conv_scope_name(0), reuse=reuse):
				mean_lr, log_sigma_lr = self.generate_conditionals(cond_inp)
				cond = self.sample_normal_conditional(mean_lr, log_sigma_lr, cond_noise)
				# import pdb
				# pdb.set_trace()
				x = tf.concat([z_var, cond], axis=1)
				x = fc(x, units=4*4*self.get_nf(0))
				x = layer_norm(x)
				x = tf.reshape(x, [-1, 4, 4, self.get_nf(0)])

				x = conv2d(x, f=self.get_nf(0), ks=(3, 3), s=(1, 1))
				x = layer_norm(x, act=tf.nn.relu)
				x = conv2d(x, f=self.get_nf(0), ks=(3, 3), s=(1, 1))
				x = layer_norm(x, act=tf.nn.relu)

			x_iden = None
			for i in range(1, stages):

				if (i == stages - 1) and t:
					x_iden = self.to_rgb(x, stages - 2)
					x_iden = upscale(x_iden, 2)

				with tf.variable_scope(self.get_conv_scope_name(i), reuse=reuse):
					x = upscale(x, 2)
					x = conv2d(x, f=self.get_nf(i), ks=(3, 3), s=(1, 1))
					x = layer_norm(x, act=tf.nn.relu)
					x = conv2d(x, f=self.get_nf(i), ks=(3, 3), s=(1, 1))
					x = layer_norm(x, act=tf.nn.relu)

			x = self.to_rgb(x, stages - 1)

			if t:
				x = tf.multiply(tf.subtract(1., alpha_trans), x_iden) + tf.multiply(alpha_trans, x)

			return x, mean_lr, log_sigma_lr

	def concat_cond4(self, x, cond):
		cond_compress = tf.expand_dims(tf.expand_dims(cond, 1), 1)
		cond_compress = tf.tile(cond_compress, [1, 4, 4, 1])
		x = tf.concat([x, cond_compress], axis=3)
		return x

	def concat_cond128(self, x, cond_inp, cond_noise=True):
		mean, log_sigma = self.generate_conditionals(cond_inp, units=256)
		cond = self.sample_normal_conditional(mean, log_sigma, cond_noise)

		cond_compress = tf.reshape(cond, [-1, 16, 16, 1])
		cond_compress = tf.tile(cond_compress, [1, 8, 8, 8])
		x = tf.concat([x, cond_compress], axis=3)
		return x, mean, log_sigma

	def get_rgb_name(self, stage):
		return 'rgb_stage_%d' % stage

	def get_conv_scope_name(self, stage):
		return 'conv_stage_%d' % stage

	def get_dnf(self, stage):
		return min(1024 // (2 ** stage) * 2, 512)

	def get_nf(self, stage):
		return min(1024 // (2 ** stage) * 4, 512)

	def from_rgb(self, x, stage):
		with tf.variable_scope(self.get_rgb_name(stage)):
			return conv2d(x, f=self.get_dnf(stage), ks=(1, 1), s=(1, 1), act=lrelu_act())

	def generate_conditionals(self, embeddings, units=128):
		"""Takes the embeddings, compresses them and builds the statistics for a multivariate normal distribution"""
		mean = fc(embeddings, units, act=lrelu_act())
		log_sigma = fc(embeddings, units, act=lrelu_act())
		return mean, log_sigma

	def sample_normal_conditional(self, mean, log_sigma, cond_noise=True):
		if cond_noise:
			epsilon = tf.truncated_normal(tf.shape(mean))
			stddev = tf.exp(log_sigma)
			return mean + stddev * epsilon
		return mean

	def kl_std_normal_loss(self, mean, log_sigma):
		loss = -log_sigma + .5 * (-1 + tf.exp(2. * log_sigma) + tf.square(mean))
		loss = tf.reduce_mean(loss)
		return loss

	def to_rgb(self, x, stage):
		with tf.variable_scope(self.get_rgb_name(stage)):
			x = conv2d(x, f=9, ks=(2, 2), s=(1, 1), act=tf.nn.relu)
			x = conv2d(x, f=3, ks=(1, 1), s=(1, 1))
			return x

	def get_adam_vars(self, opt, vars_to_train):
		opt_vars = [opt.get_slot(var, name) for name in opt.get_slot_names()
					for var in vars_to_train
					if opt.get_slot(var, name) is not None]
		opt_vars.extend(list(opt._get_beta_accumulators()))
		return opt_vars

	def get_variables_up_to_stage(self, stages):
		d_vars_to_save = tf.global_variables('d_net/%s' % self.get_rgb_name(stages - 1))
		g_vars_to_save = tf.global_variables('g_net/%s' % self.get_rgb_name(stages - 1))
		for stage in range(stages):
			d_vars_to_save += tf.global_variables('d_net/%s' % self.get_conv_scope_name(stage))
			g_vars_to_save += tf.global_variables('g_net/%s' % self.get_conv_scope_name(stage))
		return d_vars_to_save + g_vars_to_save


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





