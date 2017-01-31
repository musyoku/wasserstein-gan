# -*- coding: utf-8 -*-
import math
import numpy as np
import chainer, os, collections, six, math, random, time, copy
from chainer import cuda, Variable, optimizers, serializers, function, optimizer, initializers
from chainer.utils import type_check
from chainer import functions as F
from chainer import links as L
from params import Params
import sequential

class Object(object):
	pass

def to_object(dict):
	obj = Object()
	for key, value in dict.iteritems():
		setattr(obj, key, value)
	return obj

class Sequential(sequential.Sequential):

	def __call__(self, x, test=False):
		activations = []
		for i, link in enumerate(self.links):
			if isinstance(link, sequential.functions.dropout):
				x = link(x, train=not test)
			elif isinstance(link, chainer.links.BatchNormalization):
				x = link(x, test=test)
			else:
				x = link(x)
				if isinstance(link, sequential.functions.ActivationFunction):
					activations.append(x)
		return x, activations

class DiscriminatorParams(Params):
	def __init__(self):
		self.ndim_input = 28 * 28
		self.ndim_output = 1
		self.weight_init_std = 1
		self.weight_initializer = "Normal"		# Normal, GlorotNormal or HeNormal
		self.nonlinearity = "elu"
		self.optimizer = "Adam"
		self.learning_rate = 0.001
		self.momentum = 0.5
		self.gradient_clipping = 10
		self.weight_decay = 0
		self.use_feature_matching = False
		self.use_minibatch_discrimination = False

class GeneratorParams(Params):
	def __init__(self):
		self.ndim_input = 10
		self.ndim_output = 28 * 28
		self.distribution_output = "universal"	# universal, sigmoid or tanh
		self.weight_init_std = 1
		self.weight_initializer = "Normal"		# Normal, GlorotNormal or HeNormal
		self.nonlinearity = "relu"
		self.optimizer = "Adam"
		self.learning_rate = 0.001
		self.momentum = 0.5
		self.gradient_clipping = 10
		self.weight_decay = 0

class GAN():
	def __init__(self, params_discriminator, params_generator):
		self.params_discriminator = copy.deepcopy(params_discriminator)
		self.config_discriminator = to_object(params_discriminator["config"])

		self.params_generator = copy.deepcopy(params_generator)
		self.config_generator = to_object(params_generator["config"])

		self.build_discriminator()
		self.build_generator()
		self._gpu = False

	def build_discriminator(self):
		self.discriminator = sequential.chain.Chain()
		self.discriminator.add_sequence(sequential.from_dict(self.params_discriminator["model"]))
		config = self.config_discriminator
		self.discriminator.setup_optimizers(config.optimizer, config.learning_rate, config.momentum)

	def build_generator(self):
		self.generator = sequential.chain.Chain()
		self.generator.add_sequence(sequential.from_dict(self.params_generator["model"]))
		config = self.config_discriminator
		self.generator.setup_optimizers(config.optimizer, config.learning_rate, config.momentum)

	def cache_discriminator_weights(self):
		self.cached_weights = {}
		xp = self.xp
		optimizer = self.discriminator.optimizer
		for name, param in optimizer.target.namedparams():
			with cuda.get_device(param.data):
				self.cached_weights[name] = xp.copy(param.data)

	def restore_discriminator_weights(self):
		optimizer = self.discriminator.optimizer
		for name, param in optimizer.target.namedparams():
			with cuda.get_device(param.data):
				if name not in self.cached_weights:
					raise Exception()
				param.data = self.cached_weights[name]

	def update_learning_rate(self, lr):
		self.discriminator.update_learning_rate(lr)
		self.generator.update_learning_rate(lr)

	def to_gpu(self):
		self.discriminator.to_gpu()
		self.generator.to_gpu()
		self._gpu = True

	@property
	def gpu_enabled(self):
		if cuda.available is False:
			return False
		return self._gpu

	@property
	def xp(self):
		if self.gpu_enabled:
			return cuda.cupy
		return np

	def to_variable(self, x):
		if isinstance(x, Variable) == False:
			x = Variable(x)
			if self.gpu_enabled:
				x.to_gpu()
		return x

	def to_numpy(self, x):
		if isinstance(x, Variable) == True:
			x = x.data
		if isinstance(x, cuda.ndarray) == True:
			x = cuda.to_cpu(x)
		return x

	def get_batchsize(self, x):
		return x.shape[0]

	def zero_grads(self):
		self.optimizer_discriminator.zero_grads()
		self.optimizer_generative_model.zero_grads()

	def sample_z(self, batchsize=1, gaussian=False):
		config = self.config_generator
		ndim_z = config.ndim_input
		if gaussian:
			# gaussian
			z_batch = np.random.normal(0, 1, (batchsize, ndim_z)).astype(np.float32)
		else:
			# uniform
			z_batch = np.random.uniform(-1, 1, (batchsize, ndim_z)).astype(np.float32)
		return z_batch

	def generate_x(self, batchsize=1, test=False, as_numpy=False, from_gaussian=False):
		return self.generate_x_from_z(self.sample_z(batchsize, gaussian=from_gaussian), test=test, as_numpy=as_numpy)

	def generate_x_from_z(self, z_batch, test=False, as_numpy=False):
		z_batch = self.to_variable(z_batch)
		x_batch, _ = self.generator(z_batch, test=test, return_activations=True)
		if as_numpy:
			return self.to_numpy(x_batch)
		return x_batch

	def discriminate(self, x_batch, test=False, apply_softmax=True):
		x_batch = self.to_variable(x_batch)
		prob, activations = self.discriminator(x_batch, test=test, return_activations=True)
		if apply_softmax:
			prob = F.softmax(prob)
		return prob, activations

	def backprop_discriminator(self, loss):
		self.discriminator.backprop(loss)

	def backprop_generator(self, loss):
		self.generator.backprop(loss)


	def compute_kld(self, p, q):
		return F.reshape(F.sum(p * (F.log(p + 1e-16) - F.log(q + 1e-16)), axis=1), (-1, 1))

	def get_unit_vector(self, v):
		v /= (np.sqrt(np.sum(v ** 2, axis=1)).reshape((-1, 1)) + 1e-16)
		return v

	def compute_lds(self, x, xi=10, eps=1, Ip=1):
		x = self.to_variable(x)
		y1, _ = self.discriminate(x, apply_softmax=True)
		y1.unchain_backward()
		d = self.to_variable(self.get_unit_vector(np.random.normal(size=x.shape).astype(np.float32)))
		
		for i in xrange(Ip):
			y2, _ = self.discriminate(x + xi * d, apply_softmax=True)
			kld = F.sum(self.compute_kld(y1, y2))
			kld.backward()
			d = self.to_variable(self.get_unit_vector(self.to_numpy(d.grad)))
		
		y2, _ = self.discriminate(x + eps * d, apply_softmax=True)
		return -self.compute_kld(y1, y2)

	def load(self, dir=None):
		if dir is None:
			raise Exception()
		self.generator.load(dir + "/generator.hdf5")
		self.discriminator.load(dir + "/discriminator.hdf5")

	def save(self, dir=None):
		if dir is None:
			raise Exception()
		try:
			os.mkdir(dir)
		except:
			pass
		self.generator.save(dir + "/generator.hdf5")
		self.discriminator.save(dir + "/discriminator.hdf5")