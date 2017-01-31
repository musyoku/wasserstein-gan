import chainer
from chainer import functions as F

class Link(object):
	pass

class Gaussian(Link):
	def __init__(self, layer_mean, layer_ln_var):
		self.layer_mean = layer_mean
		self.layer_ln_var = layer_ln_var

	def __call__(self, x):
		return self.layer_mean(x), self.layer_ln_var(x)

class MinibatchDiscrimination(Link):
	def __init__(self, T, num_kernels=50, ndim_kernel=5, train_weights=True):
		self.T = T
		self.num_kernels = num_kernels
		self.ndim_kernel = ndim_kernel
		self.train_weights = train_weights
		self.initial_T = None

	def __call__(self, x):
		xp = chainer.cuda.get_array_module(x.data)
		batchsize = x.shape[0]
		if self.train_weights == False and self.initial_T is not None:
			self.T.W.data = self.initial_T

		M = F.reshape(self.T(x), (-1, self.num_kernels, self.ndim_kernel))
		M = F.expand_dims(M, 3)
		M_T = F.transpose(M, (3, 1, 2, 0))
		M, M_T = F.broadcast(M, M_T)

		norm = F.sum(abs(M - M_T), axis=2)
		eraser = F.broadcast_to(xp.eye(batchsize, dtype=x.dtype).reshape((batchsize, 1, batchsize)), norm.shape)
		c_b = F.exp(-(norm + 1e6 * eraser))
		o_b = F.sum(c_b, axis=2)

		if self.train_weights == False:
			self.initial_T = self.T.W.data

		return F.concat((x, o_b), axis=1)

class Merge(object):
	def __init__(self):
		self.merge_layers = []

	def append_layer(self, layer):
		self.merge_layers.append(layer)

	def __call__(self, *args):
		output = 0
		if len(args) != len(self.merge_layers):
			raise Exception()
		for i, data in enumerate(args):
			output += self.merge_layers[i](data)
		return output