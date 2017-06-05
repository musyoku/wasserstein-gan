import math
import numpy as np
from chainer import cuda, Variable
from chainer import initializers, link, function, variable
from chainer.utils import array, type_check
from chainer.functions.connection import linear

def _as_mat(x):
	if x.ndim == 2:
		return x
	return x.reshape(len(x), -1)

def _norm(W):
	xp = cuda.get_array_module(W)
	norm = xp.sqrt(xp.sum(W ** 2, axis=1)) + 1e-9
	norm = norm.reshape((-1, 1))
	return norm

class LinearFunction(linear.LinearFunction):

	def check_type_forward(self, in_types):
		n_in = in_types.size()
		type_check.expect(3 <= n_in, n_in <= 4)
		x_type, w_type, g_type = in_types[:3]

		type_check.expect(
			x_type.dtype.kind == "f",
			w_type.dtype.kind == "f",
			g_type.dtype.kind == "f",
			x_type.ndim >= 2,
			w_type.ndim == 2,
			g_type.ndim == 2,
			type_check.prod(x_type.shape[1:]) == w_type.shape[1],
		)

		if type_check.eval(n_in) == 4:
			b_type = in_types[3]
			type_check.expect(
				b_type.dtype == x_type.dtype,
				b_type.ndim == 1,
				b_type.shape[0] == w_type.shape[0],
			)

	def forward(self, inputs):
		x = _as_mat(inputs[0])
		V = inputs[1]
		g = inputs[2]
		xp = cuda.get_array_module(V)

		self.norm = _norm(V)
		self.V_normalized = V / self.norm
		self.W = g * self.V_normalized

		y = x.dot(self.W.T).astype(x.dtype, copy=False)
		if len(inputs) == 4:
			b = inputs[3]
			y += b
		return y,

	def backward(self, inputs, grad_outputs):
		x = _as_mat(inputs[0])
		V = inputs[1]
		g = inputs[2]
		W = self.W
		xp = cuda.get_array_module(x)

		gy = grad_outputs[0]
		gx = gy.dot(W).astype(x.dtype, copy=False).reshape(inputs[0].shape)
		gW = gy.T.dot(x).astype(W.dtype, copy=False)

		gg = xp.sum(gW * self.V_normalized, axis=1, keepdims=True).astype(g.dtype, copy=False)
		gV = g * (gW - gg * self.V_normalized) / self.norm
		gV = gV.astype(V.dtype, copy=False)

		if len(inputs) == 4:
			gb = gy.sum(0)
			return gx, gV, gg, gb
		else:
			return gx, gV, gg

def linear(x, V, g, b=None):
	if b is None:
		return LinearFunction()(x, V, g)
	else:
		return LinearFunction()(x, V, g, b)

class Linear(link.Link):

	def __init__(self, in_size, out_size=None, nobias=False, initialV=None):
		super(Linear, self).__init__()

		if out_size is None:
			in_size, out_size = None, in_size
		self.out_size = out_size
		self.nobias = nobias

		with self.init_scope():
			V_initializer = initializers._get_initializer(initialV)
			self.V = variable.Parameter(V_initializer)
			if in_size is not None:
				self.V.initialize((self.out_size, in_size))

			self.b = variable.Parameter(None)
			self.g = variable.Parameter(None)			

	def _initialize_params(self, t):
		xp = cuda.get_array_module(t)
		self.mean_t = xp.mean(t, axis=0)
		self.std_t = xp.sqrt(xp.var(t, axis=0))
		g = 1 / self.std_t
		b = -self.mean_t / self.std_t

		# print "g <- {}, b <- {}".format(g, b)

		with self.init_scope():
			if self.nobias == False:
				self.b = variable.Parameter(b)
			self.g = variable.Parameter(g.reshape((self.out_size, 1)))

	@property
	def W(self):
		V = self.V.data
		xp = cuda.get_array_module(V)
		norm = _norm(V)
		V = V / norm
		return self.g.data * V

	def __call__(self, x):
		if self.g.data is None:
			if self.V.data is None:
				self.V.initialize((self.out_size, x.size // x.shape[0]))
			xp = cuda.get_array_module(x.data)
			t = linear(x, self.V, Variable(xp.full((self.out_size, 1), 1.0).astype(x.dtype)))	# compute output with g = 1 and without bias
			self._initialize_params(t.data)
			return (t - self.mean_t) / self.std_t

		return linear(x, self.V, self.g, self.b)
