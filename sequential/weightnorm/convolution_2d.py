# -*- coding: utf-8 -*-
import math
import numpy as np
from six import moves
from chainer import cuda, variable, initializers, link, function, Variable
from chainer.utils import conv, type_check, argument
from chainer.functions.connection import convolution_2d

if cuda.cudnn_enabled:
	cudnn = cuda.cudnn
	libcudnn = cuda.cudnn.cudnn
	_cudnn_version = libcudnn.getVersion()
	_fwd_pref = libcudnn.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT
	if _cudnn_version >= 3000:
		_bwd_filter_pref = \
			libcudnn.CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT
		_bwd_data_pref = \
			libcudnn.CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT


def _check_cudnn_acceptable_type(x_dtype, W_dtype):
	return x_dtype == W_dtype and (
		_cudnn_version >= 3000 or x_dtype != numpy.float16)

def _norm(W):
	xp = cuda.get_array_module(W)
	norm = xp.sqrt(xp.sum(W ** 2, axis=(1, 2, 3))) + 1e-9
	norm = norm.reshape((-1, 1, 1, 1))
	return norm

def _pair(x):
	if hasattr(x, "__getitem__"):
		return x
	return x, x

class Convolution2DFunction(convolution_2d.Convolution2DFunction):

	def check_type_forward(self, in_types):
		n_in = in_types.size()
		type_check.expect(3 <= n_in, n_in <= 4)
		x_type = in_types[0]
		v_type = in_types[1]
		g_type = in_types[2]
		type_check.expect(
			x_type.dtype.kind == "f",
			v_type.dtype.kind == "f",
			g_type.dtype.kind == "f",
			x_type.ndim == 4,
			v_type.ndim == 4,
			g_type.ndim == 4,
			x_type.shape[1] == v_type.shape[1],
		)

		if type_check.eval(n_in) == 4:
			b_type = in_types[3]
			type_check.expect(
				b_type.dtype == x_type.dtype,
				b_type.ndim == 1,
				b_type.shape[0] == v_type.shape[0],
			)

	def forward_cpu(self, inputs):
		x, V, g = inputs[:3]
		b = inputs[3] if len(inputs) == 4 else None
		
		self.norm = _norm(V)
		self.V_normalized = V / self.norm
		self.W = g * self.V_normalized

		if b is None:
			return super(Convolution2DFunction, self).forward_cpu((x, self.W))
		return super(Convolution2DFunction, self).forward_cpu((x, self.W, b))

	def forward_gpu(self, inputs):
		x, V, g = inputs[:3]
		b = inputs[3] if len(inputs) == 4 else None

		self.norm = _norm(V)
		self.V_normalized = V / self.norm
		self.W = g * self.V_normalized

		if b is None:
			return super(Convolution2DFunction, self).forward_gpu((x, self.W))
		return super(Convolution2DFunction, self).forward_gpu((x, self.W, b))

	def backward_cpu(self, inputs, grad_outputs):
		x, V, g = inputs[:3]
		b = inputs[3] if len(inputs) == 4 else None
		if b is None:
			gb = None
			gx, gW = super(Convolution2DFunction, self).backward_cpu((x, self.W), grad_outputs)
		else:
			gx, gW, gb = super(Convolution2DFunction, self).backward_cpu((x, self.W, b), grad_outputs)

		xp = cuda.get_array_module(x)
		gg = xp.sum(gW * self.V_normalized, axis=(1, 2, 3), keepdims=True).astype(g.dtype, copy=False)
		gV = g * (gW - gg * self.V_normalized) / self.norm
		gV = gV.astype(V.dtype, copy=False)

		if b is None:
			return gx, gV, gg
		else:
			return gx, gV, gg, gb

	def backward_gpu(self, inputs, grad_outputs):
		x, V, g = inputs[:3]
		b = inputs[3] if len(inputs) == 4 else None
		if b is None:
			gb = None
			gx, gW = super(Convolution2DFunction, self).backward_gpu((x, self.W), grad_outputs)
		else:
			gx, gW, gb = super(Convolution2DFunction, self).backward_gpu((x, self.W, b), grad_outputs)

		xp = cuda.get_array_module(x)
		gg = xp.sum(gW * self.V_normalized, axis=(1, 2, 3), keepdims=True).astype(g.dtype, copy=False)
		gV = g * (gW - gg * self.V_normalized) / self.norm
		gV = gV.astype(V.dtype, copy=False)

		if b is None:
			return gx, gV, gg
		else:
			return gx, gV, gg, gb

def convolution_2d(x, V, g, b=None, stride=1, pad=0, cover_all=False, **kwargs):
	argument.check_unexpected_kwargs(
		kwargs, deterministic="deterministic argument is not "
		"supported anymore. "
		"Use chainer.using_config('cudnn_deterministic', value) "
		"context where value is either `True` or `False`.")
	argument.assert_kwargs_empty(kwargs)
	
	func = Convolution2DFunction(stride, pad, cover_all)
	if b is None:
		return func(x, V, g)
	else:
		return func(x, V, g, b)

class Convolution2D(link.Link):

	def __init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0, nobias=False, initialV=None, **kwargs):
		super(Convolution2D, self).__init__()

		argument.check_unexpected_kwargs(
			kwargs, deterministic="deterministic argument is not "
			"supported anymore. "
			"Use chainer.using_config('cudnn_deterministic', value) "
			"context where value is either `True` or `False`.")
		argument.assert_kwargs_empty(kwargs)

		if ksize is None:
			out_channels, ksize, in_channels = in_channels, out_channels, None

		self.ksize = ksize
		self.stride = _pair(stride)
		self.pad = _pair(pad)
		self.out_channels = out_channels
		self. nobias = nobias

		with self.init_scope():
			V_initializer = initializers._get_initializer(initialV)
			self.V = variable.Parameter(V_initializer)
			if in_channels is not None:
				kh, kw = _pair(self.ksize)
				V_shape = (self.out_channels, in_channels, kh, kw)
				self.V.initialize(V_shape)

			self.b = None if nobias else variable.Parameter(None)
			self.g = variable.Parameter(None)		

	def _initialize_params(self, t):
		xp = cuda.get_array_module(t)
		# 出力チャネルごとにミニバッチ平均をとる
		self.mean_t = xp.mean(t, axis=(0, 2, 3)).reshape(1, -1, 1, 1)
		self.std_t = xp.sqrt(xp.var(t, axis=(0, 2, 3))).reshape(1, -1, 1, 1)
		g = 1 / self.std_t
		b = -self.mean_t / self.std_t

		# print "g <- {}, b <- {}".format(g.reshape((-1,)), b.reshape((-1,)))

		with self.init_scope():
			if self.nobias == False:
				self.b = variable.Parameter(b.reshape((-1,)))
			self.g = variable.Parameter(g.reshape((self.out_channels, 1, 1, 1)))

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
				kh, kw = _pair(self.ksize)
				V_shape = (self.out_channels, x.shape[1], kh, kw)
				self.V.initialize(V_shape)
			xp = cuda.get_array_module(x.data)
			t = convolution_2d(x, self.V, Variable(xp.full((self.out_channels, 1, 1, 1), 1.0).astype(x.dtype)), None, self.stride, self.pad)	# compute output with g = 1 and without bias
			self._initialize_params(t.data)
			return (t - self.mean_t) / self.std_t

		return convolution_2d(x, self.V, self.g, self.b, self.stride, self.pad)
