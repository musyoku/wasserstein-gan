# -*- coding: utf-8 -*-
import math
import numpy as np
from six import moves
from chainer import cuda, Variable, initializers, link, function, variable
from chainer.utils import conv, type_check, argument
from chainer.functions.connection import deconvolution_2d, convolution_2d

if cuda.cudnn_enabled:
	cudnn = cuda.cudnn
	libcudnn = cuda.cudnn.cudnn
	_cudnn_version = libcudnn.getVersion()
	_fwd_pref = libcudnn.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT
	if _cudnn_version >= 4000:
		_bwd_filter_pref = \
			libcudnn.CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT
		_bwd_data_pref = \
			libcudnn.CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT

_check_cudnn_acceptable_type = convolution_2d._check_cudnn_acceptable_type

def _norm(W):
	xp = cuda.get_array_module(W)
	norm = xp.sqrt(xp.sum(W ** 2, axis=(0, 2, 3))) + 1e-9
	norm = norm.reshape((1, -1, 1, 1))
	return norm

def _pair(x):
	if hasattr(x, "__getitem__"):
		return x
	return x, x

class Deconvolution2DFunction(deconvolution_2d.Deconvolution2DFunction):

	def __init__(self, stride=1, pad=0, outsize=None, use_cudnn=True):
		self.sy, self.sx = _pair(stride)
		self.ph, self.pw = _pair(pad)
		self.use_cudnn = use_cudnn
		self.outh, self.outw = (None, None) if outsize is None else outsize

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
			x_type.shape[1] == v_type.shape[0]
		)

		if self.outh is not None:
			type_check.expect(
				x_type.shape[2] ==
				conv.get_conv_outsize(self.outh, v_type.shape[2],self.sy, self.ph),
			)
		if self.outw is not None:
			type_check.expect(
				x_type.shape[3] ==
				conv.get_conv_outsize(self.outw, v_type.shape[3], self.sx, self.pw),
			)

		if type_check.eval(n_in) == 4:
			b_type = in_types[3]
			type_check.expect(
				b_type.dtype == x_type.dtype,
				b_type.ndim == 1,
				b_type.shape[0] == v_type.shape[1]
			)

	def forward_cpu(self, inputs):
		x, V, g = inputs[:3]
		b = inputs[3] if len(inputs) == 4 else None
		
		self.norm = _norm(V)
		self.V_normalized = V / self.norm
		self.W = g * self.V_normalized

		if b is None:
			return super(Deconvolution2DFunction, self).forward_cpu((x, self.W))
		return super(Deconvolution2DFunction, self).forward_cpu((x, self.W, b))

	def forward_gpu(self, inputs):
		x, V, g = inputs[:3]
		b = inputs[3] if len(inputs) == 4 else None

		self.norm = _norm(V)
		self.V_normalized = V / self.norm
		self.W = g * self.V_normalized
		
		if b is None:
			return super(Deconvolution2DFunction, self).forward_gpu((x, self.W))
		return super(Deconvolution2DFunction, self).forward_gpu((x, self.W, b))

	def backward_cpu(self, inputs, grad_outputs):
		x, V, g = inputs[:3]
		b = inputs[3] if len(inputs) == 4 else None
		if b is None:
			gb = None
			gx, gW = super(Deconvolution2DFunction, self).backward_cpu((x, self.W), grad_outputs)
		else:
			gx, gW, gb = super(Deconvolution2DFunction, self).backward_cpu((x, self.W, b), grad_outputs)

		xp = cuda.get_array_module(x)
		gg = xp.sum(gW * self.V_normalized, axis=(0, 2, 3), keepdims=True).astype(g.dtype, copy=False)
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
			gx, gW = super(Deconvolution2DFunction, self).backward_gpu((x, self.W), grad_outputs)
		else:
			gx, gW, gb = super(Deconvolution2DFunction, self).backward_gpu((x, self.W, b), grad_outputs)

		xp = cuda.get_array_module(x)
		gg = xp.sum(gW * self.V_normalized, axis=(0, 2, 3), keepdims=True).astype(g.dtype, copy=False)
		gV = g * (gW - gg * self.V_normalized) / self.norm
		gV = gV.astype(V.dtype, copy=False)

		if b is None:
			return gx, gV, gg
		else:
			return gx, gV, gg, gb


def deconvolution_2d(x, V, g, b=None, stride=1, pad=0, outsize=None, **kwargs):
	argument.check_unexpected_kwargs(
		kwargs, deterministic="deterministic argument is not "
		"supported anymore. "
		"Use chainer.using_config('cudnn_deterministic', value) "
		"context where value is either `True` or `False`.")
	argument.assert_kwargs_empty(kwargs)

	func = Deconvolution2DFunction(stride, pad, outsize)
	if b is None:
		return func(x, V, g)
	else:
		return func(x, V, g, b)

class Deconvolution2D(link.Link):

	def __init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0,
				 nobias=False, outsize=None, initialV=None, 
				 **kwargs):
		super(Deconvolution2D, self).__init__()

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
		self.outsize = (None, None) if outsize is None else outsize
		self.out_channels = out_channels
		self.nobias = nobias

		with self.init_scope():
			V_initializer = initializers._get_initializer(initialV)
			self.V = variable.Parameter(V_initializer)
			if in_channels is not None:
				kh, kw = _pair(self.ksize)
				V_shape = (in_channels, self.out_channels, kh, kw)
				self.V.initialize(V_shape)

			self.b = None if nobias else variable.Parameter(None)
			self.g = variable.Parameter(None)	

	def _initialize_params(self, t):
		xp = cuda.get_array_module(t)
		# 出力チャネルごとにミニバッチ平均をとる
		self.mean_t = xp.mean(t, axis=(0, 2, 3)).reshape((1, -1, 1, 1))
		self.std_t = xp.sqrt(xp.var(t, axis=(0, 2, 3))).reshape((1, -1, 1, 1))
		g = 1 / self.std_t
		b = -self.mean_t / self.std_t

		# print "g <- {}, b <- {}".format(g.reshape((-1,)), b.reshape((-1,)))

		with self.init_scope():
			if self.nobias == False:
				self.b = variable.Parameter(b.reshape((-1,)))
			self.g = variable.Parameter(g.reshape((1, self.out_channels, 1, 1)))

	def __call__(self, x):
		if self.g.data is None:
			if self.V.data is None:
				kh, kw = _pair(self.ksize)
				V_shape = (x.shape[1], self.out_channels, kh, kw)
				self.V.initialize(V_shape)
			xp = cuda.get_array_module(x.data)
			t = deconvolution_2d(x, self.V, Variable(xp.full((1, self.out_channels, 1, 1), 1.0).astype(x.dtype)), None, self.stride, self.pad, self.outsize)	# compute output with g = 1 and without bias
			self._initialize_params(t.data)
			return (t - self.mean_t) / self.std_t

		return deconvolution_2d(x, self.V, self.g, self.b, self.stride, self.pad, self.outsize)
