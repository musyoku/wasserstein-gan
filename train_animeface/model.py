# -*- coding: utf-8 -*-
import math
import json, os, sys
from args import args
from chainer import cuda
sys.path.append(os.path.split(os.getcwd())[0])
from params import Params
from gan import GAN, DiscriminatorParams, GeneratorParams
from sequential import Sequential
from sequential.layers import Linear, BatchNormalization, Deconvolution2D, Convolution2D, MinibatchDiscrimination
from sequential.functions import Activation, dropout, gaussian_noise, tanh, sigmoid, reshape, reshape_1d
from sequential.util import get_conv_padding, get_paddings_of_deconv_layers, get_in_size_of_deconv_layers

# load params.json
try:
	os.mkdir(args.model_dir)
except:
	pass

# data
image_width = 96
image_height = image_width
ndim_z = 50

# specify discriminator
discriminator_sequence_filename = args.model_dir + "/discriminator.json"

if os.path.isfile(discriminator_sequence_filename):
	print "loading", discriminator_sequence_filename
	with open(discriminator_sequence_filename, "r") as f:
		try:
			params = json.load(f)
		except Exception as e:
			raise Exception("could not load {}".format(discriminator_sequence_filename))
else:
	config = DiscriminatorParams()
	config.weight_init_std = 0.02
	config.weight_initializer = "Normal"
	config.use_weightnorm = False
	config.nonlinearity = "elu"
	config.optimizer = "Adam"
	config.learning_rate = 0.0001
	config.momentum = 0.5
	config.gradient_clipping = 10
	config.weight_decay = 0
	config.use_feature_matching = False
	config.use_minibatch_discrimination = False

	discriminator = Sequential(weight_initializer=config.weight_initializer, weight_init_std=config.weight_init_std)
	discriminator.add(gaussian_noise(std=0.3))
	discriminator.add(Convolution2D(3, 32, ksize=4, stride=2, pad=1, use_weightnorm=config.use_weightnorm))
	discriminator.add(BatchNormalization(32))
	discriminator.add(Activation(config.nonlinearity))
	discriminator.add(Convolution2D(32, 64, ksize=4, stride=2, pad=1, use_weightnorm=config.use_weightnorm))
	discriminator.add(BatchNormalization(64))
	discriminator.add(Activation(config.nonlinearity))
	discriminator.add(Convolution2D(64, 128, ksize=4, stride=2, pad=1, use_weightnorm=config.use_weightnorm))
	discriminator.add(BatchNormalization(128))
	discriminator.add(Activation(config.nonlinearity))
	discriminator.add(Convolution2D(128, 256, ksize=4, stride=2, pad=1, use_weightnorm=config.use_weightnorm))
	discriminator.add(BatchNormalization(256))
	discriminator.add(Activation(config.nonlinearity))
	if config.use_minibatch_discrimination:
		discriminator.add(reshape_1d())
		discriminator.add(MinibatchDiscrimination(None, num_kernels=50, ndim_kernel=5, train_weights=True))
	discriminator.add(Convolution2D(256, 512, ksize=4, stride=2, pad=0, use_weightnorm=config.use_weightnorm))

	params = {
		"config": config.to_dict(),
		"model": discriminator.to_dict(),
	}

	with open(discriminator_sequence_filename, "w") as f:
		json.dump(params, f, indent=4, sort_keys=True, separators=(',', ': '))

discriminator_params = params

# specify generator
generator_sequence_filename = args.model_dir + "/generator.json"

if os.path.isfile(generator_sequence_filename):
	print "loading", generator_sequence_filename
	with open(generator_sequence_filename, "r") as f:
		try:
			params = json.load(f)
		except:
			raise Exception("could not load {}".format(generator_sequence_filename))
else:
	config = GeneratorParams()
	config.ndim_input = ndim_z
	config.distribution_output = "tanh"
	config.use_weightnorm = False
	config.weight_init_std = 0.02
	config.weight_initializer = "Normal"
	config.nonlinearity = "relu"
	config.optimizer = "Adam"
	config.learning_rate = 0.0001
	config.momentum = 0.5
	config.gradient_clipping = 10
	config.weight_decay = 0

	# model
	# compute projection width
	input_size = get_in_size_of_deconv_layers(image_width, num_layers=4, ksize=4, stride=2)
	# compute required paddings
	paddings = get_paddings_of_deconv_layers(image_width, num_layers=4, ksize=4, stride=2)

	generator = Sequential(weight_initializer=config.weight_initializer, weight_init_std=config.weight_init_std)
	generator.add(Linear(config.ndim_input, 512 * input_size ** 2, use_weightnorm=config.use_weightnorm))
	generator.add(Activation(config.nonlinearity))
	generator.add(BatchNormalization(512 * input_size ** 2))
	generator.add(reshape((-1, 512, input_size, input_size)))
	generator.add(Deconvolution2D(512, 256, ksize=4, stride=2, pad=paddings.pop(0), use_weightnorm=config.use_weightnorm))
	generator.add(BatchNormalization(256))
	generator.add(Activation(config.nonlinearity))
	generator.add(Deconvolution2D(256, 128, ksize=4, stride=2, pad=paddings.pop(0), use_weightnorm=config.use_weightnorm))
	generator.add(BatchNormalization(128))
	generator.add(Activation(config.nonlinearity))
	generator.add(Deconvolution2D(128, 3, ksize=4, stride=2, pad=paddings.pop(0), use_weightnorm=config.use_weightnorm))
	if config.distribution_output == "sigmoid":
		generator.add(sigmoid())
	if config.distribution_output == "tanh":
		generator.add(tanh())

	params = {
		"config": config.to_dict(),
		"model": generator.to_dict(),
	}

	with open(generator_sequence_filename, "w") as f:
		json.dump(params, f, indent=4, sort_keys=True, separators=(',', ': '))

generator_params = params

gan = GAN(discriminator_params, generator_params)
gan.load(args.model_dir)

if args.gpu_device != -1:
	cuda.get_device(args.gpu_device).use()
	gan.to_gpu()
