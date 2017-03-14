# -*- coding: utf-8 -*-
import math
import json, os, sys
from args import args
from chainer import cuda
sys.path.append(os.path.split(os.getcwd())[0])
from gan import GAN, DiscriminatorParams, GeneratorParams
from sequential import Sequential
from sequential.layers import Linear, BatchNormalization, Deconvolution2D, Convolution2D, PixelShuffler2D
from sequential.functions import Activation, dropout, gaussian_noise, tanh, sigmoid, reshape
from sequential.util import get_paddings_of_deconv_layers

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
	config.clamp_lower = -0.01
	config.clamp_upper = 0.01
	config.num_critic = 1
	config.weight_std = 0.001
	config.weight_initializer = "Normal"
	config.nonlinearity = "leaky_relu"
	config.optimizer = "rmsprop"
	config.learning_rate = 0.0001
	config.momentum = 0.5
	config.gradient_clipping = 10
	config.weight_decay = 0

	discriminator = Sequential()
	discriminator.add(Convolution2D(3, 32, ksize=4, stride=2, pad=1))
	discriminator.add(BatchNormalization(32))
	discriminator.add(Activation(config.nonlinearity))
	discriminator.add(Convolution2D(32, 64, ksize=4, stride=2, pad=1))
	discriminator.add(BatchNormalization(64))
	discriminator.add(Activation(config.nonlinearity))
	discriminator.add(Convolution2D(64, 128, ksize=4, stride=2, pad=1))
	discriminator.add(BatchNormalization(128))
	discriminator.add(Activation(config.nonlinearity))
	discriminator.add(Convolution2D(128, 256, ksize=4, stride=2, pad=1))
	discriminator.add(BatchNormalization(256))
	discriminator.add(Activation(config.nonlinearity))
	discriminator.add(Convolution2D(256, 512, ksize=4, stride=2, pad=0))

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
	config.weight_std = 0.02
	config.weight_initializer = "Normal"
	config.nonlinearity = "relu"
	config.optimizer = "Adam"
	config.learning_rate = 0.0001
	config.momentum = 0.5
	config.gradient_clipping = 10
	config.weight_decay = 0
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
	config.weight_std = 0.02
	config.weight_initializer = "Normal"
	config.nonlinearity = "relu"
	config.optimizer = "Adam"
	config.learning_rate = 0.0001
	config.momentum = 0.5
	config.gradient_clipping = 10
	config.weight_decay = 0

	generator = Sequential()
	projection_size = 6
	
	# Deconvolution version
	paddings = get_paddings_of_deconv_layers(image_width, num_layers=4, ksize=4, stride=2)
	generator.add(Linear(config.ndim_input, 256 * projection_size ** 2))
	generator.add(Activation(config.nonlinearity))
	generator.add(BatchNormalization(256 * projection_size ** 2))
	generator.add(reshape((-1, 256, projection_size, projection_size)))
	generator.add(Deconvolution2D(256, 128, ksize=4, stride=2, pad=paddings.pop(0)))
	generator.add(BatchNormalization(128))
	generator.add(Activation(config.nonlinearity))
	generator.add(Deconvolution2D(128, 64, ksize=4, stride=2, pad=paddings.pop(0)))
	generator.add(BatchNormalization(64))
	generator.add(Activation(config.nonlinearity))
	generator.add(Deconvolution2D(64, 32, ksize=4, stride=2, pad=paddings.pop(0)))
	generator.add(BatchNormalization(32))
	generator.add(Activation(config.nonlinearity))
	generator.add(Deconvolution2D(32, 3, ksize=4, stride=2, pad=paddings.pop(0)))

	# PixelShuffler version
	# generator.add(Linear(config.ndim_input, 256 * projection_size ** 2))
	# generator.add(Activation(config.nonlinearity))
	# generator.add(BatchNormalization(256 * projection_size ** 2))
	# generator.add(reshape((-1, 256, projection_size, projection_size)))
	# generator.add(PixelShuffler2D(256, 128, r=2))
	# generator.add(BatchNormalization(128))
	# generator.add(Activation(config.nonlinearity))
	# generator.add(PixelShuffler2D(128, 64, r=2))
	# generator.add(BatchNormalization(64))
	# generator.add(Activation(config.nonlinearity))
	# generator.add(PixelShuffler2D(64, 32, r=2))
	# generator.add(BatchNormalization(32))
	# generator.add(Activation(config.nonlinearity))
	# generator.add(PixelShuffler2D(32, 3, r=2))

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
