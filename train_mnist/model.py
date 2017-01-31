# -*- coding: utf-8 -*-
import math
import json, os, sys
from args import args
from chainer import cuda
sys.path.append(os.path.split(os.getcwd())[0])
from params import Params
from gan import GAN, DiscriminatorParams, GeneratorParams
from sequential import Sequential
from sequential.layers import Linear, BatchNormalization, MinibatchDiscrimination
from sequential.functions import Activation, dropout, gaussian_noise, softmax

# load params.json
try:
	os.mkdir(args.model_dir)
except:
	pass

# data
image_width = 28
image_height = image_width
ndim_latent_code = 50

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
	config.ndim_input = image_width * image_height
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
	discriminator.add(Linear(None, 500, use_weightnorm=config.use_weightnorm))
	# discriminator.add(gaussian_noise(std=0.5))
	discriminator.add(Activation(config.nonlinearity))
	# discriminator.add(BatchNormalization(500))
	discriminator.add(Linear(None, 250, use_weightnorm=config.use_weightnorm))
	# discriminator.add(gaussian_noise(std=0.5))
	discriminator.add(Activation(config.nonlinearity))
	# discriminator.add(BatchNormalization(250))
	if config.use_minibatch_discrimination:
		discriminator.add(MinibatchDiscrimination(None, num_kernels=50, ndim_kernel=5))
	discriminator.add(Linear(None, 20, use_weightnorm=config.use_weightnorm))

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
	config.ndim_input = ndim_latent_code
	config.ndim_output = image_width * image_height
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

	# generator
	generator = Sequential(weight_initializer=config.weight_initializer, weight_init_std=config.weight_init_std)
	generator.add(Linear(config.ndim_input, 500, use_weightnorm=config.use_weightnorm))
	generator.add(BatchNormalization(500))
	generator.add(Activation(config.nonlinearity))
	generator.add(Linear(None, 500, use_weightnorm=config.use_weightnorm))
	generator.add(BatchNormalization(500))
	generator.add(Activation(config.nonlinearity))
	generator.add(Linear(None, config.ndim_output, use_weightnorm=config.use_weightnorm))
	if config.distribution_output == "sigmoid":
		generator.add(Activation("sigmoid"))
	if config.distribution_output == "tanh":
		generator.add(Activation("tanh"))

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