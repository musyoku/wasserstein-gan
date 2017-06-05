# -*- coding: utf-8 -*-
import json, os, sys, chainer, math
from args import args
from chainer import cuda
sys.path.append(os.path.split(os.getcwd())[0])
from gan import GAN, DiscriminatorParams, GeneratorParams, to_object
from sequential import Sequential
from sequential.layers import Linear, BatchNormalization
from sequential.functions import Activation 

# load params.json
try:
	os.mkdir(args.model_dir)
except:
	pass

# specify discriminator
discriminator_sequence_filename = args.model_dir + "/discriminator.json"

if os.path.isfile(discriminator_sequence_filename):
	print "loading", discriminator_sequence_filename
	with open(discriminator_sequence_filename, "r") as f:
		try:
			params = json.load(f)
			chainer.global_config.discriminator = to_object(params["config"])
		except Exception as e:
			raise Exception("could not load {}".format(discriminator_sequence_filename))
else:
	config = DiscriminatorParams()
	config.ndim_input = 2
	config.clamp_lower = -0.01
	config.clamp_upper = 0.01
	config.num_critic = 5
	config.weight_std = 0.001
	config.weight_initializer = "Normal"
	config.nonlinearity = "leaky_relu"
	config.optimizer = "rmsprop"
	config.learning_rate = 0.0001
	config.momentum = 0.5
	config.gradient_clipping = 1
	config.weight_decay = 0

	chainer.global_config.discriminator = config

	discriminator = Sequential()
	discriminator.add(Linear(None, 128))
	discriminator.add(Activation(config.nonlinearity))
	discriminator.add(Linear(None, 128))

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
			chainer.global_config.generator = to_object(params["config"])
		except:
			raise Exception("could not load {}".format(generator_sequence_filename))
else:
	config = GeneratorParams()
	config.ndim_input = 256
	config.ndim_output = 2
	config.num_mixture = args.num_mixture
	config.distribution_output = "universal"
	config.weight_std = 0.02
	config.weight_initializer = "Normal"
	config.nonlinearity = "relu"
	config.optimizer = "adam"
	config.learning_rate = 0.0001
	config.momentum = 0.5
	config.gradient_clipping = 10
	config.weight_decay = 0

	chainer.global_config.generator = config

	# generator
	generator = Sequential()
	generator.add(Linear(config.ndim_input, 128))
	# generator.add(BatchNormalization(128))
	generator.add(Activation(config.nonlinearity))
	generator.add(Linear(None, 128))
	# generator.add(BatchNormalization(128))
	generator.add(Activation(config.nonlinearity))
	generator.add(Linear(None, config.ndim_output))

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