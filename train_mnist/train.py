import numpy as np
import os, sys, time, math
from chainer import cuda
from chainer import functions as F
sys.path.append(os.path.split(os.getcwd())[0])
import dataset
from progress import Progress
from mnist_tools import load_train_images, load_test_images
from model import discriminator_params, generator_params, gan
from args import args
from plot import plot

def main():
	# load MNIST images
	images, labels = dataset.load_train_images()

	# config
	discriminator_config = gan.config_discriminator
	generator_config = gan.config_generator

	# settings
	max_epoch = 1000
	num_updates_per_epoch = 500
	plot_interval = 5
	batchsize_true = 100
	batchsize_fake = batchsize_true

	# seed
	np.random.seed(args.seed)
	if args.gpu_device != -1:
		cuda.cupy.random.seed(args.seed)

	# training
	progress = Progress()
	for epoch in xrange(1, max_epoch + 1):
		progress.start_epoch(epoch, max_epoch)
		sum_loss_critic = 0
		sum_loss_generator = 0

		for t in xrange(num_updates_per_epoch):

			for k in xrange(discriminator_config.num_critic):
				# clamp parameters to a cube
				gan.clip_discriminator_weights()
				# gan.decay_discriminator_weights()

				# sample true data from data distribution
				images_true = dataset.sample_data(images, batchsize_true, binarize=False)
				# sample fake data from generator
				images_fake = gan.generate_x(batchsize_fake)
				images_fake.unchain_backward()

				fw_true, activations_true = gan.discriminate(images_true)
				fw_fake, _ = gan.discriminate(images_fake)

				loss_critic = -F.sum(fw_true - fw_fake) / batchsize_true
				sum_loss_critic += float(loss_critic.data) / discriminator_config.num_critic

				# update discriminator
				gan.backprop_discriminator(loss_critic)

			# generator loss
			images_fake = gan.generate_x(batchsize_fake)
			fw_fake, activations_fake = gan.discriminate(images_fake)
			loss_generator = -F.sum(fw_fake) / batchsize_fake

			# feature matching
			if discriminator_config.use_feature_matching:
				features_true = activations_true[-1]
				features_true.unchain_backward()
				if batchsize_true != batchsize_fake:
					images_fake = gan.generate_x(batchsize_true)
					_, activations_fake = gan.discriminate(images_fake, apply_softmax=False)
				features_fake = activations_fake[-1]
				loss_generator += F.mean_squared_error(features_true, features_fake)

			# update generator
			gan.backprop_generator(loss_generator)
			sum_loss_generator += float(loss_generator.data)
			if t % 10 == 0:
				progress.show(t, num_updates_per_epoch, {})

		gan.save(args.model_dir)

		progress.show(num_updates_per_epoch, num_updates_per_epoch, {
			"wasserstein": -sum_loss_critic / num_updates_per_epoch,
			"loss_g": sum_loss_generator / num_updates_per_epoch,
		})

		if epoch % plot_interval == 0 or epoch == 1:
			plot(filename="epoch_{}_time_{}min".format(epoch, progress.get_total_time()))

if __name__ == "__main__":
	main()
