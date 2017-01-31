import numpy as np
import os, sys, time, math
from chainer import cuda
from chainer import functions as F
sys.path.append(os.path.split(os.getcwd())[0])
import sampler
from progress import Progress
from model import discriminator_params, generator_params, gan
from args import args
from plot import plot_kde, plot_scatter

def plot_samples(epoch, progress):
	samples_g = gan.generate_x(10000, from_gaussian=True)
	samples_g.unchain_backward()
	samples_g = gan.to_numpy(samples_g)
	try:
		plot_scatter(samples_g, dir=args.plot_dir, filename="scatter_epoch_{}_time_{}min".format(epoch, progress.get_total_time()))
		plot_kde(samples_g, dir=args.plot_dir, filename="kde_epoch_{}_time_{}min".format(epoch, progress.get_total_time()))
	except:
		pass

def main():
	# config
	discriminator_config = gan.config_discriminator
	generator_config = gan.config_generator

	# settings
	# _u -> unlabeled
	# _g -> generated
	max_epoch = 200
	num_updates_per_epoch = 500
	plot_interval = 5
	batchsize_u = 100
	batchsize_g = batchsize_u
	scale = 2.0

	# seed
	np.random.seed(args.seed)
	if args.gpu_device != -1:
		cuda.cupy.random.seed(args.seed)

	# training
	progress = Progress()
	plot_samples(0, progress)
	for epoch in xrange(1, max_epoch + 1):
		progress.start_epoch(epoch, max_epoch)
		sum_loss_unsupervised = 0
		sum_loss_adversarial = 0
		sum_dx_unlabeled = 0
		sum_dx_generated = 0

		for t in xrange(num_updates_per_epoch):
			# unrolling
			for k in xrange(args.unrolling_steps):
				# sample from data distribution
				samples_u = sampler.gaussian_mixture_double_circle(batchsize_u, generator_config.num_mixture, scale=scale, std=0.2)
				# sample from generator
				samples_g = gan.generate_x(batchsize_g, from_gaussian=True)
				samples_g.unchain_backward()

				# unsupervised loss
				# D(x) = Z(x) / {Z(x) + 1}, where Z(x) = \sum_{k=1}^K exp(l_k(x))
				# softplus(x) := log(1 + exp(x))
				# logD(x) = logZ(x) - log(Z(x) + 1)
				# 		  = logZ(x) - log(exp(log(Z(x))) + 1)
				# 		  = logZ(x) - softplus(logZ(x))
				# 1 - D(x) = 1 / {Z(x) + 1}
				# log{1 - D(x)} = log1 - log(Z(x) + 1)
				# 				= -log(exp(log(Z(x))) + 1)
				# 				= -softplus(logZ(x))
				log_zx_u, activations_u = gan.discriminate(samples_u / scale, apply_softmax=False)
				log_dx_u = log_zx_u - F.softplus(log_zx_u)
				dx_u = F.sum(F.exp(log_dx_u)) / batchsize_u
				loss_unsupervised = -F.sum(log_dx_u) / batchsize_u	# minimize negative logD(x)
				log_zx_g, _ = gan.discriminate(samples_g / scale, apply_softmax=False)
				loss_unsupervised += F.sum(F.softplus(log_zx_g)) / batchsize_u	# minimize negative log{1 - D(x)}

				# update discriminator
				gan.backprop_discriminator(loss_unsupervised)

				if k == 0:
					gan.cache_discriminator_weights()
					sum_loss_unsupervised += float(loss_unsupervised.data)
					sum_dx_unlabeled += float(dx_u.data)

			# generator loss
			samples_g = gan.generate_x(batchsize_g, from_gaussian=True)
			log_zx_g, activations_g = gan.discriminate(samples_g / scale, apply_softmax=False)
			log_dx_g = log_zx_g - F.softplus(log_zx_g)
			dx_g = F.sum(F.exp(log_dx_g)) / batchsize_g
			loss_generator = -F.sum(log_dx_g) / batchsize_u	# minimize negative logD(x)

			# feature matching
			if discriminator_config.use_feature_matching:
				features_true = activations_u[-1]
				features_true.unchain_backward()
				if batchsize_u != batchsize_g:
					samples_g = gan.generate_x(batchsize_u, from_gaussian=True)
					_, activations_g = gan.discriminate(samples_g / scale, apply_softmax=False)
				features_fake = activations_g[-1]
				loss_generator += F.mean_squared_error(features_true, features_fake)

			# update generator
			gan.backprop_generator(loss_generator)

			# update discriminator
			gan.restore_discriminator_weights()

			sum_loss_adversarial += float(loss_generator.data)
			sum_dx_generated += float(dx_g.data)
			if t % 10 == 0:
				progress.show(t, num_updates_per_epoch, {})

		gan.save(args.model_dir)

		progress.show(num_updates_per_epoch, num_updates_per_epoch, {
			"loss_u": sum_loss_unsupervised / num_updates_per_epoch,
			"loss_g": sum_loss_adversarial / num_updates_per_epoch,
			"dx_u": sum_dx_unlabeled / num_updates_per_epoch,
			"dx_g": sum_dx_generated / num_updates_per_epoch,
		})

		if epoch % plot_interval == 0 or epoch == 1:
			plot_samples(epoch, progress)

if __name__ == "__main__":
	main()
