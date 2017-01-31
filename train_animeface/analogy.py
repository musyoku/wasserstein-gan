import sys, os
import numpy as np
from chainer import functions as F
sys.path.append(os.path.split(os.getcwd())[0])
import visualizer
from args import args
from model import gan

def run_method_1():
	# config
	discriminator_config = gan.config_discriminator
	generator_config = gan.config_generator

	num_col = 10
	num_generation = 20
	batchsize = 2 * num_generation
	base_z = gan.to_variable(gan.sample_z(batchsize))

	# optimize z
	class_true = gan.to_variable(np.zeros(batchsize, dtype=np.int32))
	for n in xrange(5):
		x_fake = gan.generate_x_from_z(base_z, test=True, as_numpy=False)
		discrimination_fake, _ = gan.discriminate(x_fake, apply_softmax=False, test=True)
		cross_entropy = F.softmax_cross_entropy(discrimination_fake, class_true)
		gan.backprop_generator(cross_entropy)
		base_z = gan.to_variable(base_z.data + base_z.grad * 0.01)
	base_z = gan.to_numpy(base_z)
	sum_z = np.sum(base_z)
	if sum_z != sum_z:
		raise Exception("NaN")

	mix_z = np.zeros((num_col * num_generation, generator_config.ndim_input), dtype=np.float32)
	for g in xrange(num_generation):
		for i in xrange(num_col):
			mix_z[g * num_col + i] = base_z[2 * g] * (i / float(num_col)) + base_z[2 * g + 1] * (1 - i / float(num_col))

	x_negative = gan.generate_x_from_z(mix_z, test=True, as_numpy=True)
	x_negative = (x_negative + 1.0) / 2.0
	visualizer.tile_rgb_images(x_negative.transpose(0, 2, 3, 1), dir=args.plot_dir, filename="analogy_1", row=num_generation, col=num_col)

if __name__ == '__main__':
	try:
		os.mkdir(args.plot_dir)
	except:
		pass
	run_method_1()
