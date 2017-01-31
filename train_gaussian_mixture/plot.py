# -*- coding: utf-8 -*-
import sampler, pylab, os
import seaborn as sns
from model import discriminator_params, generator_params, gan
from args import args
sns.set(font_scale=2)
sns.set_style("white")

def plot_kde(data, dir=None, filename="kde", color="Greens"):
	if dir is None:
		raise Exception()
	try:
		os.mkdir(dir)
	except:
		pass
	fig = pylab.gcf()
	fig.set_size_inches(16.0, 16.0)
	pylab.clf()
	bg_color  = sns.color_palette(color, n_colors=256)[0]
	ax = sns.kdeplot(data[:, 0], data[:,1], shade=True, cmap=color, n_levels=30, clip=[[-4, 4]]*2)
	ax.set_axis_bgcolor(bg_color)
	kde = ax.get_figure()
	pylab.xlim(-4, 4)
	pylab.ylim(-4, 4)
	kde.savefig("{}/{}.png".format(dir, filename))

def plot_scatter(data, dir=None, filename="scatter", color="blue"):
	if dir is None:
		raise Exception()
	try:
		os.mkdir(dir)
	except:
		pass
	fig = pylab.gcf()
	fig.set_size_inches(16.0, 16.0)
	pylab.clf()
	pylab.scatter(data[:, 0], data[:, 1], s=20, marker="o", edgecolors="none", color=color)
	pylab.xlim(-4, 4)
	pylab.ylim(-4, 4)
	pylab.savefig("{}/{}.png".format(dir, filename))

def main():
	num_samples = 10000
	samples_true = sampler.gaussian_mixture_circle(num_samples, num_cluster=generator_params["config"]["num_mixture"], scale=2, std=0.2)
	plot_scatter(samples_true, args.plot_dir, "scatter_true")
	plot_kde(samples_true, args.plot_dir, "kde_true")
	samples_fake = gan.to_numpy(gan.generate_x(num_samples, test=True))
	plot_scatter(samples_fake, args.plot_dir, "scatter_gen")
	plot_kde(samples_fake, args.plot_dir, "kde_gen")

if __name__ == "__main__":
	main()
