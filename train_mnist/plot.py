import sys, os, pylab
import numpy as np
sys.path.append(os.path.split(os.getcwd())[0])
from args import args
from model import gan

def tile_binary_images(x, dir=None, filename="x", row=10, col=10):
	if dir is None:
		raise Exception()
	try:
		os.mkdir(dir)
	except:
		pass
	fig = pylab.gcf()
	fig.set_size_inches(col * 2, row * 2)
	pylab.clf()
	pylab.gray()
	for m in range(row * col):
		pylab.subplot(row, col, m + 1)
		pylab.imshow(np.clip(x[m], 0, 1), interpolation="none")
		pylab.axis("off")
	pylab.savefig("{}/{}.png".format(dir, filename))

def plot(filename="gen"):
	try:
		os.mkdir(args.plot_dir)
	except:
		pass
	x_fake = gan.generate_x(100, test=True, as_numpy=True)
	x_fake = (x_fake + 1.0) / 2.0
	tile_binary_images(x_fake.reshape((-1, 28, 28)), dir=args.plot_dir, filename=filename)

if __name__ == "__main__":
	plot()
