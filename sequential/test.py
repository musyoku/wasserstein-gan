import numpy as np
from chainer import Variable
from chainer import functions as F
from chainer import links as L
from sequential import Sequential, Residual
import layers
import functions
import util
from chain import Chain

# residual test
seq = Sequential(weight_std=0.001)
seq.add(layers.Linear(28*28, 500))
seq.add(layers.BatchNormalization(500))
seq.add(functions.Activation("relu"))
res = Residual(weight_std=100)
res.add(layers.Linear(500, 500))
res.add(layers.BatchNormalization(500))
res.add(functions.Activation("relu"))
seq.add(res)
json_str = seq.to_json()
seq = Sequential()
seq.from_json(json_str)
seq.build("Normal", 1)

x = np.random.normal(scale=1, size=(2, 28*28)).astype(np.float32)
x = Variable(x)
y = seq(x)
print y.data.shape
# check
for link in seq.links:
	if isinstance(link, L.Linear):
		print np.std(link.W.data), np.mean(link.W.data)
	if isinstance(link, Residual):
		for _link in link.links:
			if isinstance(_link, L.Linear):
				print np.std(_link.W.data), np.mean(_link.W.data)

# residual test 2
seq = Sequential()
seq.add(layers.Linear(28*28, 500))
seq.add(layers.BatchNormalization(500))
seq.add(functions.Activation("relu"))
if True:
	res1 = Residual()
	res1.add(layers.Linear(500, 1000))
	res1.add(layers.BatchNormalization(1000))
	res1.add(functions.Activation("relu"))
	if True:
		res2 = Residual()
		res2.add(layers.Linear(1000, 1000))
		res2.add(layers.BatchNormalization(1000))
		res2.add(functions.Activation("relu"))
		res1.add(res2)
	res1.add(layers.Linear(1000, 500))
	res1.add(layers.BatchNormalization(500))
	res1.add(functions.Activation("relu"))
	seq.add(res1)
json_str = seq.to_json()
seq = Sequential()
seq.from_json(json_str)
seq.build("Normal", 1)

x = np.random.normal(scale=1, size=(2, 28*28)).astype(np.float32)
x = Variable(x)
y = seq(x)
print y.data.shape


# JSON test
seq1 = Sequential()
seq1.add(layers.Linear(28*28, 500))
seq1.add(layers.BatchNormalization(500))
seq1.add(functions.relu())
json_str = seq1.to_json()
seq2 = Sequential()
seq2.from_json(json_str)
seq2.build("HeNormal", 0.1)

# Linear test
x = np.random.normal(scale=1, size=(2, 28*28)).astype(np.float32)
x = Variable(x)

seq = Sequential()
seq.add(layers.Linear(28*28, 500))
seq.add(layers.BatchNormalization(500))
seq.add(layers.Linear(500, 500))
seq.add(functions.Activation("clipped_relu"))
seq.add(layers.Linear(500, 500, use_weightnorm=True))
seq.add(functions.Activation("crelu"))	# crelu outputs 2x 
seq.add(layers.BatchNormalization(1000))
seq.add(layers.Linear(1000, 500))
seq.add(functions.Activation("elu"))
seq.add(layers.Linear(500, 500, use_weightnorm=False))
seq.add(functions.Activation("hard_sigmoid"))
seq.add(layers.BatchNormalization(500))
seq.add(layers.Linear(500, 500))
seq.add(functions.Activation("leaky_relu"))
seq.add(layers.Linear(500, 500, use_weightnorm=True))
seq.add(functions.Activation("relu"))
seq.add(layers.BatchNormalization(500))
seq.add(layers.Linear(500, 500))
seq.add(functions.Activation("sigmoid"))
seq.add(layers.Linear(500, 500, use_weightnorm=False))
seq.add(functions.Activation("softmax"))
seq.add(layers.BatchNormalization(500))
seq.add(layers.Linear(500, 500))
seq.add(functions.Activation("softplus"))
seq.add(layers.Linear(500, 500, use_weightnorm=True))
seq.add(functions.Activation("tanh"))
seq.add(layers.Linear(500, 10))
seq.build("Normal", 1.0)

y = seq(x)
print y.data.shape

# Conv test
x = np.random.normal(scale=1, size=(2, 3, 96, 96)).astype(np.float32)
x = Variable(x)

seq = Sequential()
seq.add(layers.Convolution2D(3, 64, ksize=4, stride=2, pad=0))
seq.add(layers.BatchNormalization(64))
seq.add(functions.Activation("relu"))
seq.add(layers.Convolution2D(64, 128, ksize=4, stride=2, pad=0))
seq.add(layers.BatchNormalization(128))
seq.add(functions.Activation("relu"))
seq.add(layers.Convolution2D(128, 256, ksize=4, stride=2, pad=0))
seq.add(layers.BatchNormalization(256))
seq.add(functions.Activation("relu"))
seq.add(layers.Convolution2D(256, 512, ksize=4, stride=2, pad=0))
seq.add(layers.BatchNormalization(512))
seq.add(functions.Activation("relu"))
seq.add(layers.Convolution2D(512, 1024, ksize=4, stride=2, pad=0))
seq.add(layers.BatchNormalization(1024))
seq.add(functions.Activation("relu"))
seq.add(layers.Linear(None, 10, use_weightnorm=True))
seq.add(functions.softmax())
seq.build("GlorotNormal", 0.5)

y = seq(x)
print y.data.shape

# Deconv test
x = np.random.normal(scale=1, size=(2, 100)).astype(np.float32)
x = Variable(x)

image_size = 96
# compute projection width
input_size = util.get_in_size_of_deconv_layers(image_size, num_layers=3, ksize=4, stride=2)
# compute required paddings
paddings = util.get_paddings_of_deconv_layers(image_size, num_layers=3, ksize=4, stride=2)

seq = Sequential()
seq.add(layers.Linear(100, 64 * input_size ** 2))
seq.add(layers.BatchNormalization(64 * input_size ** 2))
seq.add(functions.Activation("relu"))
seq.add(functions.reshape((-1, 64, input_size, input_size)))
seq.add(layers.Deconvolution2D(64, 32, ksize=4, stride=2, pad=paddings.pop(0), use_weightnorm=True))
seq.add(layers.BatchNormalization(32))
seq.add(functions.Activation("relu"))
seq.add(layers.Deconvolution2D(32, 16, ksize=4, stride=2, pad=paddings.pop(0), use_weightnorm=False))
seq.add(layers.BatchNormalization(16))
seq.add(functions.Activation("relu"))
seq.add(layers.Deconvolution2D(16, 3, ksize=4, stride=2, pad=paddings.pop(0), use_weightnorm=True))
json_str = seq.to_json()
seq.from_json(json_str)
seq.build("HeNormal", 0.1)

y = seq(x)
print y.data.shape

# train test
x = np.random.normal(scale=1, size=(128, 28*28)).astype(np.float32)
x = Variable(x)

seq = Sequential()
seq.add(layers.Linear(28*28, 500))
seq.add(layers.BatchNormalization(500))
seq.add(functions.Activation("relu"))
res = Residual()
res.add(layers.Linear(500, 500))
res.add(layers.BatchNormalization(500))
res.add(functions.Activation("relu"))
seq.add(res)
seq.add(layers.Linear(500, 500))
seq.add(layers.BatchNormalization(500))
seq.add(functions.Activation("relu"))
seq.add(layers.Linear(500, 28*28))

chain = Chain("GlorotNormal", 10)
chain.add_sequence(seq)
chain.setup_optimizers("adam", 0.01, momentum=0.9, weight_decay=0.000001, gradient_clipping=10)

# check
for link in seq.links:
	if isinstance(link, L.Linear):
		print np.std(link.W.data), np.mean(link.W.data)
	if isinstance(link, Residual):
		for _link in link.links:
			if isinstance(_link, L.Linear):
				print np.std(_link.W.data), np.mean(_link.W.data)

for i in xrange(1000):
	y = chain(x)
	loss = F.mean_squared_error(x, y)
	chain.backprop(loss)
	if i % 100 == 0:
		print float(loss.data)

# check
for link in seq.links:
	if isinstance(link, L.Linear):
		print np.std(link.W.data), np.mean(link.W.data)
	if isinstance(link, Residual):
		for _link in link.links:
			if isinstance(_link, L.Linear):
				print np.std(_link.W.data), np.mean(_link.W.data)
chain.save("model")

# weight test
x = np.random.normal(scale=1, size=(128, 28*28)).astype(np.float32)
x = Variable(x)

seq1 = Sequential()
seq1.add(layers.Linear(28*28, 500))

seq2 = Sequential("Normal", 100)
seq2.add(layers.Linear(28*28, 500))

seq3 = Sequential()
seq3.add(layers.Linear(28*28, 500))

chain = Chain("HeNormal", 0.1)
chain.add_sequence(seq1, name="seq1")
chain.add_sequence(seq2, name="seq2")
chain.add_sequence(seq3, name="seq3")

for link in chain.seq1.links:
	print np.std(link.W.data), np.mean(link.W.data)

for link in chain.seq2.links:
	print np.std(link.W.data), np.mean(link.W.data)

for link in chain.seq3.links:
	print np.std(link.W.data), np.mean(link.W.data)


seq1 = Sequential(weight_std=1.0)
seq1.add(layers.Linear(28*28, 500))

seq2 = Sequential(weight_std=0.1)
seq2.add(layers.Linear(28*28, 500))

seq3 = Sequential(weight_std=0.01)
seq3.add(layers.Linear(28*28, 500))

chain = Chain("Normal", 1.0)
chain.add_sequence(seq1, name="seq1")
chain.add_sequence(seq2, name="seq2")
chain.add_sequence(seq3, name="seq3")

# check
for link in chain.seq1.links:
	print np.std(link.W.data), np.mean(link.W.data)

for link in chain.seq2.links:
	print np.std(link.W.data), np.mean(link.W.data)

for link in chain.seq3.links:
	print np.std(link.W.data), np.mean(link.W.data)
