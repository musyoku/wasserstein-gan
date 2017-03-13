# -*- coding: utf-8 -*-
import os
import numpy as np
import mnist_tools

def load_train_images():
	return mnist_tools.load_train_images()

def load_test_images():
	return mnist_tools.load_test_images()

def sample_data(images, batchsize):
	ndim_x = images[0].size
	image_batch = np.zeros((batchsize, ndim_x), dtype=np.float32)
	indices = np.random.choice(np.arange(len(images), dtype=np.int32), size=batchsize, replace=False)
	for j in range(batchsize):
		data_index = indices[j]
		img = images[data_index].astype(np.float32)
		image_batch[j] = img.reshape((ndim_x,))
	return image_batch