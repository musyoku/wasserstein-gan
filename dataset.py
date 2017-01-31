# -*- coding: utf-8 -*-
import os, sys
import numpy as np
from StringIO import StringIO
from PIL import Image

def _load_images(image_dir, convert_to="RGB"):
	dataset = []
	fs = os.listdir(image_dir)
	i = 0
	for fn in fs:
		f = open("%s/%s" % (image_dir, fn), "rb")
		if convert_to == "L":
			img = np.asarray(Image.open(StringIO(f.read())).convert("L"), dtype=np.float32) / 255.0
		elif convert_to == "RGB":
			img = np.asarray(Image.open(StringIO(f.read())).convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255.0
		elif convert_to == "RGBA":
			img = np.asarray(Image.open(StringIO(f.read())).convert("RGBA"), dtype=np.float32).transpose(2, 0, 1) / 255.0
		else:
			raise Exception()

		img = (img * 2.0) - 1.0
		dataset.append(img)
		f.close()
		i += 1
		if i % 100 == 0 or i == len(fs):
			sys.stdout.write("\rloading images...({} / {})".format(i, len(fs)))
			sys.stdout.flush()
	sys.stdout.write("\n")
	return dataset

def load_binary_images(image_dir):
	return _load_images(image_dir, "L")

def load_rgb_images(image_dir):
	return _load_images(image_dir, "RGB")

def load_rgba_images(image_dir):
	return _load_images(image_dir, "RGBA")

def binarize_data(x, sampling=True, threshold=None):
	if sampling:
		threshold = np.random.uniform(size=x.shape)
	if threshold is None:
		raise Exception()
	return np.where(threshold < x, 1.0, 0.0).astype(np.float32)	