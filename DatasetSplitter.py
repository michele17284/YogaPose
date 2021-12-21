import os, cv2, json
from shutil import copy
from os.path import join
import random


def split(root, percentage=0.9, outdir=""):
	if outdir == "": outdir = root
	train_path = join(outdir, "train")
	test_path = join(outdir, "test")

	for _, directory_class in enumerate(os.listdir(root)):
		class_path = join(root, directory_class)
		onlyfiles = [f for f in os.listdir(class_path) if os.path.isfile(join(class_path, f))]
		random.shuffle(onlyfiles)
		train_split_position = (len(onlyfiles) // 10) * int(percentage * 10)
		train = onlyfiles[:train_split_position]
		test = onlyfiles[train_split_position:]
		for file in train:
			src = join(class_path, file)
			dst = join(train_path, directory_class)
			if not os.path.exists(dst): os.makedirs(dst)
			dst = join(dst, file)
			copy(src, dst)
		for file in test:
			src = join(class_path, file)
			dst = join(test_path, directory_class)
			if not os.path.exists(dst): os.makedirs(dst)
			dst = join(dst, file)
			copy(src, dst)


#split("./data/images", outdir="./data/split")
