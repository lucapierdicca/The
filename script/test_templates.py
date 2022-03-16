from pprint import pprint
from draw_map import draw_map_templates
from utils import loadDataset, GaussianFilter
from utils import Classifier
from utils import Point

import time



def test_extract_feature_template():
	# controllare che i feature template siano calcolati bene nel filtro
	template = loadDataset("data/train/template2.csv",
						   "data/train/train_map_ground_truth.pickle")
	template = [step for step in template if step["clock"] == 10]

	filter = GaussianFilter("linear", "template", template)
	feature = filter.extractFeature(filter.templates["I"][0]["z"], handle_occlusion=True)

	for k, v in filter.templates.items():
		print(k, len(v))

	print(len(feature))
	pprint(feature)


def test_vis_collected_template():
	# controllare by visual inspection che i template collezionati siano ok
	template = loadDataset("data/train/template2.csv",
						   "data/train/train_map_ground_truth.pickle")
	#template = [step for step in template if step["clock"] == 10]
	template = [step for step in template if step["clock"] == 10 and (step["true_class"] != "C" or step["true_class"] == "C" and step["y"] > 2)]

	print(len(template))

	classifier = Classifier()
	rp, ro, me = [], [], []

	for step in template:
		rp.append([step["x"], step["y"]])
		ro.append(step["theta"])
		z = classifier.preProcess(step["world_model_long"], 3)
		me.append(z)

	draw_map_templates(rp, ro, me)


def test_pre_compute_rotated_template():
	z = {1: {"distance": 1},
		 2: {"distance": 2},
		 3: {"distance": 3},
		 4: {"distance": 4},
		 5: {"distance": 5}}

	tm = GaussianFilter.pre_compute_rotated_template(z)

	print(tm)


def test_min_distance(handle_occlusion):
	template = {1: {"distance": 1,
					"occluded": False},
				2: {"distance": 2,
					"occluded": False},
				3: {"distance": 3,
					"occluded": False}}


	tm = GaussianFilter.pre_compute_rotated_template(template)

	print(tm)

	z = {1: {"distance": 4,
			 "occluded": True},
		 2: {"distance": 6,
			 "occluded": False},
		 3: {"distance": 1,
			 "occluded": False}}

	min = GaussianFilter.min_distance(tm, z, handle_occlusion=handle_occlusion)

	print(min)


def test_naive_min_distance(handle_occlusion):
	template = {1: {"distance": 1,
					"occluded": False},
				2: {"distance": 2,
					"occluded": False},
				3: {"distance": 3,
					"occluded": False}}
	z = {1: {"distance": 4,
			 "occluded": True},
		 2: {"distance": 6,
			 "occluded": False},
		 3: {"distance": 1,
			 "occluded": False}}
	min = GaussianFilter.naive_min_distance(template, z, handle_occlusion=handle_occlusion)
	print(min)


if __name__ == '__main__':
	# handle_occlusion = False
	# start_time = time.time()
	# test_min_distance(handle_occlusion)
	# print(time.time() - start_time)
	# start_time = time.time()
	# test_naive_min_distance(handle_occlusion)
	# print(time.time() - start_time)

	test_vis_collected_template()
