import os
from bisect import bisect_left
from pprint import pprint

import numpy as np
from matplotlib import gridspec
from scipy.spatial import ConvexHull
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn import preprocessing
from sklearn.neighbors import BallTree
import csv
from collections import namedtuple, deque, Counter
import pickle
from sklearn.svm import SVC
from tabulate import tabulate
from tqdm import tqdm
from itertools import islice
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
from pgmpy.models import MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import VariableElimination


Point = namedtuple('Point','x y')

def to_radians(degree):
	return np.deg2rad(degree)

def norm(v):
	return np.linalg.norm(v)

def get_bearing(v1,v2,theta):
	R = np.array([[np.cos(theta), -np.sin(theta)],
				  [np.sin(theta), np.cos(theta)]])

	v3 = R.T@(v2-v1)

	return np.arctan2(v3[1],v3[0])[0]

def is_inside(p: np.array, poly_boundary_vertices: list[Point]):

	turn_angle = 0.0
	closed_poly_boundary_vertices = list(poly_boundary_vertices) + [poly_boundary_vertices[0]]

	for i in range(len(closed_poly_boundary_vertices) - 1):
		v = np.array([[closed_poly_boundary_vertices[i][0]], [closed_poly_boundary_vertices[i][1]]], dtype='float64')
		v_ = np.array([[closed_poly_boundary_vertices[i + 1][0]], [closed_poly_boundary_vertices[i + 1][1]]], dtype='float64')

		v -= p
		v_ -= p

		v_angle = np.arctan2(v[1],v[0])
		v__angle = np.arctan2(v_[1],v_[0])

		# non-euclidean
		angle_diff = (v_angle - v__angle)
		if abs(angle_diff) > np.pi:
			angle_diff = 2*np.pi - abs(angle_diff)

		turn_angle += angle_diff

		#print(v_angle, v__angle, angle_diff, turn_angle)

	if math.isclose(abs(turn_angle), 2*np.pi, rel_tol=0.001):
		return True
	else:
		return False

def is_inside_square(p: np.array, poly_boundary_vertices: list[Point]):
	pbv = poly_boundary_vertices
	x_distance = np.abs(pbv[0].x-pbv[2].x)
	y_distance = np.abs(pbv[0].y-pbv[2].y)
	if (np.abs(p[0] - pbv[0].x) < x_distance and
		np.abs(p[0] - pbv[2].x) < x_distance and
		np.abs(p[1] - pbv[0].y) < y_distance and
		np.abs(p[1] - pbv[2].y) < y_distance):

		return True
	else:
		return False

def load_dataset(dataset_path, map_ground_truth_path, row_range=None):

	with open(map_ground_truth_path,"rb") as f:
		map_ground_truth = pickle.load(f)

	file = open(dataset_path, mode="r")
	reader = csv.reader(file, delimiter='|')

	if row_range is not None:
		reader_sliced = islice(reader, row_range[0], row_range[1])
	else:
		reader_sliced = reader

	dataset = []
	for row in reader_sliced:
		step = {'clock':0,
				'fb_id':'',
				'x':0, 'y':0,
				'theta':0,
				'v_left':0,
				'v_right':0,
				'true_class':'',
				'color':None,
				'world_model_long':{},
				'world_model_short':{},
				'rab_readings':[]}

		step['clock'] = int(row[0])
		step['fb_id'] = str(row[1])
		step['x'] = float(row[2].replace(",","."))
		step['y'] = float(row[3].replace(",","."))
		step['theta'] = float(row[4].replace(",","."))
		step['v_left'] = float(row[5].replace(",","."))
		step['v_right'] = float(row[6].replace(",","."))

		p = np.array([[step['x']],[step['y']]])
		for classlbl, data in map_ground_truth.items():
			for bv in data['poly_boundary_vertices']:
				if is_inside(p, bv):
					step['true_class'] = classlbl

		if step['true_class'] not in ['V','C','I','G','S']:
			print(p)
			print("WARNING: no true class. Skip this step.")
			continue

		step['color'] = map_ground_truth[step['true_class']]['color']

		step['world_model_long'] = {float(row[i].replace(",", ".")): {"distance": float(row[i + 1].replace(",", ".")),
																	  "occluded": bool(int(row[i + 2]))}
									for i in range(7, 120 * 3 + 7, 3)}

		#step['world_model_short'] = {float(row[i].replace(",",".")):float(row[i+1].replace(",",".")) for i in range(120*2+7,120*2+120*2+7,2)}

		if "data" in dataset_path.split("/")[-1]:
			split_rab = row[-1].split("_")
			split_rab.pop() # remove the last element because it's a ""
			for i in range(0,len(split_rab)-2,3):
				distance, bearing, id = float(split_rab[i]),float(split_rab[i+1]), int(split_rab[i+2])
				r_p_neigh = np.array([[(distance/100)*np.cos(bearing)],
									  [(distance/100)*np.sin(bearing)]])

				w_R_r = np.array([[np.cos(step["theta"]), -np.sin(step["theta"])],
								  [np.sin(step["theta"]), np.cos(step["theta"])]])
				w_p_neigh = p + w_R_r@r_p_neigh

				true_class = ""
				for classlbl, data in map_ground_truth.items():
					for bv in data['poly_boundary_vertices']:
						if is_inside(w_p_neigh, bv):
							true_class = classlbl

				if true_class not in ['V','C','I','G','S']:
					print(p)
					print("WARNING: no true class. Skip this neighbor.")
					continue

				step['rab_readings'].append({"id":id,
									"range":distance,
									"bearing":bearing,
									"r_x":r_p_neigh[0][0], "r_y":r_p_neigh[1][0],
									"w_x":w_p_neigh[0][0], "w_y":w_p_neigh[1][0],
									"true_class":true_class})

		dataset.append(step)

	# 2nd pass to retrieve observations of each robot neighbor
	# if "data" in dataset_path.split("/")[-1]:
	# 	for step in dataset:
	# 		clock = step["clock"]
	# 		for neigh in step["rab_readings"]:
	# 			file.seek(0)
	# 			neighbor_id = int(neigh["id"])
	# 			start = ((neighbor_id)*(10000-9))+(clock-10)
	# 			reader2 = islice(reader, start, start+1)
	# 			for row in reader2:
	# 				world_model_long = {float(row[i].replace(",", ".")): {"distance": float(row[i + 1].replace(",", ".")),
	# 																			  "occluded": bool(int(row[i + 2]))}
	# 											for i in range(7, 120 * 3 + 7, 3)}
	#
	# 			neigh["world_model_long"] = world_model_long



	file.close()

	return dataset

class Data:
	def __init__(self, dataset_path, map_ground_truth_path):
		self.dataset_path = dataset_path
		with open(map_ground_truth_path,"rb") as f:
			self.map_ground_truth = pickle.load(f)

		self._init_data_structure_is_inside()

		self.files, self.readers = [],[]
		for file_name in [f for f in os.listdir(dataset_path) if f.endswith(".csv")]:
			prefix = file_name[:3]
			id = file_name[3:-4]
			extension = file_name[-4:]
			new_file_name = prefix+id.rjust(2,"0")+extension
			os.rename(dataset_path+"/"+file_name, dataset_path+"/"+new_file_name)
		for file_name in sorted([f for f in os.listdir(dataset_path) if f.endswith(".csv")]):
			self.files.append(open(dataset_path+"/"+file_name, mode="r"))
		for f in self.files:
			self.readers.append(csv.reader(f, delimiter='|'))

	def _init_data_structure_is_inside(self):

		classbound_to_colidx = {}
		array_idx = 0
		for classlbl, data in self.map_ground_truth.items():
			for bv_idx, bv in enumerate(data["poly_boundary_vertices"]):
				classbound_to_colidx[(classlbl,bv_idx)] = array_idx
				array_idx+=1

		self.colidx_to_classlbl = {v:k for k,v in classbound_to_colidx.items()}

		n_boundary = 0
		for classlbl, data in self.map_ground_truth.items():
			n_boundary+=len(data["poly_boundary_vertices"])

		# righe ci sono le xmin, xmax, ymin, ymax di tutti i poly_boundary_vertices
		self.separate_coord_matrix = np.zeros((4,n_boundary))

		for classlbl, data in self.map_ground_truth.items():
			for bv_idx, bv in enumerate(data["poly_boundary_vertices"]):
				sorted_bv = sorted(bv, key=lambda p:p[0])
				self.separate_coord_matrix[0, classbound_to_colidx[classlbl,bv_idx]] = sorted_bv[0].x
				self.separate_coord_matrix[1, classbound_to_colidx[classlbl,bv_idx]] = sorted_bv[-1].x
				sorted_bv = sorted(bv, key=lambda p:p[1])
				self.separate_coord_matrix[2, classbound_to_colidx[classlbl,bv_idx]] = sorted_bv[0].y
				self.separate_coord_matrix[3, classbound_to_colidx[classlbl,bv_idx]] = sorted_bv[-1].y


	def _is_inside(self, point):
		check_matrix = np.zeros_like(self.separate_coord_matrix)
		check_matrix[0,:] = np.where(self.separate_coord_matrix[0,:] < point[0],1,0)
		check_matrix[1,:] = np.where(self.separate_coord_matrix[1,:] > point[0],1,0)
		check_matrix[2,:] = np.where(self.separate_coord_matrix[2,:] < point[1],1,0)
		check_matrix[3,:] = np.where(self.separate_coord_matrix[3,:] > point[1],1,0)

		# print(point)
		# print(self.separate_coord_matrix)
		# print(check_matrix)
		# print(np.product(check_matrix, axis=0))

		colidx = np.argwhere(np.product(check_matrix, axis=0))
		#print(colidx[0,0])

		return self.colidx_to_classlbl[colidx[0,0]][0]

	def next(self, skip):

		h_steps = []

		try:
			h_rows = [next(re) for re in self.readers]
		except StopIteration:
			return 0

		id_to_trueclass = {}

		if int(h_rows[0][0]) % skip == 0:
			for row in h_rows:
				step = {'clock':0,
						'fb_id':'',
						'x':0, 'y':0,
						'theta':0,
						'v_left':0,
						'v_right':0,
						'true_class':'',
						'color':None,
						'world_model_long':{},
						'world_model_short':{},
						'rab_readings':[]}

				step['clock'] = int(row[0])
				step['fb_id'] = str(row[1])
				step['x'] = float(row[2].replace(",","."))
				step['y'] = float(row[3].replace(",","."))
				step['theta'] = float(row[4].replace(",","."))
				step['v_left'] = float(row[5].replace(",","."))
				step['v_right'] = float(row[6].replace(",","."))

				p = np.array([[step['x']],[step['y']]])
				if int(step["fb_id"][3:]) not in id_to_trueclass:
					for classlbl, data in self.map_ground_truth.items():
						for bv in data['poly_boundary_vertices']:
							if is_inside_square(p, bv):
								step['true_class'] = classlbl
								id_to_trueclass[int(step["fb_id"][3:])] = classlbl
								break
						if(step["true_class"] != ''):
							break
				else:
					step['true_class'] = id_to_trueclass[int(step["fb_id"][3:])]

				#step["true_class"] = self._is_inside(p)

				if step['true_class'] not in ['V','C','I','G','S']:
					print(p)
					print("WARNING: no true class. Skip this step.")
					continue

				step['color'] = self.map_ground_truth[step['true_class']]['color']

				step['world_model_long'] = {float(row[i].replace(",", ".")): {"distance": float(row[i + 1].replace(",", ".")),
																			  "occluded": bool(int(row[i + 2]))}
											for i in range(7, 120 * 3 + 7, 3)}

				#step['world_model_short'] = {float(row[i].replace(",",".")):float(row[i+1].replace(",",".")) for i in range(120*2+7,120*2+120*2+7,2)}

				if int(self.dataset_path.split("/")[-3]) > 1:
					split_rab = row[-1].split("_")
					split_rab.pop() # remove the last element because it's a ""
					for i in range(0,len(split_rab)-2,3):
						distance, bearing, id = float(split_rab[i]),float(split_rab[i+1]), int(split_rab[i+2])
						r_p_neigh = np.array([[(distance/100)*np.cos(bearing)],
											  [(distance/100)*np.sin(bearing)]])

						w_R_r = np.array([[np.cos(step["theta"]), -np.sin(step["theta"])],
										  [np.sin(step["theta"]), np.cos(step["theta"])]])
						w_p_neigh = p + w_R_r@r_p_neigh

						true_class = ""
						if id not in id_to_trueclass:
							for classlbl, data in self.map_ground_truth.items():
								for bv in data['poly_boundary_vertices']:
									if is_inside_square(w_p_neigh, bv):
										true_class = classlbl
										id_to_trueclass[id] = classlbl
										break
								if(true_class != ""):
									break
						else:
							true_class = id_to_trueclass[id]


						if true_class not in ['V','C','I','G','S']:
							print(p)
							print("WARNING: no true class. Skip this neighbor.")
							continue

						step['rab_readings'].append({"id":id,
													 "range":distance,
													 "bearing":bearing,
													 "r_x":r_p_neigh[0][0], "r_y":r_p_neigh[1][0],
													 "w_x":w_p_neigh[0][0], "w_y":w_p_neigh[1][0],
													 "true_class":true_class})
				h_steps.append(step)

		# uncomment se vuoi mettere il world_model_long dei neighbors (se ce ne sono) in rab_readings

		# neighbors_id = [[rr["id"] for rr in step['rab_readings']] for step in h_steps]
		# for n_idx, n_ids in enumerate(neighbors_id):
		# 	for rr_idx, n_id in enumerate(n_ids):
		# 		h_steps[n_idx]["rab_readings"][rr_idx]["world_model_long"] = h_steps[n_id]["world_model_long"]

		return h_steps

	def close(self):
		for f in self.files:
			f.close()

class Classifier:

	def __init__(self):

		with open("data/test/test_map_ground_truth.pickle","rb") as f:
			map_ground_truth = pickle.load(f)

		self.classlbl_to_template = {'I':[0,4,0,0],
									 'C':[0,0,0,2],
									 'V':[0,2,0,1],
									 'G':[0,1,2,0]}

		self.classlbl_to_id = {c:i for i,c in enumerate(list(map_ground_truth.keys()))}
		self.id_to_classlbl = {i:c for c,i in self.classlbl_to_id.items()}

	def preProcess(self, measurement, w_len):
		if isinstance(list(measurement.values())[0], float):
			d = list(measurement.values())
		else:
			d = [data["distance"] for data in list(measurement.values())]

		d_pre = [0.0]*len(d)

		for i in range(len(d)):
			avg = 0
			for j in range(w_len):
				avg = avg + d[(i+j) % len(d)]

			avg = avg/w_len

			d_pre[(i+int(w_len/2)) % len(d_pre)] = avg

		if isinstance(list(measurement.values())[0], float):
			return {float(a):d for a,d in zip(measurement.keys(), d_pre)}
		else:
			return {float(a):{"distance":d, "occluded":measurement[a]["occluded"]} for a,d in zip(measurement.keys(), d_pre)}

	def extractFeature(self, measurement):

		local_min_angles = []
		a = list(measurement.keys())
		d = list(measurement.values())

		for i in range(len(d)-4):
			ll = d[i]
			l = d[i+1]
			c = d[i+2]
			r = d[i+3]
			rr = d[i+4]

			if ll>l and l>c and c<r and r<rr:
				local_min_angles.append(a[i+2])


		n_intervals = 8
		aperture = to_radians(360.0/n_intervals)
		start = -np.pi + aperture/2
		interval_ids = []
		feature = [0]*4
		placed = False

		for lm_angle in local_min_angles:
			placed = False
			for i in range(n_intervals-1):
				if lm_angle > start + i*aperture and lm_angle <= start + (i+1)*aperture:
					interval_ids.append(i)
					placed = True

		# handling the last sector (sector id 7 range [180-aperture/2, -180+aperture/2])
		if not placed: interval_ids.append(n_intervals-1)

		# niente multiple interval_ids
		interval_ids = list(set(interval_ids))

		if(len(interval_ids) <= 1):
			return local_min_angles, feature

		for i in range(len(interval_ids)-1):
			distance_in_intervals = abs(interval_ids[i+1] - interval_ids[i])
			if(distance_in_intervals > n_intervals/2):
				distance_in_intervals = n_intervals - distance_in_intervals

			feature[distance_in_intervals-1]+=1


		# handling the last and first interval
		distance_in_intervals = abs(interval_ids[len(interval_ids)-1] - interval_ids[0])
		if(distance_in_intervals > n_intervals/2):
			distance_in_intervals = n_intervals - distance_in_intervals

		feature[distance_in_intervals-1]+=1


		return local_min_angles, feature

	def extractFeatureRaw(self, measurement):

		local_min_angles = []
		a = list(measurement.keys())
		d = list(measurement.values())

		for i in range(len(d)-4):
			ll = d[i]
			l = d[i+1]
			c = d[i+2]
			r = d[i+3]
			rr = d[i+4]

			if ll>l and l>c and c<r and r<rr:
				local_min_angles.append(i+2)

		n_intervals = 120
		feature = [0]*60
		placed = False

		if(len(local_min_angles) <= 1):
			return local_min_angles, feature

		for i in range(len(local_min_angles)-1):
			distance_in_indices = local_min_angles[i+1] - local_min_angles[i]
			if(distance_in_indices > n_intervals/2):
				distance_in_indices = n_intervals - distance_in_indices

			feature[distance_in_indices-1]+=1


		# handling the last and first interval
		distance_in_indices = abs(local_min_angles[-1] - local_min_angles[0])
		if(distance_in_indices > n_intervals/2):
			distance_in_indices = n_intervals - distance_in_indices

		feature[distance_in_indices-1]+=1


		return local_min_angles, feature

	def predict(self, feature):

		classlbls = list(self.classlbl_to_template.keys())
		templates = list(self.classlbl_to_template.values())

		norms = [norm(np.array(feature) - np.array(t)) for t in templates]

		argmin = np.argmin(np.array(norms))

		return classlbls[argmin]


	def classification_report_(self, y_true, y_pred, output_dict=False):
		return classification_report(y_true,
									 y_pred,
									 zero_division=0,
									 output_dict=output_dict,
									 labels=list(self.classlbl_to_id.keys()))

	def confusion_matrix_(self, y_true, y_pred):
		return confusion_matrix(y_true, y_pred, labels=list(self.classlbl_to_id.keys()))

class GaussianFilter:

	def __init__(self, classifier_type, feature_type, template_dataset=None):
		self.state_dim = 5
		self.classifier = Classifier()
		self.classifier_type = classifier_type
		self.feature_type = feature_type
		self.scaler = preprocessing.StandardScaler()

		if self.classifier_type not in ["linear", "quadratic"]:
			raise ValueError(f"Classifier cannot be {self.classifier_type}")

		if self.feature_type not in ["template", "geometricB", "geometricP"]:
			raise ValueError(f"Features cannot be {self.feature_type}")

		if self.feature_type == "template" and template_dataset==None:
			raise ValueError(f"Template dataset cannot be {template_dataset}")

		if self.feature_type == "template":
			self.templates = {}
			for step in template_dataset:
				z = self.classifier.preProcess(step['world_model_long'], 3)
				if step['true_class'] not in self.templates:
					self.templates[step['true_class']] = [{"pre_rotated":self.pre_compute_rotated_template(z),
														   "z":z}]
				else:
					self.templates[step['true_class']].append({"pre_rotated":self.pre_compute_rotated_template(z),
															   "z":z})

	@staticmethod
	def pre_compute_rotated_template(z):
		tm = []
		v1 = deque([data["distance"] for data in list(z.values())])

		for _ in range(len(v1)):
			tm.append(list(v1))
			v1.rotate(1)

		return np.array(tm)

	@staticmethod
	def min_distance(tm, z, handle_occlusion):
		z_distance = np.array([data["distance"] for data in z.values()])

		if handle_occlusion == True:
			z_occluded = [data["occluded"] for data in z.values()]
			z_mask = np.array(z_occluded)
			z_sliced = z_distance[~z_mask]

			tm_sliced = tm[:,np.where(~z_mask == True)[0].tolist()]
			tm_sliced_unique = np.unique(tm_sliced, axis=0)
			tm = tm_sliced_unique-z_sliced

		else:
			tm = tm - z_distance


		mins = np.diag(tm@tm.T)
		return np.min(mins)

	@staticmethod
	def naive_min_distance(tm, z, handle_occlusion):
		tm_distance = np.array([data["distance"] for data in tm.values()])
		z_distance = deque([data["distance"] for data in z.values()])
		z_data = deque(list(z.values()))
		convolutions = []
		
		if handle_occlusion == True:
			z_data_rotations = []
			for _ in range(len(z_data)):
				z_data_rotations.append(list(z_data))
				z_data.rotate(1)

			for z_data_rotated in z_data_rotations:
				filtered_z_distance = []
				filtered_tm_distance = []
				for i in range(len(z_data_rotated)):
					if z_data_rotated[i]["occluded"] == False:
						filtered_z_distance.append(z_data_rotated[i]["distance"])
						filtered_tm_distance.append(tm_distance[i])

				diff = np.array(filtered_z_distance) - np.array(filtered_tm_distance)
				convolutions.append(np.inner(diff,diff))
				
		else:
			z_distance_rotations = []
			for _ in range(len(z_distance)):
				z_distance_rotations.append(list(z_distance))
				z_distance.rotate(1)

			for z_distance_rotated in z_distance_rotations:
				diff = np.array(z_distance_rotated) - tm_distance
				convolutions.append(np.inner(diff,diff))

		return min(convolutions)

	def estimateTransitionModel(self, dataset, unique=False):

		# counts start from 1 (add-one smoothing)
		joint = np.ones((self.state_dim, self.state_dim))

		if unique:
			dataset_unique = []
			dataset_unique.append(dataset[0])
			for i in range(1,len(dataset),1):
				if dataset[i]["true_class"] != dataset_unique[-1]["true_class"]:
					dataset_unique.append(dataset[i])

			# learn from all steps
			for i in range(len(dataset_unique)-1):
				classid = self.classifier.classlbl_to_id[dataset_unique[i]['true_class']]
				xt_1 = classid

				classid = self.classifier.classlbl_to_id[dataset_unique[i+1]['true_class']]
				xt = classid

				joint[xt,xt_1] += 1.0

		else:
			# learn from %10 steps
			for i in range(len(dataset)-10):
				classid = self.classifier.classlbl_to_id[dataset[i]['true_class']]
				xt_1 = classid

				classid = self.classifier.classlbl_to_id[dataset[i+10]['true_class']]
				xt = classid

				joint[xt,xt_1] += 1.0

		self.transition_model = joint/np.sum(joint, axis=0)

	def estimateObservationModel(self, dataset=None, save_model=True, load_model=True, scale=True):

		if load_model == False and dataset == None:
			raise ValueError(f"Dataset cannot be {dataset}")

		self.parameters = {}

		if load_model == True:
			with open(f"data/train/observation_model_{self.classifier_type}_{self.feature_type}.pickle","rb") as f:
				self.clf = pickle.load(f)
			with open(f"data/train/scaler_{self.feature_type}.pickle","rb") as f:
				self.scaler = pickle.load(f)
		else:
			X_train,y_train = [],[]
			for step in tqdm(dataset):
				z = self.classifier.preProcess(step['world_model_long'],3)
				X_train.append(self.extractFeature(z, handle_occlusion=False)) #perché nel train non ci sono occlusioni (robot=1)
				y_train.append(self.classifier.classlbl_to_id[step['true_class']])

			if scale:
				#------- Standardization (mean=0, variance=1) ------
				self.scaler.fit(X_train)
				X_train = self.scaler.transform(X_train)
				#---------------------------------------------------
			self.clf = None
			if self.classifier_type == "linear":
				self.clf = LinearDiscriminantAnalysis(store_covariance=True)
			elif self.classifier_type == "quadratic":
				self.clf = QuadraticDiscriminantAnalysis(store_covariance=True)

			self.clf.fit(X_train, y_train)
			if self.classifier_type == "linear":
				self.clf.intercept_ -= np.log(self.clf.priors_)

			if save_model:
				with open(f"data/train/observation_model_{self.classifier_type}_{self.feature_type}.pickle",'wb') as f:
					pickle.dump(self.clf, f)
				with open(f"data/train/scaler_{self.feature_type}.pickle",'wb') as f:
					pickle.dump(self.scaler, f)

		# if self.classifier_type == "linear":
		# 	for i in range(self.clf.means_.shape[0]):
		# 		self.parameters[self.classifier.id_to_classlbl[i]] = {'mu':self.clf.means_[i,:].reshape((-1,1)),
		# 															  'sigma_inv':np.linalg.inv(self.clf.covariance_),
		# 															  'sigma':self.clf.covariance_}
		# elif self.classifier_type == "quadratic":
		# 	for i in range(self.clf.means_.shape[0]):
		# 		self.parameters[self.classifier.id_to_classlbl[i]] = {'mu':self.clf.means_[i,:].reshape((-1,1)),
		# 															  'sigma_inv':np.linalg.pinv(self.clf.covariance_[i]),
		# 															  'sigma':self.clf.covariance_[i]}

	def extractFeature(self, measurement, handle_occlusion):
		if self.feature_type == "template":
			return self.extractFeatureTemplate(measurement, handle_occlusion)
		elif self.feature_type == "geometricB":
			return self.extractFeatureGeometricB(measurement, handle_occlusion)
		elif self.feature_type == "geometricP":
			return self.extractFeatureGeometricP(measurement)

	def extractFeatureGeometricB(self, measurement, handle_occlusion):

		Point = namedtuple('Point', 'x y')

		if handle_occlusion == True:
			v = [Point(data["distance"]*np.cos(a),data["distance"]*np.sin(a))
				 for a,data in measurement.items() if data["occluded"] == False]
		else:
			v = [Point(data["distance"]*np.cos(a),data["distance"]*np.sin(a))
				 for a,data in measurement.items()]


		v.append(v[0])

		# B.1 and B.2
		# (shoelace formula)
		A, P = 0.0, 0.0
		for i in range(len(v)-1):
			A += v[i].x*v[i+1].y - v[i].y*v[i+1].x
			P += np.linalg.norm(np.array(v[i+1])-np.array(v[i]))
		B1, B2 = A, P

		# B.3
		B3 = A/P

		# B.4
		cx, cy = 0.0, 0.0
		for i in range(len(v)-1):
			m = (v[i].x*v[i+1].y - v[i+1].x*v[i].y)
			cx += (v[i].x + v[i+1].x) * m
			cy += (v[i].y + v[i+1].y) * m

		c = np.array(Point(cx/6*A, cy/6*A))
		avg_shape = 0.0
		#d = [np.linalg.norm(np.array(v[i]) - c) for i in range(len(v)-1)]
		for i in range(len(v)-1):
			avg_shape += np.linalg.norm(np.array(v[i]) - c)#/max(d)

		avg_shape /= (len(v)-1)
		B4 = avg_shape

		# B.5
		std_shape = 0.0
		#d = [np.power(np.linalg.norm(np.array(v[i]) - c) - B4, 2)]
		for i in range(len(v)-1):
			std_shape += np.power(np.linalg.norm(np.array(v[i]) - c) - B4, 2)

		std_shape /= (len(v)-1)
		#std_shape /= max(d)
		std_shape  = np.sqrt(std_shape)
		B5 = std_shape

		# B.7 B.8
		# t = 0
		# ds_s, t_s = [],[]
		# for i in range(len(v)-1):
		# 	dv = (v[i+1].x + v[i+1].y * 1j) - (v[i].x + v[i].y * 1j)
		# 	ds = dv / abs(dv)
		# 	t += abs(dv)
		# 	ds_s.append(ds)
		# 	t_s.append(t)
		#
		# c=0
		# for i in range(len(ds_s)):
		# 	c += (ds_s[i+1] - ds[i])*np.exp(-1j*np.pi*(2*np.pi/P)*t[i])
		#
		# coeff = P/(2*np.pi)**2
		# c_1 = coeff*c
		# c__1 = coeff

		# Perimeter and Area convex_hull
		# hull_indices = ConvexHull(poly_v[:-1]).vertices
		# hull_v = [poly_v[index] for index in hull_indices]
		# hull_v.append(hull_v[0])
		#
		# convex_A = 0.0
		# convex_P = 0.0
		# for i in range(len(hull_v)-1):
		# 	convex_A += hull_v[i][0]*hull_v[i+1][1] - hull_v[i][1]*hull_v[i+1][0]
		# 	convex_P += np.linalg.norm(np.array(hull_v[i+1])-np.array(hull_v[i]))

		#Max and min Feret's diameters

		# B.7 B.8 simply
		# diameters = []
		# if handle_occlusion == True:
		# 	d = [data["distance"] for data in list(measurement.values()) if data["occluded"] == False]
		# else:
		# 	d = [data["distance"] for data in list(measurement.values())]
		#
		# for i in range(len(d) - 60):
		# 	diameters.append(d[i]+d[i+59])
		#
		# B7 = max(diameters)
		# B8 = min(diameters)

		return [B1, B2, B3, B4, B5] #B7, B8]

	def extractFeatureGeometricP(self, measurement):
		beam_len = list(measurement.values())
		beam_len.append(beam_len[0])

		# P.1 P.2
		diffs = [abs(beam_len[i]-beam_len[i+1]) for i in range(len(beam_len)-1)]
		P1, P2 = np.mean(diffs), np.std(diffs)

		# P.5 P.6
		P5, P6 = np.mean(beam_len), np.std(beam_len)

		# P.7
		# th = [20*(i+1) for i in range(int(140//20))]
		# n_gaps = [1.0]*len(th)
		# for i,t in enumerate(th):
		# 	for di in diffs:
		# 		if di > t:
		# 			n_gaps[i] += 1

		# P.11 P.12
		rel = [beam_len[i]/beam_len[i+1] for i in range(len(beam_len)-1)]
		P11, P12 = np.mean(rel), np.std(rel)

		# P.13 P14
		beam_len_norm = [bl/max(beam_len) for bl in beam_len]
		P13, P14 = np.mean(beam_len_norm), np.std(beam_len_norm)

		# P.16
		kurtosis =  0.0
		for i in range(len(beam_len)-1):
			kurtosis += np.power(beam_len[i] - P5,4)
		kurtosis /= (len(beam_len)-1)*P6
		P16 = kurtosis - 3

		return [P1, P2, P5, P6, P11, P12, P13, P14, P16]

	def extractFeatureTemplate(self, measurement, handle_occlusion):


		# def minDistance(value1, value2):
		# 	tm = []
		# 	v1 = deque(value1)
		#
		# 	for _ in range(120):
		# 		v1.rotate(1)
		# 		tm.append(list(v1))
		#
		# 	tm = np.array(tm)
		# 	value2 = np.array(value2)
		# 	tm = tm-value2
		# 	mins = np.diag(tm@tm.T)
		# 	return np.min(mins)#/(120*150**2)



		# feature = []
		# for class_lbl,templates_readings in self.templates.items():
		# 	avg_min_distance = []
		# 	for template_reading in templates_readings:
		# 		avg_min_distance.append(minDistance(template_reading["pre_rotated"],list(measurement.values())))
		#
		# 	#print(class_lbl)
		# 	#print(avg_min_distance)
		#
		# 	avg_min_distance = sum(avg_min_distance)/len(avg_min_distance)
		# 	#avg_min_distance = min(avg_min_distance)
		# 	feature.append(avg_min_distance)

		feature = []
		for class_lbl,templates_readings in self.templates.items():
			avg_min_distance = []
			for template_reading in templates_readings:
				avg_min_distance.append(self.naive_min_distance(template_reading["z"],
														  measurement,
														  handle_occlusion))


			#print(class_lbl)
			#print(avg_min_distance)

			#avg_min_distance = sum(avg_min_distance)/len(avg_min_distance)
			#avg_min_distance = min(avg_min_distance)
			feature += avg_min_distance

		return feature

	def gaussian(self, x, mu, sig):
		mahalanobis = (x-mu).T@sig@(x-mu)
		pdf = 0.0
		try:
			pdf = np.exp(-0.5*mahalanobis) + 0.01
		except RuntimeWarning:
			print(mahalanobis)
			print(pdf)

		return pdf[0][0], mahalanobis

	def update(self, belief, feature, weight=None, log=False):
		#predict step
		bt_t_1 = [0.0]*self.state_dim
		for i in range(self.state_dim):
			for j in range(self.state_dim):
				bt_t_1[j] += self.transition_model[j, i]*belief[i]

		pdfs,mahalas = [],[]
		if not log:
			#update step
			btt = [0.0]*self.state_dim
			den = 0.0
			pdfs,mahalas = [],[]
			for i in range(self.state_dim):
				pdf, mahalanobis = self.gaussian(np.array(feature).reshape((-1,1)),
												 self.parameters[self.classifier.id_to_classlbl[i]]['mu'],
												 self.parameters[self.classifier.id_to_classlbl[i]]['sigma_inv'])
				pdfs.append(pdf)
				mahalas.append(mahalanobis)
				btt[i] = pdf*bt_t_1[i]
				den+=btt[i]

			try:
				btt = [b/den for b in btt]
			except RuntimeWarning:
				print(belief)
				print(feature)
				print(bt_t_1)
				print(mahalas)
				print(pdfs)
				print(den)
				print(btt)

		else:

			# preso da sklearn
			x = np.array(feature).reshape((-1,1))
			log_prior = np.log(np.array(bt_t_1).reshape((-1,1)))
			#(4,1) = (4,9)x(9,1) + (4,1) + (4,1)
			if weight is None:
				decision_function = self.clf.coef_@x + self.clf.intercept_.reshape((-1,1)) + log_prior
			else:
				if weight <= 0.5:
					decision_function = self.clf.coef_@x + self.clf.intercept_.reshape((-1,1)) + log_prior
				else:
					decision_function = (1-weight)*(self.clf.coef_@x + self.clf.intercept_.reshape((-1,1))) \
										+ (weight*log_prior)
			#softmax
			btt = np.exp(decision_function)/np.sum(np.exp(decision_function))

		if isinstance(btt, list):
			btt = btt
		else:
			btt = btt.reshape((1,-1)).tolist()[0]

		return btt

	def predict(self, belief):
		argmax = np.argmax(np.array(belief))
		return self.classifier.id_to_classlbl[argmax]

class HybridFilter:

	def __init__(self):
		self.state_dim = 5
		self.classifier = Classifier()
		# self.topology_model = np.array([[0.45, 0.2, 0.033, 0.033, 0.033],
		# 								[0.45, 0.2, 0.45, 0.45, 0.45],
		# 								[0.033, 0.2, 0.45, 0.033, 0.033],
		# 								[0.033, 0.2, 0.033, 0.45, 0.033],
		# 								[0.033, 0.2, 0.033, 0.033, 0.45]])

		# self.topology_model = np.array([[0.55, 0.1125, 0.033, 0.033, 0.033],
		# 								[0.35, 0.55, 0.35, 0.35, 0.35],
		# 								[0.033, 0.1125, 0.55, 0.033, 0.033],
		# 								[0.033, 0.1125, 0.033, 0.55, 0.033],
		# 								[0.033, 0.1125, 0.033, 0.033, 0.55]])

		# self.topology_model = np.array([[0.85, 0.0375, 0.0375, 0.0375, 0.0375],
		# 								[0.0375, 0.85, 0.0375, 0.0375, 0.0375],
		# 								[0.0375, 0.0375, 0.85, 0.0375, 0.0375],
		# 								[0.0375, 0.0375, 0.0375, 0.85, 0.0375],
		# 								[0.0375, 0.0375, 0.0375, 0.0375, 0.85]])

		self.topology_model = np.array([[0.99, 0.0025, 0.0025, 0.0025, 0.0025],
										[0.0025, 0.99, 0.0025, 0.0025, 0.0025],
										[0.0025, 0.0025, 0.99, 0.0025, 0.0025],
										[0.0025, 0.0025, 0.0025, 0.99, 0.0025],
										[0.0025, 0.0025, 0.0025, 0.0025, 0.99]])

		self.topology_model = np.eye(5)*(1*10**20) + np.ones((5,5))
		self.topology_model /=np.sum(self.topology_model, axis=0)

	def estimateTransitionModel(self, dataset=None):
		"""
		P( Xr@t | Xr@t-1 )
		"""

		if dataset is not None:
			# counts start from 1 (add-one smoothing)
			joint = np.ones((self.state_dim, self.state_dim))

			# learn from %10 steps
			for i in range(len(dataset)-10):
				classid = self.classifier.classlbl_to_id[dataset[i]['true_class']]
				xt_1 = classid

				classid = self.classifier.classlbl_to_id[dataset[i+10]['true_class']]
				xt = classid

				joint[xt,xt_1] += 1.0

			self.transition_model = joint/np.sum(joint, axis=0)

			with open(f"data/train/transition_model_svc.pickle",'wb') as f:
				pickle.dump(self.transition_model, f)
		else:
			with open(f"data/train/transition_model_svc.pickle","rb") as f:
				self.transition_model = pickle.load(f)

		# self.transition_model = np.array([[0.45,0.2,0.033,0.033,0.033],
		# 								  [0.45,0.2,0.45,0.45,0.45],
		# 								  [0.033,0.2,0.45,0.033,0.033],
		# 								  [0.033,0.2,0.033,0.45,0.033],
		# 								  [0.033,0.2,0.033,0.033,0.45]])

	def estimateObservationModel(self, dataset):
		"""
		P( Z | Xr )
		"""

		X_train, y_train = [],[]

		if dataset is not None:
			for step in tqdm(dataset):
				z = self.classifier.preProcess(step["world_model_long"], 3)
				X_train.append(self.extractFeature(z, handle_occlusion=False))
				y_train.append(self.classifier.classlbl_to_id[step['true_class']])

			X_train = np.array(X_train)
			print(X_train.shape)

			self.scaler = preprocessing.StandardScaler()
			self.scaler.fit(X_train)
			X_train = self.scaler.transform(X_train)

			self.clf = SVC(probability=True)
			self.clf.fit(X_train,y_train)

			self.P_c = np.zeros(len(self.clf.classes_))
			count = Counter(y_train)
			for i,l in enumerate(self.clf.classes_):
				self.P_c[i] = count[l] / len(y_train)

			with open(f"data/train/P_c.pickle",'wb') as f:
				pickle.dump(self.P_c, f)
			with open(f"data/train/observation_model_svc.pickle",'wb') as f:
				pickle.dump(self.clf, f)
			with open(f"data/train/scaler_svc.pickle",'wb') as f:
				pickle.dump(self.scaler, f)
		else:
			with open(f"data/train/P_c.pickle","rb") as f:
				self.P_c = pickle.load(f)
			with open(f"data/train/observation_model_svc.pickle","rb") as f:
				self.clf = pickle.load(f)
			with open(f"data/train/scaler_svc.pickle","rb") as f:
				self.scaler = pickle.load(f)

	def estimateSpatialModel(self, dataset):
		"""
		P( D,Xn | Xr )
		"""
		distance_th = 200
		bin_width = 10
		self.edges = np.array(range(0,distance_th+bin_width, bin_width))
		k = 1000

		if dataset is not None:
			# counts start from 1 (add-one smoothing)
			joint = np.ones((len(self.edges), self.state_dim, self.state_dim))

			per_class = {"G": [],
						 "C": [],
						 "S": [],
						 "V": [],
						 "I": []}

			for step in dataset:
				per_class[step["true_class"]].append([step["x"],
													  step["y"],
													  step["theta"]])

			for class_r in list(per_class.keys()):
				for class_n in list(per_class.keys()):
					sample_r = random.sample(per_class[class_r], k=k)
					sample_n = random.sample(per_class[class_n], k=k)

					ranges = []
					for r in sample_r:
						for n in sample_n:
							position_r = np.array([[r[0]],[r[1]]])
							position_n = np.array([[n[0]],[n[1]]])
							range_d = norm(position_n-position_r)*100
							#theta = r[2]
							#a = get_alpha(theta, position1, position2)
							if range_d >0.0 and range_d <= distance_th:
								ranges.append(range_d)
							#alpha.append(a)

					counts,_ = np.histogram(ranges, bins=self.edges)
					joint[:,self.classifier.classlbl_to_id[class_r],
					self.classifier.classlbl_to_id[class_n]] += np.insert(counts,0,0)

			self.spatial_model = joint/np.sum(joint, axis=0, keepdims=True)

			with open(f"data/train/spatial_model_svc.pickle",'wb') as f:
				pickle.dump(self.spatial_model, f)

			with open(f"data/train/edges.pickle",'wb') as f:
				pickle.dump(self.edges, f)
		else:
			with open(f"data/train/spatial_model_svc.pickle","rb") as f:
				self.spatial_model = pickle.load(f)

			with open(f"data/train/edges.pickle","rb") as f:
				self.edges = pickle.load(f)

	def estimateHullModel(self, dataset):
		"""
		P( MAX_ALPHA | Xr )
		"""
		range_min, range_max = 200, 300
		bin_width = to_radians(10)
		self.hull_edges = np.array(np.arange(0,(2*np.pi)+bin_width, bin_width))
		n_sample_max = 1000

		if dataset is not None:
			# counts start from 1 (add-one smoothing)
			joint = np.ones((len(self.hull_edges), self.state_dim))

			class_to_querypointsindices = {"G": [],
										   "C": [],
										   "S": [],
										   "V": [],
										   "I": []}

			# build the balltree out of the whole dataset
			tree = BallTree([np.array([step["x"], step["y"]]) for step in dataset])

			# sample n random query-points per class (using indices)
			n_sample = 0
			while n_sample < self.state_dim*n_sample_max:
				random_index = random.randint(len(dataset))
				if len(class_to_querypointsindices[dataset[random_index]["true_class"]]) < n_sample_max:
					class_to_querypointsindices[dataset[random_index]["true_class"]].append(random_index)
					n_sample+=1

			# query the balltree for each query-point
			# compute and store the query-point neighbors position and bearing
			# if their range is between range_min and range_max
			neighbors_data = []
			for k,v in class_to_querypointsindices.items():
				query_points = [np.array([dataset[index]["x"], dataset[index]["y"]]) for index in v]
				ind, dist = tree.query_radius(query_points,
											  r=range_max,
											  return_distance=True)
				max_alphas = []
				for i in range(n_sample_max):
					qp = query_points[i]
					for index, distance in zip(ind[i],dist[i]):
						if distance >= range_min:
							position_r = qp
							theta_r = dataset[index]["theta"]
							position_n = np.array([[dataset[index]["x"]],[dataset[index]["y"]]])
							bearing = get_bearing(position_r, position_n, theta_r)
							neighbors_data.append([position_n, distance, bearing])

					# compute the convex-hull of the neighbors of the current query-point
					neighbors_hull_indices = ConvexHull([nd[0] for nd in neighbors_data]).vertices # cc order

					# compute the normalized bearing the convex-hull of the neighbors and select the max
					convex_bearings = [neighbors_data[i][2] for i in neighbors_hull_indices]
					convex_bearings.sort(reverse=False) # ascending order
					convex_bearings_distances = [convex_bearings[i+1]-convex_bearings[i] for i in range(len(convex_bearings)-1)]
					max_alpha = max(convex_bearings_distances)
					max_alphas.append(max_alpha)

				counts,_ = np.histogram(max_alphas, bins=self.edges)
				joint[:,self.classifier.classlbl_to_id[k]] += np.insert(counts,0,0)

			self.hull_model = joint/np.sum(joint, axis=0)

			with open(f"data/train/hull_model_svc.pickle",'wb') as f:
				pickle.dump(self.hull_model, f)

			with open(f"data/train/hull_edges.pickle",'wb') as f:
				pickle.dump(self.hull_edges, f)
		else:
			with open(f"data/train/hull_model_svc.pickle","rb") as f:
				self.hull_model = pickle.load(f)

			with open(f"data/train/hull_edges.pickle","rb") as f:
				self.hull_edges = pickle.load(f)

	def extractFeature(self, measurement, handle_occlusion=False):
		Point = namedtuple('Point', 'x y')

		if handle_occlusion == True:
			v = [Point(data["distance"]*np.cos(a),data["distance"]*np.sin(a))
				 for a,data in measurement.items() if data["occluded"] == False]
		else:
			# v = [Point(data["distance"]*np.cos(a),data["distance"]*np.sin(a))
			# 	 for a,data in measurement.items()]
			#
			# v.append(v[0])

			vv = [[a, data["distance"]] for a,data in measurement.items()]
			vv.append(vv[0])

			polar = np.array(vv, dtype=np.float)
			cartesian = np.zeros_like(polar)
			cartesian[:,0] = polar[:,1]*np.cos(polar[:,0])
			cartesian[:,1] = polar[:,1]*np.sin(polar[:,0])


		current_idx = list(range(len(cartesian)-1))
		next_idx = list(range(1,len(cartesian),1))

		# B.1 and B.2
		# (shoelace formula)
		# A, P = 0.0, 0.0
		# for i in range(len(v)-1):
		# 	A += v[i].x*v[i+1].y - v[i].y*v[i+1].x
		# 	P += np.linalg.norm(np.array(v[i+1])-np.array(v[i]))
		# B1, B2 = A, P

		A = np.sum(cartesian[:,0][current_idx]*cartesian[:,1][next_idx] - \
			 cartesian[:,1][current_idx]*cartesian[:,0][next_idx])
		P = np.sum(np.sqrt(np.sum((cartesian[next_idx]-cartesian[current_idx])**2, axis=1)))
		B1, B2 = A, P

		# B.3
		B3 = A/P

		# B.4
		# cx, cy = 0.0, 0.0
		# for i in range(len(v)-1):
		# 	m = (v[i].x*v[i+1].y - v[i+1].x*v[i].y)
		# 	cx += (v[i].x + v[i+1].x) * m
		# 	cy += (v[i].y + v[i+1].y) * m
		#
		# c = np.array(Point(cx/6*A, cy/6*A))
		# avg_shape = 0.0
		# d = [np.linalg.norm(np.array(v[i]) - c) for i in range(len(v)-1)]
		# for i in range(len(v)-1):
		# 	avg_shape += np.linalg.norm(np.array(v[i]) - c)#/max(d)
		#
		# avg_shape /= (len(v)-1)
		# B4 = avg_shape

		m = cartesian[:,0][current_idx]*cartesian[:,1][next_idx] - cartesian[:,1][current_idx]*cartesian[:,0][next_idx]
		cx = np.sum((cartesian[:,0][current_idx] + cartesian[:,0][next_idx]) * m)
		cy = np.sum((cartesian[:,1][current_idx] + cartesian[:,1][next_idx]) * m)
		c = np.array([[A*(cx/6), A*(cy/6)]])
		temp = np.sqrt(np.sum((cartesian-c)**2, axis=1))
		avg_shape = np.mean(temp)
		B4 = avg_shape


		# B.5
		# std_shape = 0.0
		# d = [np.power(np.linalg.norm(np.array(v[i]) - c) - B4, 2)]
		# for i in range(len(v)-1):
		# 	std_shape += np.power(np.linalg.norm(np.array(v[i]) - c) - B4, 2)
		#
		# std_shape /= (len(v)-1)
		# #std_shape /= max(d)
		# std_shape  = np.sqrt(std_shape)
		# B5 = std_shape
		B5 = np.std(temp)


		# B.7 B.8
		# t = 0
		# ds_s, t_s = [],[]
		# for i in range(len(v)-1):
		# 	dv = (v[i+1].x + v[i+1].y * 1j) - (v[i].x + v[i].y * 1j)
		# 	ds = dv / abs(dv)
		# 	t += abs(dv)
		# 	ds_s.append(ds)
		# 	t_s.append(t)
		#
		# c=0
		# for i in range(len(ds_s)):
		# 	c += (ds_s[i+1] - ds[i])*np.exp(-1j*np.pi*(2*np.pi/P)*t[i])
		#
		# coeff = P/(2*np.pi)**2
		# c_1 = coeff*c
		# c__1 = coeff

		# Perimeter and Area convex_hull
		# hull_indices = ConvexHull(poly_v[:-1]).vertices
		# hull_v = [poly_v[index] for index in hull_indices]
		# hull_v.append(hull_v[0])
		#
		# convex_A = 0.0
		# convex_P = 0.0
		# for i in range(len(hull_v)-1):
		# 	convex_A += hull_v[i][0]*hull_v[i+1][1] - hull_v[i][1]*hull_v[i+1][0]
		# 	convex_P += np.linalg.norm(np.array(hull_v[i+1])-np.array(hull_v[i]))

		#Max and min Feret's diameters

		# B.7 B.8 simply
		# diameters = []
		# if handle_occlusion == True:
		# 	d = [data["distance"] for data in list(measurement.values()) if data["occluded"] == False]
		# else:
		# 	d = [data["distance"] for data in list(measurement.values())]
		#
		# for i in range(len(d) - 60):
		# 	diameters.append(d[i]+d[i+59])
		#
		# B7 = max(diameters)
		# B8 = min(diameters)

		return [B1, B2, B3, B4, B5]# B7, B8]

	def predict(self, belief):
		if type(belief) == list:
			argmax = np.argmax(np.array(belief))
		else:
			argmax = np.argmax(belief).flatten()[0]
		return self.classifier.id_to_classlbl[argmax]

	def update_instantaneous(self, h_step, r_id):
		P_c_x_debug = self.clf.predict_proba(h_step[r_id]["feature_std"]).reshape((-1,1)) # (5,1)
		return P_c_x_debug

	def update(self, belief, h_step, r_id):
		# # predict step
		# bt_t_1 = [0.0]*self.state_dim
		# for i in range(self.state_dim):
		# 	for j in range(self.state_dim):
		# 		bt_t_1[j] += self.transition_model[j, i]*belief[i]
		#
		# # update step
		# log_prior = np.log(np.array(bt_t_1).reshape((-1,1)))
		# #########
		# P_c_x_debug = self.clf.predict_proba(feature).reshape((-1,1)) # (5,1)
		# P_x_c = P_c_x_debug / np.array(self.P_c).reshape((-1,1))
		# #########
		# # log_likelihood = np.log(P_x_c)
		# P_c_x = self.clf.predict_log_proba(feature).reshape((-1,1)) # (5,1)
		# log_likelihood = P_c_x - np.log(np.array(self.P_c).reshape((-1,1)))
		#
		# log_posterior = log_likelihood + log_prior
		#
		# #softmax
		# btt = np.exp(log_posterior)/np.sum(np.exp(log_posterior))

		Z = self.clf.predict_proba(h_step[r_id]["feature_std"]).reshape((-1,1))/ \
			np.array(self.P_c, dtype=float).reshape((-1,1))

		belief = (self.transition_model@belief)*(Z/np.sum(Z))
		belief = belief/np.sum(belief)

		return belief

	def update_lag(self, belief, h_step, h_step_future, r_id):
		Z = self.clf.predict_proba(h_step[r_id]["feature_std"]).reshape((-1,1))/ \
			np.array(self.P_c, dtype=float).reshape((-1,1))
		belief = (self.transition_model@belief)*(Z/np.sum(Z))
		belief = belief/np.sum(belief)

		beta_t = np.ones((5,1))
		for step in reversed(h_step_future):
			Z = self.clf.predict_proba(step[r_id]["feature_std"]).reshape((-1,1))/ \
				np.array(self.P_c, dtype=float).reshape((-1,1))
			beta_t = self.transition_model.T@(beta_t*(Z/np.sum(Z)))
			beta_t = beta_t/np.sum(beta_t)

		return belief, (belief*beta_t)/np.sum((belief*beta_t)), beta_t

	def _construct_local_graph(self, h_step, r_id, range_th):
		self.graph = {}

		f = [r_id]

		while len(f) != 0:
			curr_n = f.pop(0)
			if curr_n == r_id:
				adjacent_nodes = [rr["id"] for rr in h_step[curr_n]["rab_readings"]
								  if rr["range"] <= range_th]
			else:
				adjacent_nodes = [r_id]
			masks = np.ones((len(adjacent_nodes)+1,self.state_dim,len(adjacent_nodes)+1), dtype=int)
			for i in range(masks.shape[0]):
				masks[i,:,i] = 0

			self.graph[curr_n] = {"adj":adjacent_nodes+[-1],
								  "msgs":np.ones((self.state_dim, len(adjacent_nodes)+1)),
								  "masks":masks}

			f += [adj for adj in adjacent_nodes if adj not in self.graph and adj not in f]

	def update_neighbors(self,  filter_beliefs, h_step, r_id, range_th, n_iteration=2):
		w = 1000

		for i in range(len(h_step)):
			if i == r_id:
				Z = self.clf.predict_proba(h_step[i]["feature_std"]).reshape((-1,1))/ \
					 np.array(self.P_c).reshape((-1,1)) # (5,1)
				# Z = np.ones((5,1))
				# class_idx = self.classifier.classlbl_to_id[h_step[i]["true_class"]]
				# Z[class_idx] = w
			else:
				Z = np.ones((5,1))
				class_idx = self.classifier.classlbl_to_id[h_step[i]["true_class"]]
				Z[class_idx] = w
				#print(Z)

			filter_beliefs[i] = (self.transition_model@filter_beliefs[i]) * (Z/np.sum(Z))
			filter_beliefs[i] /= np.sum(filter_beliefs[i])
			h_step[i]["filter_belief"] = filter_beliefs[i]

		# total_belief = np.ones((5,1))
		# for n in h_step[r_id]["rab_readings"]:
		# 	if n["range"] <= range_th:
		# 		D = bisect_left(self.edges, n["range"])
		# 		#msg = self.spatial_model[D,:,:]@filter_beliefs[n["id"]]
		# 		msg = self.topology_model@filter_beliefs[n["id"]]
		# 		n["msg"] = (msg/np.sum(msg))
		# 		total_belief = total_belief*(msg/np.sum(msg))
		#
		# total_belief = total_belief*filter_beliefs[r_id]
		# total_belief /= np.sum(total_belief)
		# filter_beliefs[r_id] = total_belief
		# h_step[r_id]["total_belief"] = total_belief

		self._construct_local_graph(h_step, r_id, range_th)
		h_step[r_id]["graph"] = self.graph

		if len(self.graph) > 1:

			#pprint(filter_beliefs)

			for robot_id, data in self.graph.items():
				data["msgs"][:,len(data["adj"])-1] = filter_beliefs[robot_id].flatten()

			#pprint(self.graph)
			#old_total_belief = np.ones((5,1))
			for k in range(n_iteration):

				for robot_id, data in self.graph.items():

					masked_msgs = data["msgs"]*data["masks"]
					new_msgs = np.product(np.where(masked_msgs==0, 1, masked_msgs), axis=2).T
					new_msgs = new_msgs/np.sum(new_msgs, axis=0)
					new_msgs = self.topology_model.T@new_msgs

					for i in range(len(data["adj"])-1):
						receiver_id = data["adj"][i]
						r_idx_in_receiver_adj = self.graph[receiver_id]["adj"].index(robot_id)
						self.graph[receiver_id]["msgs"][:,r_idx_in_receiver_adj] = new_msgs[:,i]

			#total_belief = np.product(self.graph[r_id]["msgs"], axis=1)
			#total_belief /= np.sum(total_belief)

			#print(k, np.linalg.norm(old_total_belief-total_belief))
			#old_total_belief = total_belief
			#pprint(self.graph)

			total_belief = np.product(self.graph[r_id]["msgs"], axis=1, keepdims=True)
			total_belief /= np.sum(total_belief)
			filter_beliefs[r_id] = total_belief

		else:
			total_belief = filter_beliefs[r_id]

		h_step[r_id]["total_belief"] = total_belief

		return total_belief

	def _construct_graph(self, h_step, r_id, range_th):
		graph = {"connectivity":{},
				 "edges":[]}

		f = [r_id]

		while len(f) != 0:
			curr_n = f.pop(0)
			adjacent_nodes = [rr["id"] for rr in h_step[curr_n]["rab_readings"] if rr["range"] <= range_th]
			masks = np.ones((len(adjacent_nodes) + 1, self.state_dim, len(adjacent_nodes) + 1), dtype=int)
			for i in range(masks.shape[0]):
				masks[i, :, i] = 0

			graph["connectivity"][curr_n] = {"adj": adjacent_nodes + [-1],
											 "msgs": np.ones((self.state_dim, len(adjacent_nodes) + 1)),
											 "masks": masks}

			for adj in adjacent_nodes:
				if (curr_n, adj) not in graph["edges"] and (adj, curr_n) not in graph["edges"]:
					graph["edges"].append((curr_n, adj))

			f += [adj for adj in adjacent_nodes if adj not in graph["connectivity"] and adj not in f]

		return graph

	def update_loop(self, filter_beliefs, h_step, spy_robot_id, range_th, n_iteration):

		w = 1000

		# for all robots update the filter belief
		for robot_id in range(len(h_step)):
			if robot_id == spy_robot_id:
				Z = self.clf.predict_proba(h_step[robot_id]["feature_std"]).reshape((-1,1))/ \
					np.array(self.P_c).reshape((-1,1)) # (5,1)
				# Z = np.ones((5,1))
				# class_idx = self.classifier.classlbl_to_id[h_step[robot_id]["true_class"]]
				# Z[class_idx] = w
			else:
				# Z = self.clf.predict_proba(h_step[robot_id]["feature_std"]).reshape((-1,1))/ \
				# 	np.array(self.P_c).reshape((-1,1)) # (5,1)
				Z = np.ones((5,1))
				class_idx = self.classifier.classlbl_to_id[h_step[robot_id]["true_class"]]
				Z[class_idx] = w

			filter_beliefs[robot_id] = (self.transition_model@filter_beliefs[robot_id]) * (Z/np.sum(Z))
			filter_beliefs[robot_id] /= np.sum(filter_beliefs[robot_id])
			h_step[robot_id]["filter_belief"] = filter_beliefs[robot_id]

		
		h_step_graphs = []
		# for all robots TRY to construct a full graph
		for robot_id in range(len(h_step)):
			if "visited" not in h_step[robot_id]:
				graph = self._construct_graph(h_step, robot_id, range_th)

				for robot_node_id, data in graph["connectivity"].items():
					h_step[robot_node_id]["visited"] = True

				if len(graph["connectivity"]) > 1:
					# init the msg coming from the fake node -1 (the filter belief)
					for robot_node_id, data in graph["connectivity"].items():
						data["msgs"][:,len(data["adj"])-1] = filter_beliefs[robot_node_id].flatten()
						data["total_belief_history"] = []

					# loopy BP for n_iteration
					for k in range(n_iteration):
						for robot_node_id, data in graph["connectivity"].items():
							masked_msgs = data["msgs"]*data["masks"]
							new_msgs = np.product(np.where(masked_msgs==0, 1, masked_msgs), axis=2).T
							new_msgs = new_msgs/np.sum(new_msgs, axis=0)
							new_msgs = self.topology_model.T@new_msgs

							for i in range(len(data["adj"])-1):
								receiver_id = data["adj"][i]
								r_idx_in_receiver_adj = graph["connectivity"][receiver_id]["adj"].index(robot_node_id)
								graph["connectivity"][receiver_id]["msgs"][:,r_idx_in_receiver_adj] = new_msgs[:,i]

						for robot_node_id, data in graph["connectivity"].items():
							current_total_belief = np.product(data["msgs"], axis=1, keepdims=True)
							data["total_belief_history"].append(current_total_belief/np.sum(current_total_belief))

					# save the full graph centered on spy_robot_id, in the spy_robot_id step
					h_step_graphs.append(graph)

					# compute the product of the msgs (the marginals for each robot in the full graph)
					# and put the marginals into the filter_beliefs list for the next time step
					for robot_node_id, data in graph["connectivity"].items():
						if robot_node_id == spy_robot_id:
							# init the next filter_belief with the current total_belief and save it
							filter_beliefs[robot_node_id] = np.product(data["msgs"], axis=1, keepdims=True)
							filter_beliefs[robot_node_id] /= np.sum(filter_beliefs[robot_node_id])
							h_step[robot_node_id]["total_belief"] = filter_beliefs[robot_node_id]
						else:
							# just save the current total_belief
							temp = np.product(data["msgs"], axis=1, keepdims=True)
							temp /= np.sum(temp)
							h_step[robot_node_id]["total_belief"] = temp


		# in any case (sia se esiste il full graph che se non esiste)
		total_belief = filter_beliefs[spy_robot_id]

		return total_belief, h_step_graphs

	def _update_loop(self, filter_beliefs, h_step, spy_robot_id, range_th, n_iteration, order_type="sequential"):

		assert order_type in ["sequential", "random"], "wrong order_type"

		h_step_graphs = []
		# for all robots TRY to construct a full graph
		for robot_id in range(len(h_step)):
			if "visited" not in h_step[robot_id]:
				graph = self._construct_graph(h_step, robot_id, range_th)

				for robot_node_id, data in graph["connectivity"].items():
					h_step[robot_node_id]["visited"] = True

				if len(graph["connectivity"]) > 1:
					# init the msg coming from the fake node -1 (the filter belief)
					for robot_node_id, data in graph["connectivity"].items():
						data["msgs"][:,len(data["adj"])-1] = filter_beliefs[robot_node_id].flatten()
						data["total_belief_history"] = []

					# loopy BP for n_iteration
					for k in range(n_iteration):

						for robot_node_id, data in graph["connectivity"].items():
							current_total_belief = np.product(data["msgs"], axis=1, keepdims=True)
							data["total_belief_history"].append(current_total_belief/np.sum(current_total_belief))

						if order_type == "random":
							# random order
							robot_node_ids = list(graph["connectivity"].keys())
							random.shuffle(robot_node_ids)
							for robot_node_id in robot_node_ids:
								data = graph["connectivity"][robot_node_id]
								masked_msgs = data["msgs"]*data["masks"]
								new_msgs = np.product(np.where(masked_msgs==0, 1, masked_msgs), axis=2).T
								new_msgs = new_msgs/np.sum(new_msgs, axis=0)
								new_msgs = self.topology_model.T@new_msgs

								for i in range(len(data["adj"])-1):
									receiver_id = data["adj"][i]
									r_idx_in_receiver_adj = graph["connectivity"][receiver_id]["adj"].index(robot_node_id)
									graph["connectivity"][receiver_id]["msgs"][:,r_idx_in_receiver_adj] = new_msgs[:,i]

						elif order_type == "sequential":
							# sequential order
							for robot_node_id, data in graph["connectivity"].items():
								masked_msgs = data["msgs"]*data["masks"]
								new_msgs = np.product(np.where(masked_msgs==0, 1, masked_msgs), axis=2).T
								new_msgs = new_msgs/np.sum(new_msgs, axis=0)
								new_msgs = self.topology_model.T@new_msgs

								for i in range(len(data["adj"])-1):
									receiver_id = data["adj"][i]
									r_idx_in_receiver_adj = graph["connectivity"][receiver_id]["adj"].index(robot_node_id)
									graph["connectivity"][receiver_id]["msgs"][:,r_idx_in_receiver_adj] = new_msgs[:,i]



					# save the full graph centered on spy_robot_id, in the spy_robot_id step
					h_step_graphs.append(graph)

					# compute the product of the msgs (the marginals for each robot in the full graph)
					# and put the marginals into the filter_beliefs list for the next time step
					for robot_node_id, data in graph["connectivity"].items():
						#if robot_node_id == spy_robot_id:
						filter_beliefs[robot_node_id] = np.product(data["msgs"], axis=1, keepdims=True)
						filter_beliefs[robot_node_id] /= np.sum(filter_beliefs[robot_node_id])
						h_step[robot_node_id]["total_belief"] = filter_beliefs[robot_node_id]


		# in any case (sia se esiste il full graph che se non esiste)
		total_belief = filter_beliefs[spy_robot_id]

		return total_belief, h_step_graphs

class BabySitter:
	def __init__(self):
		self.classifier = Classifier()

	def display_classification_metrics(self, experiments, robot_ids, experiments_graphs):
		# P = tp / tp + fp
		# R = tp / tp + fn
		y_true, y_pred = [],[]
		for r_id, exp in zip(robot_ids, experiments):
			for h_step in exp:
				# if "pred_class" in h_step[r_id] and len([rr for rr in h_step[r_id]["rab_readings"] if rr["range"] <= range_th]) > 0:
				# 	y_true.append(h_step[r_id]["true_class"])
				# 	y_pred.append(h_step[r_id]["pred_class"])
				if "pred_class" in h_step[r_id]:
					y_true.append(h_step[r_id]["true_class"])
					y_pred.append(h_step[r_id]["pred_class"])

		c = Classifier()



		# score_lbls = ["G", "C", "S", "V", "I", "macro avg", "weighted avg"]
		# headers = [" ", "precision", "recall", "f1-score", "support"]
		# table = []
		# report_dict = classification_report(y_true,
		# 					  y_pred,
		# 					  zero_division=0,
		# 					  output_dict=True,
		# 					  labels=list(c.classlbl_to_id.keys()))
		#
		# for score_lbl in score_lbls:
		# 	row = []
		# 	for header in headers:
		# 		if header in report_dict[score_lbl]:
		# 			row.append(report_dict[score_lbl][header])
		# 		else:
		# 			row.append(score_lbl)
		#
		# 	table.append(row)
		#
		# print(tabulate(table, headers, tablefmt="github"))
		#
		# print(tabulate(confusion_matrix(y_true,
		# 					   y_pred,
		# 					   labels=list(c.classlbl_to_id.keys())), tablefmt="github"))

		print(classification_report(y_true,
							  y_pred,
							  zero_division=0,
							  output_dict=False,
							  labels=list(c.classlbl_to_id.keys())))

		print(confusion_matrix(y_true,
							   y_pred,
							   labels=list(c.classlbl_to_id.keys())))


	def draw_map_predicted_trajectories(self, experiments, robot_ids, o_idx, range_th=None):
		with open("data/test/test_map_ground_truth.pickle", "rb") as f:
			map_ground_truth = pickle.load(f)

		with open("data/test/test_map_wall_boundary_vertices.pickle", "rb") as f:
			map_wall_boundary_vertices = pickle.load(f)

		fig, ax = plt.subplots()
		fig.set_size_inches(6.4,9.6)
		fig.set_dpi(200.0)

		for boundary_id, vertices in map_wall_boundary_vertices.items():
			vertices = [[v[0], v[1]] for v in vertices]
			boundary = plt.Polygon(vertices, closed=True, fill=None, edgecolor='black')
			ax.add_patch(boundary)

		for class_lbl, poly_data in map_ground_truth.items():
			for vertices in poly_data["poly_boundary_vertices"]:
				boundary = plt.Polygon(vertices, closed=True, fill=True, edgecolor=None,
									   facecolor=poly_data["color"], alpha=0.3)
				ax.add_patch(boundary)

		circle_radius = 0.05
		face_radius = 0.008
		for exp_idx, (exp, r_id, oi) in enumerate(zip(experiments, robot_ids, o_idx)):
			for h_idx,(h_step,o) in enumerate(zip(exp, oi)):
				#nn = [s["true_class"] for s in step["rab_readings"] if s["range"]<=range_th]
				if "pred_class" in h_step[r_id]:
					if h_step[r_id]["true_class"] != h_step[r_id]["pred_class"]:
						c = map_ground_truth[h_step[r_id]["pred_class"]]["color"]
						position = (h_step[r_id]["x"], h_step[r_id]["y"])
						circle = plt.Circle(position, radius=circle_radius, edgecolor=None, facecolor=(c[0], c[1], c[2], 0.5))
						s = f"{str(exp_idx)}_{str(o)}"

						ax.text(*position, s, ha='center', va='center', size=5.0)
						ax.add_patch(circle)

		# plt.axis('scaled')
		plt.axis('equal')
		plt.axis('off')
		#plt.savefig('img.png', dpi=200.0)
		plt.show()

	def draw_map_h_step_graph(self, h_step, r_id, h_step_graphs):
		with open("data/test/test_map_ground_truth.pickle", "rb") as f:
			map_ground_truth = pickle.load(f)

		with open("data/test/test_map_wall_boundary_vertices.pickle", "rb") as f:
			map_wall_boundary_vertices = pickle.load(f)

		fig, ax = plt.subplots()
		fig.set_size_inches(6.4,9.6)

		for boundary_id, vertices in map_wall_boundary_vertices.items():
			vertices = [[v[0], v[1]] for v in vertices]
			boundary = plt.Polygon(vertices, closed=True, fill=None, edgecolor='black')
			ax.add_patch(boundary)

		for class_lbl, poly_data in map_ground_truth.items():
			for vertices in poly_data["poly_boundary_vertices"]:
				boundary = plt.Polygon(vertices, closed=True, fill=True, edgecolor=None,
									   facecolor=poly_data["color"], alpha=0.1)
				ax.add_patch(boundary)

		r_radius = 0.2
		n_radius = 0.1
		face_radius = 0.008

		# i robot
		circles, lines, texts = [], [], []
		for step_idx, step in enumerate(h_step):
			radius = r_radius if int(step["fb_id"][3:]) == r_id else n_radius

			if "total_belief" in step:
				facecolor = map_ground_truth[self.classifier.id_to_classlbl[np.argmax(step["total_belief"])]]["color"]
			elif "filter_belief" in step:
				facecolor = map_ground_truth[self.classifier.id_to_classlbl[np.argmax(step["filter_belief"])]]["color"]

			circles.append(plt.Circle((step["x"],step["y"]), radius=radius, facecolor=facecolor, zorder=2))
			ax.text(step["x"], step["y"], step["fb_id"][3:], size=5.0, color="black", weight="bold")

		if h_step[r_id]["pred_class"] != h_step[r_id]["true_class"]:
			circles.append(plt.Circle((-4,11), radius=2.0, facecolor="red", zorder=2))

		print(f"Graphs: {len(h_step_graphs)}")
		for graph in h_step_graphs:
			print()
			for node_id, data in graph.items():
				print(node_id, data["adj"])
				for adj in data["adj"][:len(data["adj"])-1]:
					lines.append(plt.Line2D((h_step[node_id]["x"],h_step[adj]["x"]),(h_step[node_id]["y"],h_step[adj]["y"]),
											linewidth=0.8, color="blue", zorder=1))


		for c in circles:
			ax.add_patch(c)
		for l in lines:
			ax.add_patch(l)

		plt.axis('equal')
		plt.axis('off')
		plt.show()

		for c in circles:
			c.remove()
		for l in lines:
			l.remove()

		ax.texts.clear()

	def draw_map_h_step_graph_plus_convergence(self, h_step, r_id):
		with open("data/test/test_map_ground_truth.pickle", "rb") as f:
			map_ground_truth = pickle.load(f)

		with open("data/test/test_map_wall_boundary_vertices.pickle", "rb") as f:
			map_wall_boundary_vertices = pickle.load(f)

		fig = plt.figure()
		gs = gridspec.GridSpec(5, 2)
		ax = fig.add_subplot(gs[:,0])
		right_column_axs = []
		for i in range(5):
			right_column_axs.append(fig.add_subplot(gs[i,1]))
		fig.set_size_inches(12,12)

		for boundary_id, vertices in map_wall_boundary_vertices.items():
			vertices = [[v[0], v[1]] for v in vertices]
			boundary = plt.Polygon(vertices, closed=True, fill=None, edgecolor='black')
			ax.add_patch(boundary)

		for class_lbl, poly_data in map_ground_truth.items():
			for vertices in poly_data["poly_boundary_vertices"]:
				boundary = plt.Polygon(vertices, closed=True, fill=True, edgecolor=None,
									   facecolor=poly_data["color"], alpha=0.1)
				ax.add_patch(boundary)

		r_radius = 0.2
		n_radius = 0.1
		face_radius = 0.008

		# i robot e il grafo
		circles, lines, texts = [], [], []
		for step_idx, step in enumerate(h_step):
			radius = r_radius if int(step["fb_id"][3:]) == r_id else n_radius

			if "total_belief" in step:
				facecolor = map_ground_truth[self.classifier.id_to_classlbl[np.argmax(step["total_belief"])]]["color"]
			elif "filter_belief" in step:
				facecolor = map_ground_truth[self.classifier.id_to_classlbl[np.argmax(step["filter_belief"])]]["color"]

			circles.append(plt.Circle((step["x"],step["y"]), radius=radius, facecolor=facecolor, zorder=2))
			ax.text(step["x"], step["y"], step["fb_id"][3:], size=5.0, color="black", weight="bold")

		if h_step[r_id]["pred_class"] != h_step[r_id]["true_class"]:
			circles.append(plt.Circle((-4,11), radius=2.0, facecolor="red", zorder=2))

		if "graph" in h_step[r_id]:
			for node_id, data in h_step[r_id]["graph"].items():
				#print(node_id, data["adj"])
				for adj in data["adj"][:len(data["adj"])-1]:
					lines.append(plt.Line2D((h_step[node_id]["x"],h_step[adj]["x"]),(h_step[node_id]["y"],h_step[adj]["y"]),
											linewidth=0.8, color="blue", zorder=1))
		ax.axis('equal')
		ax.axis('off')


		#i plot convergenza
		if "graph" in h_step[r_id]:
			selected_nodes = [r_id]
			node_population = list(h_step[r_id]["graph"].keys())[1:]
			selected_nodes += random.choices(node_population, k=min(4,len(node_population)))

			for index, id in enumerate(selected_nodes):
				print(np.hstack(h_step[r_id]["graph"][id]["total_belief_history"]).T.shape)
				right_column_axs[index].plot(np.hstack(h_step[r_id]["graph"][id]["total_belief_history"]).T, "-o")

		for c in circles:
			ax.add_patch(c)
		for l in lines:
			ax.add_patch(l)

		plt.show()

		for c in circles:
			c.remove()
		for l in lines:
			l.remove()

		ax.texts.clear()

		for a in right_column_axs:
			a.cla()

	def save_map_h_step_graph(self, experiments, robot_ids, experiments_graphs, o_idx, belief_type):
		with open("data/test/test_map_ground_truth.pickle", "rb") as f:
			map_ground_truth = pickle.load(f)

		with open("data/test/test_map_wall_boundary_vertices.pickle", "rb") as f:
			map_wall_boundary_vertices = pickle.load(f)

		fig, ax = plt.subplots()
		fig.set_size_inches(3.2,4.8)
		fig.set_dpi(150.0)

		for boundary_id, vertices in map_wall_boundary_vertices.items():
			vertices = [[v[0], v[1]] for v in vertices]
			boundary = plt.Polygon(vertices, closed=True, fill=None, edgecolor='black')
			ax.add_patch(boundary)

		for class_lbl, poly_data in map_ground_truth.items():
			for vertices in poly_data["poly_boundary_vertices"]:
				boundary = plt.Polygon(vertices, closed=True, fill=True, edgecolor=None,
									   facecolor=poly_data["color"], alpha=0.1)
				ax.add_patch(boundary)

		r_radius = 0.2
		n_radius = 0.1
		face_radius = 0.008

		for exp_idx, (r_id, exp, oi) in enumerate(zip(robot_ids, experiments, o_idx)):
			for h_step_idx, (h_step, o) in enumerate(zip(exp, oi)):
				circles, lines, texts = [], [], []
				for step_idx, step in enumerate(h_step):
					radius = r_radius if int(step["fb_id"][3:]) == r_id else n_radius

					if belief_type == "total_belief":
						if "total_belief" in step:
							facecolor = map_ground_truth[self.classifier.id_to_classlbl[np.argmax(step["total_belief"])]]["color"]
						else:
							facecolor = map_ground_truth[self.classifier.id_to_classlbl[np.argmax(step["filter_belief"])]]["color"]
					elif belief_type == "filter_belief":
						facecolor = map_ground_truth[self.classifier.id_to_classlbl[np.argmax(step["filter_belief"])]]["color"]

					circles.append(plt.Circle((step["x"],step["y"]), radius=radius, facecolor=facecolor, zorder=2))

					if int(step["fb_id"][3:]) == r_id:
						if step["pred_class"] != step["true_class"]:
							circles.append(plt.Circle((-4,11), radius=2.0, facecolor="red", zorder=2))

					# if int(step["fb_id"][3:]) == r_id:
					# 	for node_id, data in step["graph"].items():
					# 		for adj in data["adj"][:len(data["adj"])-1]:
					# 			lines.append(plt.Line2D((h_step[node_id]["x"],h_step[adj]["x"]),(h_step[node_id]["y"],h_step[adj]["y"]), linewidth=0.8, color="blue", zorder=1))

				for graph in experiments_graphs[exp_idx][h_step_idx]:
					#print()
					for node_id, data in graph["connectivity"].items():
						#print(node_id, data["adj"])
						for adj in data["adj"][:len(data["adj"])-1]:
							lines.append(plt.Line2D((h_step[node_id]["x"],h_step[adj]["x"]),(h_step[node_id]["y"],h_step[adj]["y"]),
													linewidth=0.8, color="blue", zorder=1))

				for c in circles:
					ax.add_patch(c)
				for l in lines:
					ax.add_patch(l)

				plt.axis('equal')
				plt.axis('off')
				if belief_type== "total_belief":
					plt.savefig("script/video_global_graph/"+str(exp_idx).rjust(2,'0')+"_"+str(o).rjust(3, '0')+".png", dpi=150.0)
				elif belief_type == "filter_belief":
					plt.savefig("script/video_local_graph/"+str(exp_idx).rjust(2,'0')+"_"+str(o).rjust(3, '0')+".png", dpi=150.0)
				#plt.show()

				for c in circles:
					c.remove()
				for l in lines:
					l.remove()

				ax.texts.clear()

	def stack_map_h_step_graphs_frame(self, input_dirs):

		h_frames_paths = []
		for dir in input_dirs:
			full_paths = []
			for file_name in sorted(os.listdir(dir)):
				full_paths.append(os.path.join(dir,file_name))
			h_frames_paths.append(full_paths)

		pprint(h_frames_paths)


		for idx, h_frames_path in enumerate(zip(*h_frames_paths)):
			h_frames_imgs = []
			for frame_path in h_frames_path:
				h_frames_imgs.append(mpimg.imread(frame_path))

			plt.imsave(f"script/video_stacked_frame/{str(idx).rjust(3, '0')}.png", np.hstack(h_frames_imgs))

	def _draw_map_h_step_graph(self, h_step, r_id, h_step_graphs, type):
		with open("data/test/test_map_ground_truth.pickle", "rb") as f:
			map_ground_truth = pickle.load(f)

		fig, ax = plt.subplots()
		fig.set_size_inches(6.4,6.4)

		r_radius = 4
		n_radius = 2

		# i robot
		circles, lines, texts = [], [], []
		for step_idx, step in enumerate(h_step):
			radius = r_radius if int(step["fb_id"][3:]) == r_id else n_radius

			if type=="belief":
				facecolor = map_ground_truth[self.classifier.id_to_classlbl[np.argmax(step["belief"])]]["color"]
				facecolor = (facecolor[0], facecolor[1], facecolor[2], np.max(step["belief"]))
			else:
				if "total_belief" in step:
					facecolor = map_ground_truth[self.classifier.id_to_classlbl[np.argmax(step["total_belief"])]]["color"]
				elif "belief" in step:
					facecolor = map_ground_truth[self.classifier.id_to_classlbl[np.argmax(step["belief"])]]["color"]

			circles.append(plt.Circle((step["x"],step["y"]), radius=radius, facecolor=facecolor, zorder=2))
			#ax.text(step["x"], step["y"], step["fb_id"][3:], size=8.0, color="red", weight="bold")

		print(f"Graphs: {len(h_step_graphs)}")
		for graph in h_step_graphs:
			#print()
			for node_id, data in graph["connectivity"].items():
				#print(node_id, data["adj"])
				for adj in data["adj"][:len(data["adj"])-1]:
					lines.append(plt.Line2D((h_step[node_id]["x"],h_step[adj]["x"]),(h_step[node_id]["y"],h_step[adj]["y"]),
											linewidth=0.4, color="grey", zorder=1))


		for c in circles:
			ax.add_patch(c)
		for l in lines:
			ax.add_patch(l)

		plt.axis('equal')
		plt.axis('off')
		plt.show()


class CustomPGM:
	def __init__(self, filter):
		self.filter = filter

		f_xttxt = DiscreteFactor(["xtt","xt"], [5,5], filter.transition_model.flatten())
		ff_fxtt = DiscreteFactor(["xtt","f"],[5,1], np.ones(5))
		ff_ztxt = DiscreteFactor(["xt","zt"], [5,1], np.ones(5))


		self.G = MarkovNetwork()
		self.G.add_nodes_from(["xtt","xt","zt","f"])
		self.G.add_factors(f_xttxt, ff_fxtt, ff_ztxt)
		self.G.add_edges_from([("xtt","xt"),("xt","zt"),("xtt","f")])


	def update(self, belief, h_step, robot_id, range_th):

		self.G.get_factors("f")[0].values = belief

		Z = self.filter.clf.predict_proba(h_step[robot_id]["feature_std"]).reshape((-1,1))/ \
			np.array(self.filter.P_c).reshape((-1,1))

		self.G.get_factors("zt")[0].values = Z

		model = VariableElimination(self.G)
		q = model.query(variables=["xt"], evidence={"f":0, "zt":0})
		q = np.array(q.values, dtype=float).reshape((5,1))
		return q/np.sum(q)


	def predict(self):
		pass


if __name__ == '__main__':

	import time

	# filter = HybridFilter()
	# filter.estimateSpatialModel(dataset=None)
	# slice = filter.spatial_model[2,:,:]
	#
	# print(np.array2string(slice/np.sum(slice), precision=5))



	with open(f"data/test/60/experiments/1/experiment.pickle","rb") as f:
		experiment = pickle.load(f)

	range_th = 100
	HybridFilter()._construct_graph(experiment[500],3,range_th)
	BabySitter().draw_h_step_graph(experiment[500],3,range_th)












