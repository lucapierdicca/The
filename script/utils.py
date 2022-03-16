from bisect import bisect_left
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn import preprocessing
import csv
from collections import namedtuple, deque, Counter
import pickle

from sklearn.svm import SVC
from tqdm import tqdm
from itertools import islice
import math
from pprint import pprint


Point = namedtuple('Point','x y')


def toRadians(degree):
	return degree*np.pi/180.0

def norm(vector):
	return np.linalg.norm(vector)

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

def load_dataset(dataset_path, map_ground_truth_path, row_range=None):

	with open(map_ground_truth_path,"rb") as f:
		map_ground_truth = pickle.load(f)

	file = open(dataset_path, mode="r")
	reader = csv.reader(file, delimiter='|')

	if row_range is not None:
		reader = islice(reader, row_range[0], row_range[1])

	dataset = []
	for row in reader:
		step = {'clock':0,
				'fb_id':'',
				'x':0, 'y':0,
				'theta':0,
				'v_left':0,
				'v_right':0,
				'true_class':'',
				'color':None,
				'world_model_long':{},
				'world_model_short':{}}

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

		if "template" in dataset_path or "test" in dataset_path or "train" in dataset_path: # perché per ora è l'unico che ha il campo occluded nel .csv
			step['world_model_long'] = {float(row[i].replace(",",".")):{"distance":float(row[i+1].replace(",",".")),
																	"occluded":bool(int(row[i+2]))}
									for i in range(7,120*3+7,3)}

		else:
			step['world_model_long'] = {float(row[i].replace(",",".")):float(row[i+1].replace(",",".")) for i in range(7,120*2+7,2)}
		#step['world_model_short'] = {float(row[i].replace(",",".")):float(row[i+1].replace(",",".")) for i in range(120*2+7,120*2+120*2+7,2)}

		dataset.append(step)

	file.close()

	return dataset

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
		aperture = toRadians(360.0/n_intervals)
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
		d = [np.linalg.norm(np.array(v[i]) - c) for i in range(len(v)-1)]
		for i in range(len(v)-1):
			avg_shape += np.linalg.norm(np.array(v[i]) - c)#/max(d)

		avg_shape /= (len(v)-1)
		B4 = avg_shape

		# B.5
		std_shape = 0.0
		d = [np.power(np.linalg.norm(np.array(v[i]) - c) - B4, 2)]
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
		diameters = []
		if handle_occlusion == True:
			d = [data["distance"] for data in list(measurement.values()) if data["occluded"] == False]
		else:
			d = [data["distance"] for data in list(measurement.values())]

		for i in range(len(d) - 60):
			diameters.append(d[i]+d[i+59])

		B7 = max(diameters)
		B8 = min(diameters)

		return [B1, B2, B3, B4, B5, B7, B8]

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

	def estimateObservationModel(self, dataset, load_model=True):

		X_train, y_train = [],[]

		if load_model == False:
			for step in tqdm(dataset):
				z = self.classifier.preProcess(step["world_model_long"], 3)
				X_train.append(self.extractFeature(z, handle_occlusion=False))
				y_train.append(step["true_class"])

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

			with open(f"P_c.pickle",'wb') as f:
				pickle.dump(self.P_c, f)
			with open(f"model_svc.pickle",'wb') as f:
				pickle.dump(self.clf, f)
			with open(f"scaler_svc.pickle",'wb') as f:
				pickle.dump(self.scaler, f)

		else:
			with open(f"P_c.pickle","rb") as f:
				self.P_c = pickle.load(f)
			with open(f"model_svc.pickle","rb") as f:
				self.clf = pickle.load(f)
			with open(f"scaler_svc.pickle","rb") as f:
				self.scaler = pickle.load(f)

	def extractFeature(self, measurement, handle_occlusion=False):
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
		d = [np.linalg.norm(np.array(v[i]) - c) for i in range(len(v)-1)]
		for i in range(len(v)-1):
			avg_shape += np.linalg.norm(np.array(v[i]) - c)#/max(d)

		avg_shape /= (len(v)-1)
		B4 = avg_shape

		# B.5
		std_shape = 0.0
		d = [np.power(np.linalg.norm(np.array(v[i]) - c) - B4, 2)]
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
		diameters = []
		if handle_occlusion == True:
			d = [data["distance"] for data in list(measurement.values()) if data["occluded"] == False]
		else:
			d = [data["distance"] for data in list(measurement.values())]

		for i in range(len(d) - 60):
			diameters.append(d[i]+d[i+59])

		B7 = max(diameters)
		B8 = min(diameters)

		return [B1, B2, B3, B4, B5, B7, B8]

	def update(self, belief, feature, weight=None):
		# predict step
		bt_t_1 = [0.0]*self.state_dim
		for i in range(self.state_dim):
			for j in range(self.state_dim):
				bt_t_1[j] += self.transition_model[j, i]*belief[i]

		# update step
		log_prior = np.log(np.array(bt_t_1).reshape((-1,1)))

		P_c_x = self.clf.predict_proba(feature) # (5,1)
		P_x_c = P_c_x / self.P_c
		log_likelihood = np.log(P_x_c)

		if weight is None:
			log_posterior = log_likelihood + log_prior
		else:
			log_posterior = (1.0-weight)*(log_likelihood) + (weight*log_prior)

		#softmax
		btt = np.exp(log_posterior)/np.sum(np.exp(log_posterior))

		return btt.reshape((1,-1)).tolist()[0]

	def predict(self, belief):
		argmax = np.argmax(np.array(belief))
		return self.classifier.id_to_classlbl[argmax]


if __name__ == '__main__':


	filter = GaussianFilter("linear", "geometricB")
	print(filter.classifier.classlbl_to_id)

	# train = load_dataset("data/train/unstructured_occluded_3.csv",
	# 					 "data/train/train_map_2_ground_truth.pickle")
	#
	# print(len(train))
	# filter.estimateTransitionModel(train)
	# print(filter.transition_model)
	#
	# train_unique = []
	# train_unique.append(train[0])
	# for i in range(1,len(train),1):
	# 	if train[i]["true_class"] != train_unique[-1]["true_class"]:
	# 		train_unique.append(train[i])
	#
	# print(train_unique)
	#
	#
	# print(len(train_unique))
	# filter.estimateTransitionModel(train_unique)
	# print(filter.transition_model)







