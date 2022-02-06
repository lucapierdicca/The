import numpy as np

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from pprint import pprint
from scipy.spatial import ConvexHull
import csv

import pickle

PI = 3.14159265358979323846264338327950288

map_ground_truth = {'I':{'bb':[[(-0.75,3.5),(0.75,2.0)]],'color':(255,0,0,255)},
					'C':{'bb':[[(-2.5,9.0),(-0.75,7.5)],
					 			[(0.75,9.0),(2.5,7.5)],
					 			[(-0.75,7.5),(0.75,3.5)],
					 			[(-2.5,3.5),(-0.75,2.0)],
					 			[(0.75,3.5),(2.5,2.0)],
					 			[(-0.75,2.0),(0.75,-0.5)]], 'color':(0,255,0,255)},
					'V':{'bb':[[(-4.0,9.0),(-2.5,7.5)],
					 			[(2.5,9.0),(4.0,7.5)],
					 			[(-4.0,3.5),(-2.5,2.0)],
					 			[(2.5,3.5),(4.0,2.0)],
					 			[(-0.75,-0.5),(0.75,-2.0)]], 'color':(0,0,255,255)},
					'G':{'bb':[[(-0.75,9.0),(0.75,7.5)]],'color':(255,255,0,255)}}

def toRadians(degree):
	return degree*PI/180.0

def norm(vector):
	return np.linalg.norm(vector)

def loadDataset(path="dynamic_dataset.csv"):
	file = open(path)
	reader = csv.reader(file, delimiter='|')

	dataset = []

	for row in reader:
		step = {'clock':0, 
		'x':0, 'y':0, 
		'theta':0, 
		'v_left':0, 
		'v_right':0, 
		'true_class':'',
		'color':None,
		'world_model_long':{}, 
		'world_model_short':{}}
		
		step['clock'] = int(row[0])
		step['x'] = float(row[1].replace(",","."))
		step['y'] = float(row[2].replace(",","."))
		step['theta'] = float(row[3].replace(",","."))
		step['v_left'] = float(row[4].replace(",","."))
		step['v_right'] = float(row[5].replace(",","."))

		classlbl = -1
		for classlbl, data in map_ground_truth.items():
			for bb in data['bb']:
				if step['x'] > bb[0][0] and step['x'] < bb[1][0] and step['y'] > bb[1][1] and step['y'] < bb[0][1]:
					step['true_class'] = classlbl

		assert classlbl != -1

		step['color'] = map_ground_truth[step['true_class']]['color']

		step['world_model_long'] = {float(row[i].replace(",",".")):float(row[i+1].replace(",",".")) for i in range(6,120*2+6,2)}
		step['world_model_short'] = {float(row[i].replace(",",".")):float(row[i+1].replace(",",".")) for i in range(120*2,120*2+120*2+6,2)}

		dataset.append(step)

	return dataset


class Classifier:

	def __init__(self):

		self.classlbl_to_template = {'I':[0,4,0,0],
							'C':[0,0,0,2],
							'V':[0,2,0,1],
							'G':[0,1,2,0]}

		self.classlbl_to_id = {c:i for i,c in enumerate(list(map_ground_truth.keys()))}
		self.id_to_classlbl = {i:c for c,i in self.classlbl_to_id.items()}

	def preProcess(self, measurement, w_len):
		d = list(measurement.values())
		d_pre = [0.0]*len(d)

		for i in range(len(d)):
			avg = 0
			for j in range(w_len):
				avg = avg + d[(i+j) % len(d)]

			avg = avg/w_len

			d_pre[(i+int(w_len/2)) % len(d_pre)] = avg


		return {float(a):d for a,d in zip(measurement.keys(), d_pre)}

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
		start = -PI + aperture/2
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


	def classification_report_(self, y_true,y_pred):
		return classification_report(y_true, y_pred, labels=list(self.classlbl_to_id.keys()))

	def confusion_matrix_(self, y_true, y_pred):
		return confusion_matrix(y_true, y_pred, labels=list(self.classlbl_to_id.keys()))


class DiscreteFilter:

	def __init__(self, dataset):
		self.state_dim = 4
		self.classifier = Classifier()
		self.estimateTransitionModel(dataset)
		self.estimateObservationModel(dataset)

	# learn from %10 steps
	# counts start from 0 (no add-one smoothing)
	def estimateTransitionModel(self, dataset):

		joint = np.zeros((self.state_dim, self.state_dim))

		for i in range(len(dataset)-10):
			classid = self.classifier.classlbl_to_id[dataset[i]['true_class']]
			xt_1 = classid

			classid = self.classifier.classlbl_to_id[dataset[i+10]['true_class']]
			xt = classid

			joint[xt,xt_1] += 1.0

		self.transition_model = joint/np.sum(joint, axis=0)

	# learn from ALL steps
	# counts start from 1 (add-one smoothing)
	def estimateObservationModel(self, dataset):
		
		joint_dict = {}

		for i in range(len(dataset)):
			classid = self.classifier.classlbl_to_id[dataset[i]['true_class']]
			

			z = self.classifier.preProcess(dataset[i]['world_model_long'],3)
			_, feature = self.classifier.extractFeatureRaw(z)
			if tuple(feature) not in joint_dict:
				joint_dict[tuple(feature)] = [0.0]*self.state_dim
			joint_dict[tuple(feature)][classid] += 1.0


		self.feature_to_id = {feature:i for i,feature in enumerate(list(joint_dict.keys()))}

		joint = np.ones((len(joint_dict),self.state_dim))

		for feature,state_counts in joint_dict.items():
			for i,counts in enumerate(state_counts):
				joint[self.feature_to_id[feature], i] += counts

		self.observation_model = joint/np.sum(joint, axis=0)

	# if UNK symbol just return b(t,t-1) (perform predict step only)
	def update(self, belief, feature):
		bt_t_1 = [0.0]*self.state_dim
		for i in range(self.state_dim):
			for j in range(self.state_dim):
				bt_t_1[j] += self.transition_model[j, i]*belief[i]

		if tuple(feature) not in self.feature_to_id:
			return bt_t_1

		btt = [0.0]*self.state_dim
		den = 0.0
		for i in range(self.state_dim):
			btt[i] = self.observation_model[self.feature_to_id[tuple(feature)],i]*bt_t_1[i]
			den+=btt[i]

		return [b/den for b in btt]

	def predict(self, belief):
		argmax = np.argmax(np.array(belief))
		return self.classifier.id_to_classlbl[argmax]


class GaussianFilter:
	
	def __init__(self, feature_type, template_dataset=None):
		self.state_dim = 4
		self.classifier = Classifier()
		self.feature_type = feature_type
		
		if self.feature_type != "geometric":
			assert template_dataset != None

			self.templates = {}
			for step in template_dataset:
				z = self.classifier.preProcess(step['world_model_long'], 3)
				if step['true_class'] not in self.templates:
					self.templates[step['true_class']] = [z]
				else:
					self.templates[step['true_class']].append(z)



	# learn from %10 steps
	# counts start from 1 (add-one smoothing)
	def estimateTransitionModel(self, dataset):
		joint = np.ones((self.state_dim, self.state_dim))

		for i in range(len(dataset)-10):
			classid = self.classifier.classlbl_to_id[dataset[i]['true_class']]
			xt_1 = classid

			classid = self.classifier.classlbl_to_id[dataset[i+10]['true_class']]
			xt = classid

			joint[xt,xt_1] += 1.0

		self.transition_model = joint/np.sum(joint, axis=0)


	def estimateObservationModel(self, dataset="load"):
		self.parameters = {}

		if dataset == "load":
			clf = pickle.load(open("observation_model_template.pickle",'rb'))
		else:
			X,y = [],[]
			for i,step in enumerate(dataset):
				#print(i)
				z = self.classifier.preProcess(step['world_model_long'],3)
				X.append(self.extractFeature(z))
				y.append(self.classifier.classlbl_to_id[step['true_class']])

			clf = LinearDiscriminantAnalysis(store_covariance=True)
			clf.fit(X, y)

			#pickle.dump(clf, open("observation_model_template.pickle",'wb'))

		for i in range(clf.means_.shape[0]):
			self.parameters[self.classifier.id_to_classlbl[i]] = {'mu':clf.means_[i,:].reshape((-1,1)), 'sigma':np.linalg.inv(clf.covariance_)}
 

	def extractFeature(self, measurement):
		if self.feature_type == "geometric":
			return self.extractFeatureGeometric(measurement)
		else:
			return self.extractFeatureTemplate(measurement)
	

	def extractFeatureGeometric(self, measurement):
		poly_v = []
		for a,d in measurement.items():
			x = d*np.cos(a)
			y = d*np.sin(a)

			poly_v.append((x,y))

		poly_v.append(poly_v[0])


		# Perimeter and Area (shoelace formula)
		A = 0.0
		P = 0.0
		for i in range(len(poly_v)-1):
			A += poly_v[i][0]*poly_v[i+1][1] - poly_v[i][1]*poly_v[i+1][0]
			P += np.linalg.norm(np.array(poly_v[i+1])-np.array(poly_v[i]))

		# Perimeter and Area convex_hull
		hull_indices = ConvexHull(poly_v[:-1]).vertices
		hull_v = [poly_v[index] for index in hull_indices]
		hull_v.append(hull_v[0])

		convex_A = 0.0
		convex_P = 0.0
		for i in range(len(hull_v)-1):
			convex_A += hull_v[i][0]*hull_v[i+1][1] - hull_v[i][1]*hull_v[i+1][0]
			convex_P += np.linalg.norm(np.array(hull_v[i+1])-np.array(hull_v[i]))

		#Max and min Feret's diameters
		max_diameter = 0.0
		min_diameter = 0.0

		diameters = []

		d = list(measurement.values())
		for i in range(len(d) - 60):
			diameters.append(d[i]+d[i+59])

		max_diameter = max(diameters)
		min_diameter = min(diameters)

		return [(0.5*A)/(PI*150.0**2), P/(2*PI*150.0)]


	def extractFeatureTemplate(self, measurement):

		def minDistance(value1, value2):
			distance_values = []

			for i in range(120):
				s = 0
				for v1,v2 in zip(value1,value2):
					s += v1*v1 - 2*v1*v2 + v2*v2
				distance_values.append(s)

				value2.insert(0, value2[-1])
				value2.pop()

			return min(distance_values)/(120*150**2)


		feature = []
		for classlbl,templates_readings in self.templates.items():
			avg_min_distance = []
			for template_reading in templates_readings:
				avg_min_distance.append(minDistance(list(template_reading.values()), list(measurement.values())))

			avg_min_distance = sum(avg_min_distance)/len(avg_min_distance)
			feature.append(avg_min_distance)

		return feature


	def gaussian(self, x, mu, sig):
	    return np.exp(-0.5*(x-mu).T@sig@(x-mu))

	def update(self, belief, feature):
		#predict step
		bt_t_1 = [0.0]*self.state_dim
		for i in range(self.state_dim):
			for j in range(self.state_dim):
				bt_t_1[j] += self.transition_model[j, i]*belief[i]

		#update step
		btt = [0.0]*self.state_dim
		den = 0.0
		for i in range(self.state_dim):
			pdf = self.gaussian(np.array(feature).reshape((-1,1)),
								self.parameters[self.classifier.id_to_classlbl[i]]['mu'],
								self.parameters[self.classifier.id_to_classlbl[i]]['sigma'])
			btt[i] = pdf*bt_t_1[i]
			den+=btt[i]

		return [b/den for b in btt]


	def predict(self, belief):
		argmax = np.argmax(np.array(belief))
		return self.classifier.id_to_classlbl[argmax]


if __name__ == '__main__':
	from pprint import pprint
	
	template_dataset = loadDataset('template.csv')
	train_unstructured = loadDataset("train_unstructured.csv")
	gaussian_filter = GaussianFilter('template', template_dataset)

	z = gaussian_filter.classifier.preProcess(train_unstructured[0]['world_model_long'], 3)
	feature = gaussian_filter.extractFeatureTemplate(z)

	#pickle.dump(feature, open("f.pickle", 'wb'))
	f = pickle.load(open("f.pickle", 'rb'))

	print(f)
	

