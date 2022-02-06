from p5 import *
import numpy as np
from Utils import loadDataset
import matplotlib.pyplot as plt
import csv
from collections import deque
from scipy.spatial import ConvexHull
from pprint import pprint
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score





def setup():
	size(500, 800)
	no_loop()

def draw():
	background(160)
	translate(250.0,600.0,0.0)
	rotate_z(3.1415)
	rotate_y(3.1415)

	
	for classlbl,data in map_ground_truth.items():

		for bb in data['bb']:
			bb_vertices = [(bb[0][0]*50,bb[0][1]*50), 
							(bb[1][0]*50,bb[0][1]*50),
							(bb[1][0]*50,bb[1][1]*50),
							(bb[0][0]*50,bb[1][1]*50)]

			begin_shape()
			for v in bb_vertices:
				vertex(v[0],v[1])
			end_shape("CLOSE")

def extractFeature(measurement):

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

	return [convex_A, convex_P, max_diameter, min_diameter]





if __name__ == '__main__':

	dataset = loadDataset("randomwalk_dataset.csv")
	print("Dataset: ", len(dataset))


	support = {}
	for step in dataset:
		if step['true_class'] not in support:
			support[step['true_class']] = 1
		else:
			support[step['true_class']] += 1

	pprint(support)


	classlbl_to_id = {classlbl:index for index, classlbl in enumerate(list(support.keys()))}

	X,y = [],[]
	for step in dataset[:1000]:
		X.append(extractFeature(step['world_model_long']))
		y.append(classlbl_to_id[step['true_class']])

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)



	clf = LinearDiscriminantAnalysis(store_covariance=True)
	clf.fit(np.array(X_train), np.array(y_train))

	print(clf.means_[0,:].reshape((-1,1)))

	y_pred = clf.predict(X_test)

	print(classification_report(y_test, y_pred))


	# x,y,c = [],[],[]
	# for step in dyn_dataset:
	# 	solidity, convexity = extractFeature(step['world_model_long'])

	# 	x.append(solidity)
	# 	#y.append(convexity)
	# 	if step['true_class'] == 'I': y.append(1)
	# 	if step['true_class'] == 'C': y.append(2)
	# 	if step['true_class'] == 'V': y.append(3)
	# 	if step['true_class'] == 'G': y.append(4)	

	# 	c.append((step['color'][0]/255.0, step['color'][1]/255.0, step['color'][2]/255.0))


	# plt.scatter(x, y, s=3.0, c=c, alpha=0.2)
	# plt.show()





