from collections import deque
from pprint import pprint
import pickle
from matplotlib import pyplot as plt
from utils import loadDataset, Classifier, Point
from sklearn.cluster import KMeans, MeanShift
from sklearn import preprocessing
import numpy as np
import math

def draw_map_templates(robot_positions, robot_orientations, measurements):
    with open("data/train/train_map_ground_truth.pickle", "rb") as f:
        map_ground_truth = pickle.load(f)

    with open("data/train/train_map_wall_boundary_vertices.pickle", "rb") as f:
        map_wall_boundary_vertices = pickle.load(f)

    fig, ax = plt.subplots()

    for boundary_id, vertices in map_wall_boundary_vertices.items():
        vertices = [[v[0], v[1]] for v in vertices]
        boundary = plt.Polygon(vertices, closed=True, fill=None, edgecolor='black')

        ax.add_patch(boundary)

    for class_lbl, poly_data in map_ground_truth.items():
        for vertices in poly_data["poly_boundary_vertices"]:
            boundary = plt.Polygon(vertices, closed=True, fill=True, edgecolor=None,
                                   facecolor=poly_data["color"], alpha=0.3)
            ax.add_patch(boundary)


    for rp,ro,z in zip(robot_positions, robot_orientations, measurements):
        w_R_r = np.array([[np.cos(ro), -np.sin(ro)],
                          [np.sin(ro), np.cos(ro)]])

        r_face = np.array([[rp[0]],[rp[1]]]) + w_R_r@np.array([[0.05*np.cos(ro)],
                                                               [0.05*np.sin(ro)]])
        face = plt.Circle(r_face, radius=0.02, edgecolor=None,
                          facecolor="black")
        ax.add_patch(face)

        body = plt.Circle(rp, radius=0.05, edgecolor=None,
                          facecolor=(0.0, 0.0, 1.0, 1.0))
        ax.add_patch(body)

        for angle,d in z.items():
            r_p = np.array([[d/100*np.cos(angle)],
                            [d/100*np.sin(angle)]])
            w_p = np.array([[rp[0]],[rp[1]]]) + w_R_r@r_p
            ray = plt.Line2D([rp[0],w_p[0]],[rp[1],w_p[1]],
                             linewidth=0.2,
                             antialiased=True,
                             color=(1.0,0.0,0.0,1.0))
            ax.add_line(ray)

    # plt.axis('scaled')
    plt.axis('equal')
    plt.axis('off')
    # plt.savefig(name)
    plt.show()

def draw_templates(robot_positions, robot_orientations, measurements):
    fig, ax = plt.subplots()

    for rp,ro,z in zip(robot_positions, robot_orientations, measurements):
        w_R_r = np.array([[np.cos(ro), -np.sin(ro)],
                          [np.sin(ro), np.cos(ro)]])

        r_face = np.array([[rp[0]],[rp[1]]]) + w_R_r@np.array([[0.05*np.cos(ro)],
                                                               [0.05*np.sin(ro)]])
        face = plt.Circle(r_face, radius=0.02, edgecolor=None,
                          facecolor="black")
        ax.add_patch(face)

        body = plt.Circle(rp, radius=0.05, edgecolor=None,
                          facecolor=(0.0, 0.0, 1.0, 1.0))
        ax.add_patch(body)

        for angle,d in z.items():
            r_p = np.array([[d/100*np.cos(angle)],
                            [d/100*np.sin(angle)]])
            w_p = np.array([[rp[0]],[rp[1]]]) + w_R_r@r_p
            ray = plt.Line2D([rp[0],w_p[0]],[rp[1],w_p[1]],
                             linewidth=0.2,
                             antialiased=True,
                             color=(1.0,0.0,0.0,1.0))
            ax.add_line(ray)

        # plt.axis('scaled')
    plt.axis('equal')
    plt.axis('off')
    # plt.savefig(name)
    plt.show()

def draw_map_robot_positions(robot_positions):
    with open("data/train/train_map_ground_truth.pickle", "rb") as f:
        map_ground_truth = pickle.load(f)

    with open("data/train/train_map_wall_boundary_vertices.pickle", "rb") as f:
        map_wall_boundary_vertices = pickle.load(f)

    fig, ax = plt.subplots()

    for boundary_id, vertices in map_wall_boundary_vertices.items():
        vertices = [[v[0], v[1]] for v in vertices]
        boundary = plt.Polygon(vertices, closed=True, fill=None, edgecolor='black')

        ax.add_patch(boundary)

    for class_lbl, poly_data in map_ground_truth.items():
        for vertices in poly_data["poly_boundary_vertices"]:
            boundary = plt.Polygon(vertices, closed=True, fill=True, edgecolor=None,
                                   facecolor=poly_data["color"], alpha=0.3)
            ax.add_patch(boundary)


    for rp in robot_positions:

        body = plt.Circle(rp, radius=0.05, edgecolor=None,
                          facecolor=(0.0, 0.0, 1.0, 1.0))
        ax.add_patch(body)

    plt.axis('equal')
    plt.axis('off')

    plt.show()

def pre_compute_rotated_template(z):
    tm = []
    v1 = deque(z)

    for _ in range(120):
        v1.rotate(1)
        tm.append(list(v1))

    return np.array(tm)

def minDistance(tm, value2):
    value2 = np.array(value2)
    tm = tm-value2
    mins = np.diag(tm@tm.T)
    return np.min(mins)#/(120*(150-10)**2)

# load dataset
classifier = Classifier()
train = loadDataset("data/train/unstructured.csv",
                    "data/train/train_map_ground_truth.pickle")
print("train: ", len(train))

# filter by class and location?
train_V = [step for step in train if step["true_class"] == "V" and step["y"] < 2.0]
print(len(train_V))

# measurements only dataset
X = []
for step in train_V:
    #print(step["theta"])
    z = classifier.preProcess(step['world_model_long'],3)
    #pprint(z)
    zz = {np.arctan2(np.sin(angle+step["theta"]),np.cos(angle+step["theta"])):distance for angle,distance in z.items()}
    dict(sorted(zz.items()))
    #pprint(zz)

    X.append(list(zz.values()))


# look at the robot positions
robot_positions = [[step["x"],step["y"]] for step in train_V]
draw_map_robot_positions(robot_positions)

# scale dataset and find clusters
scaler = preprocessing.StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

n_clusters = 9
kmeans = MeanShift().fit(X)#KMeans(n_clusters=n_clusters, random_state=0).fit(X)


# compute clusters distance matrix
distance_matrix = np.zeros((len(kmeans.cluster_centers_),
                            len(kmeans.cluster_centers_)))

for i in range(len(kmeans.cluster_centers_)):
    tm = pre_compute_rotated_template(kmeans.cluster_centers_[i])
    for j in range(len(kmeans.cluster_centers_)):
        distance_matrix[i,j] = minDistance(tm, kmeans.cluster_centers_[j])

pprint(distance_matrix)



with open(f"data/train/cluster_centers.pickle",'wb') as f:
    pickle.dump(scaler.inverse_transform(kmeans.cluster_centers_), f)

with open(f"data/train/training_cluster_labels.pickle",'wb') as f:
    pickle.dump(kmeans.labels_, f)


with open(f"data/train/cluster_centers.pickle","rb") as f:
    cluster_centers = pickle.load(f)

with open(f"data/train/training_cluster_labels.pickle","rb") as f:
    labels = pickle.load(f)

#pprint(cluster_centers)
#pprint(labels)

cluster_label_to_position = {i:[] for i in range(len(cluster_centers))}
for step,label in zip(train_V, labels):
        cluster_label_to_position[label].append([step["x"], step["y"]])

robot_positions = []
for cl,p in cluster_label_to_position.items():
    print(cl, np.mean(np.array(p), axis=0))
    robot_positions.append(np.mean(np.array(p), axis=0))


robot_orientations = [0.0]*len(cluster_centers)
measurements = [{a:d for a,d in zip(list(train_V[0]["world_model_long"].keys()),c)} for c in cluster_centers]

draw_map_templates(robot_positions, robot_orientations, measurements)

robot_positions = []
r = math.ceil(np.sqrt(len(cluster_centers)))
for i in range(r):
    for j in range(r):
        robot_positions.append([i*5,j*5])
draw_templates(robot_positions[:len(cluster_centers)],robot_orientations, measurements)

