import pickle
import random
import xml.etree.ElementTree as ET
import os
from pprint import pprint
import numpy as np
from matplotlib import pyplot as plt


def T(point):
    T = np.array([[0.0,0.0,point[0]],
                  [0.0,0.0,point[1]],
                  [0.0,0.0,0.0]])
    return T

def S(alpha_x, alpha_y):
    S = np.array([[alpha_x,0.0,0.0],
                  [0.0,alpha_y,0.0],
                  [0.0,0.0,1.0]])
    return S

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

# apri la mappa di training
with open("data/train/train_map_ground_truth.pickle","rb") as f:
    map_ground_truth = pickle.load(f)

# trasforma i vertici dei boundaries in homogeneous coordinates
# lascia solo la colonna centrale della mappa (percé per adesso ti interessa solo quella)
map_ground_truth = {key:[[[v[0], v[1], 1.0] for v in boundary]
                         # solo la colonna centrale della mappa
                         for boundary in value["poly_boundary_vertices"] if boundary[0][0] == -0.75]
                            for key,value in map_ground_truth.items()}

# ottieni le posizioni dove vuoi campionare le classi facendo questo:
# scali i 4 vertici originali dei boundaries (trasli scali e ritrasli)
# e poi calcoli i 5 midpoints dei nuovi vertici
# per ogni classe avrai 4+5 posizioni (una griglia di 9 punti)
s = 0.7
class_lbl_to_template_positions = {}
for class_lbl,boundaries in map_ground_truth.items():
    for b in boundaries:
        vertices = np.array(b)
        # class boundary centroid
        c = np.mean(vertices, axis=0)

        l1 = np.linalg.norm(vertices[0] - vertices[1])
        l2 = np.linalg.norm(vertices[1] - vertices[2])

        #vertices_prime = np.transpose((np.eye(3)+T(c))@S((l1-0.4)/l1, (l2-0.4)/l2)@(np.eye(3)-T(c))@vertices.T)
        sx,sy = 0.7,0.28 if class_lbl == "C" else 0.7
        vertices_prime = np.transpose((np.eye(3)+T(c))@S(sx,sy)@(np.eye(3)-T(c))@vertices.T)
        vertices_mid_points = [(vertices_prime[i%4,:] + vertices_prime[(i+1)%4,:]) / 2.0 for i in range(4)]
        #template_positions = list(vertices_prime) + vertices_mid_points + [c]
        template_positions = [c]

        if class_lbl not in class_lbl_to_template_positions:
            class_lbl_to_template_positions[class_lbl] = [template_positions]
        else:
            class_lbl_to_template_positions[class_lbl].append(template_positions)

pprint(class_lbl_to_template_positions)

# plotta le posizioni per un check
template_coordinates = [[v[0],v[1]] for k,bs in class_lbl_to_template_positions.items() for b in bs for v in b]
draw_map_robot_positions(template_coordinates)


#-----------------------------------------
#
argos_proto_file = "wall_proto.argos"
argos_file = "wall_template.argos"

n_run = 1
n_robot = [1]

root_seed = 123
random.seed(root_seed)
runs_seeds = random.sample(range(5000), k=n_run*len(n_robot))
runs_seeds = np.array(runs_seeds).reshape((len(n_robot), n_run))

tree = ET.parse(argos_proto_file)
root = tree.getroot()

params = None
experiment = None
entity = None
system = None
for _ in root.iter("params"): params = _
for _ in root.iter("experiment"): experiment = _
for _ in root.iter("entity"): entity = _
for _ in root.iter("system"): system = _
for _ in root.iter("position"): position = _

#----------------------------------------------

params.attrib["collect_data"] = "true"
system.attrib["threads"] = "1"
experiment.attrib["length"] = "10" # ovvero 100 step quindi int template.csv troverai 10-100 per molte volte

for i,nr in enumerate(n_robot):
    try:
        os.mkdir("data/train/")
    except OSError as error:
        print(error)

    for j,s in enumerate(runs_seeds[i]):
        for tc in template_coordinates:
            print(f"Experiment {i+1}_{nr}_{s}")
            experiment.attrib["random_seed"] = str(s)
            entity.attrib["quantity"] = str(nr)
            params.attrib["file_name"] =  f"data/train/template2.csv"
            position.attrib["min"] = f"{tc[0]},{tc[1]},0.0"
            position.attrib["max"] = f"{tc[0]},{tc[1]},0.0"
            tree.write(argos_file)
            os.system("argos3 -c "+argos_file)