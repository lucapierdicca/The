import pickle
import random
import xml.etree.ElementTree as ET
import os
from collections import namedtuple
import numpy as np

Point = namedtuple('Point','x y')

def parse_ggb():
    parsed_ggb = {"vertices": {}, "polygons": {}}

    tree = ET.parse(ggb_file)
    root = tree.getroot()

    for element in root.iter("element"):
        if element.attrib["type"] == "point":
            key = element.attrib["label"]
            value = Point(float(element.find("coords").attrib["x"]),
                          float(element.find("coords").attrib["y"]))

            parsed_ggb["vertices"].update({key: value})

    for command in root.iter("command"):
        if command.attrib["name"] == "Polygon":
            key = command.find("output").attrib["a0"]
            value = {"cw_vertices": list(command.find("input").attrib.values()),
                     "class": key[0]}

            parsed_ggb["polygons"].update({key: value})

    return parsed_ggb

def create_map_argos(parsed_ggb):

    box_tag_prototype = "<box id=\"{id}\" size=\"{w},{h},0.5\" movable=\"false\"> <body position=\"{x},{y},0\" orientation=\"0,0,0\" /> </box>"

    id = -1
    n_boundaries = 0
    box_tags = []
    wall_boundary_vertices = {}

    for v_name in list(parsed_ggb["vertices"].keys()):
        if len(v_name) == 6:
            if int(v_name[2]) >= n_boundaries:
                n_boundaries = int(v_name[2])

    for v_name, v_coord in sorted(parsed_ggb["vertices"].items(), key=lambda item: item[0]):
        if len(v_name) == 6:
            if int(v_name[2]) not in wall_boundary_vertices:
                wall_boundary_vertices[int(v_name[2])] = [v_coord]
            else:
                wall_boundary_vertices[int(v_name[2])].append(v_coord)

    for vs in list(wall_boundary_vertices.values()):
        vs.append(vs[0])

    for k, vs in wall_boundary_vertices.items():
        for i in range(len(vs) - 1):
            id += 1
            position = Point(0.5 * (vs[i].x + vs[i + 1].x),
                             0.5 * (vs[i].y + vs[i + 1].y))
            size = Point(0.1 if vs[i].x == vs[i + 1].x else abs(vs[i].x - vs[i + 1].x),
                         0.1 if vs[i].y == vs[i + 1].y else abs(vs[i].y - vs[i + 1].y))

            box_tags.append(box_tag_prototype.format_map({"id": id, "w": size.x, "h": size.y, "x": position.x, "y": position.y}))

    return wall_boundary_vertices, box_tags

def dump_map_ground_truth(parsed_ggb):

    ground_truth = {}

    for poly_name, poly_data_dict in parsed_ggb["polygons"].items():
        poly_boundary_vertices = [parsed_ggb["vertices"][v_name] for v_name in poly_data_dict["cw_vertices"]]
        if poly_data_dict["class"] not in ground_truth:
            if poly_data_dict["class"] == "I":
                color = (1.0, 0, 0, 1.0)
            if poly_data_dict["class"] == "C":
                color = (0, 1.0, 0, 1.0)
            if poly_data_dict["class"] == "V":
                color = (0, 0, 1.0, 1.0)
            if poly_data_dict["class"] == "G":
                color = (1.0, 1.0, 0, 1.0)

            ground_truth[poly_data_dict["class"]] = {"poly_boundary_vertices": [poly_boundary_vertices], "color": color}

        else:
            ground_truth[poly_data_dict["class"]]["poly_boundary_vertices"].append(poly_boundary_vertices)


    with open(f"data/test/test_map_ground_truth.pickle", 'wb') as f:
        pickle.dump(ground_truth, f)

def dump_map_wall_boundary_vertices(wall_boundary_vertices):
    with open(f"data/test/test_map_wall_boundary_vertices.pickle", 'wb') as f:
        pickle.dump(wall_boundary_vertices, f)

ggb_file = "ggb/test_map/geogebra.xml"
argos_proto_file = "wall_proto.argos"
argos_file = "wall_test.argos"

n_run = 30
n_robot = [1,2,5,10,20,30]


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

for b in root.find("arena").findall("box"):
    root.find("arena").remove(b)

parsed_ggb = parse_ggb()
wall_boundary_vertices, box_tags = create_map_argos(parsed_ggb)

#----------------------------------------------

params.attrib["collect_data"] = "true"
system.attrib["threads"] = "1"
experiment.attrib["length"] = "1000"

for b in box_tags:
    box_tag = ET.fromstring(b)
    root.find("arena").append(box_tag)

dump_map_ground_truth(parsed_ggb)
dump_map_wall_boundary_vertices(wall_boundary_vertices)

for i,nr in enumerate(n_robot):

    try:
        os.mkdir("data/test/"+str(nr))
    except OSError as error:
        print(error)

    for j,s in enumerate(runs_seeds[i]):
        print(f"Experiment {i+1}_{nr}_{s}")
        experiment.attrib["random_seed"] = str(s)
        entity.attrib["quantity"] = str(nr)
        params.attrib["file_name"] =  f"data/test/{nr}/{nr}_{j+1}_{s}_data.csv"
        tree.write(argos_file)
        os.system("argos3 -c "+argos_file)