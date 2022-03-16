import random
import xml.etree.ElementTree as ET
import os
import numpy as np



argos_proto_file = "wall_proto.argos"
argos_file = "wall_train.argos"

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

#----------------------------------------------

params.attrib["collect_data"] = "true"
system.attrib["threads"] = "1"
experiment.attrib["length"] = "5000" # which means 50000 step

for i,nr in enumerate(n_robot):

    try:
        os.mkdir("data/train/")
    except OSError as error:
        print(error)

    for j,s in enumerate(runs_seeds[i]):
        print(f"Experiment {i+1}_{nr}_{s}")
        experiment.attrib["random_seed"] = str(s)
        entity.attrib["quantity"] = str(nr)
        params.attrib["file_name"] =  f"data/train/unstr.csv"
        tree.write(argos_file)
        os.system("argos3 -c "+argos_file)