import pickle
from tqdm import tqdm
from utils import HybridFilter, Point, Data
import os
import time
from collections import deque
from pprint import pprint
import numpy as np
import copy

n_robots = 20
n_experiments = len([ed for ed in os.listdir(f"data/test/{n_robots}/experiments/")
                     if not ed.startswith('.')
                     and not "experiments" in ed])
maxlen = 5
mem = deque(maxlen=maxlen)
load_model = True
filter = HybridFilter()
filter.estimateObservationModel(dataset=None)

print(f"Robots: {n_robots}")

for exp_idx in tqdm(range(n_experiments)):

    test = Data(f"data/test/{n_robots}/experiments/{str(exp_idx+1)}",
                "data/test/test_map_ground_truth.pickle")

    experiment = []
    while(True):
        tic = time.time()
        h_step = test.next(10)
        tac = time.time()
        #print("read ",tac-tic)
        if h_step == 0: break

        if len(h_step) > 0:

            # prendo il massimo degli occlusi durante la finestra temporale
            mem.append(h_step)
            h_step_copy = copy.deepcopy(h_step)
            occlusion_matrix = np.zeros((len(mem),len(h_step[0]["world_model_long"])),dtype=np.uint8)
            distance_matrix = np.zeros_like(occlusion_matrix, dtype=float)
            for robot_id in range(n_robots):
                for row,mem_h_step in enumerate(mem):
                    for col,(angle, reading_data) in enumerate(mem_h_step[robot_id]["world_model_long"].items()):
                        occlusion_matrix[row,col] = reading_data["occluded"]
                        distance_matrix[row,col] = reading_data["distance"]

                occ_colums = np.argwhere(np.sum(occlusion_matrix, axis=0) > 0).flatten()
                M = np.max(distance_matrix[:,occ_colums], axis=0).flatten()

                j = 0
                for idx,(angle,reading_data) in enumerate(h_step_copy[robot_id]["world_model_long"].items()):
                    if idx in occ_colums:
                        reading_data["distance"] = M[j]
                        j+=1

            # elimino gli occlusi
            # for robot_id in range(n_robots):
            #     k_to_del = []
            #     for k, data in h_step_copy[robot_id]["world_model_long"].items():
            #         if data["occluded"] == True:
            #             k_to_del.append(k)
            #
            #     for k in k_to_del:
            #         del h_step_copy[robot_id]["world_model_long"][k]



            for hs in h_step_copy:
                tic = time.time()
                z = filter.classifier.preProcess(hs['world_model_long'], 3)
                tac = time.time()
                #print("prepro: ", tac-tic)
                tic = time.time()
                feature = filter.extractFeature(z, handle_occlusion=False)
                tac = time.time()
                #print("feature: ", tac-tic)
                tic = time.time()
                feature_std = filter.scaler.transform([feature])  # standardize
                tac = time.time()
                #print("scaler: ", tac-tic)
                hs["feature_std"] = feature_std

                del hs["world_model_long"]

            experiment.append(h_step_copy)


    file_name = f"data/test/{n_robots}/experiments/{str(exp_idx+1)}/experiment_history_window_{maxlen}.pickle"
    with open(file_name, 'wb') as f:
        pickle.dump(experiment, f)

    test.close()
