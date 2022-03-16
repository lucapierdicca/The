import pickle
from collections import Counter

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from utils import HybridFilter, BabySitter, Point, load_dataset, CustomPGM
import random
from pprint import pprint
import numpy as np
import os

n_robots = 20
n_experiments = len([ed for ed in os.listdir(f"data/test/{n_robots}/experiments/")
                     if not ed.startswith('.')
                     and not "experiments" in ed])


filter = HybridFilter()
filter.estimateTransitionModel(dataset=None)
filter.estimateObservationModel(dataset=None)
filter.estimateSpatialModel(dataset=None)
print(filter.clf.classes_)

filter.topology_model = filter.transition_model


# random.seed(3215)
# robot_ids = []
# experiments = []
# for exp_idx in tqdm(range(n_experiments)):
#     with open(f"data/test/{n_robots}/experiments/{str(exp_idx+1)}/experiment.pickle","rb") as f:
#         experiment = pickle.load(f)
#
#     belief = np.ones((filter.state_dim, 1)) # GCSVI
#     r_id = random.choice(range(n_robots))
#     robot_ids.append(r_id)
#     for h_step in experiment:
#         posterior = filter.update_istantaneous(h_step, r_id)
#
#         prediction = filter.predict(posterior)
#         h_step[r_id]["pred_class"] = prediction
#
#     experiments.append(experiment)
#
#
# print(robot_ids)
# BabySitter().display_classification_metrics(experiments, robot_ids)
# BabySitter().draw_map_predicted_trajectories(experiments, robot_ids)



# random.seed(3215)
# robot_ids = []
# experiments = []
# for exp_idx in tqdm(range(n_experiments)):
#     with open(f"data/test/{n_robots}/experiments/{str(exp_idx+1)}/experiment.pickle","rb") as f:
#         experiment = pickle.load(f)
#
#     belief = np.ones((filter.state_dim, 1)) # GCSVI
#     r_id = random.choice(range(n_robots))
#     robot_ids.append(r_id)
#     for h_step in experiment:
#         belief = filter.update(belief, h_step, r_id)
#
#         prediction = filter.predict(belief)
#         h_step[r_id]["pred_class"] = prediction
#
#     experiments.append(experiment)
#
#
# print(robot_ids)
# BabySitter().display_classification_metrics(experiments, robot_ids)
# BabySitter().draw_map_predicted_trajectories(experiments, robot_ids)




# random.seed(3215)
# robot_ids = []
# future = 3
# experiments = []
# for exp_idx in tqdm(range(n_experiments)):
#     with open(f"data/test/{n_robots}/experiments/{str(exp_idx+1)}/experiment.pickle","rb") as f:
#         experiment = pickle.load(f)
#
#     belief = np.ones((filter.state_dim, 1)) #GCSVI
#     r_id = random.choice(range(n_robots))
#     robot_ids.append(r_id)
#     for i in range(len(experiment)-future):
#         belief, total_belief, beta_t = filter.update_lag(belief, experiment[i], experiment[i+1:i+1+future], r_id)
#
#         prediction = filter.predict(total_belief)
#         experiment[i][r_id]["pred_class"] = prediction
#
#     experiments.append(experiment)
#
#
# print(robot_ids)
# BabySitter().display_classification_metrics(experiments, robot_ids)
# BabySitter().draw_map_predicted_trajectories(experiments, robot_ids)





# graph = CustomPGM(filter)
# random.seed(3215)
# belief = [1.0/filter.state_dim]*filter.state_dim # G,C,S,V,I
# belief = np.array(belief, dtype=float).reshape((5,1))
# robot_ids = []
# for exp in tqdm(experiments):
#     r_id = random.choice(range(n_robots))
#     robot_ids.append(r_id)
#     for h_step in exp:
#         belief = graph.update(belief, h_step, r_id, 200)
#         prediction = filter.predict(belief)
#         h_step[r_id]["pred_class"] = prediction
#
#
#
# print(robot_ids)
# BabySitter().display_classification_metrics(experiments, robot_ids)
# BabySitter().display_comparison_metrics(experiments)
# BabySitter().display_mean_wrong_seq_info(experiments)
# BabySitter().draw_map_predicted_trajectories(experiments)



# random.seed(3215)
# range_inference_th = 100
# range_score_th = 100
# robot_ids = []
# experiments = []
# for exp_idx in tqdm(range(n_experiments)):
#     with open(f"data/test/{n_robots}/experiments/{str(exp_idx+1)}/experiment.pickle","rb") as f:
#         experiment = pickle.load(f)
#
#     beliefs = [np.ones((5,1))*1.0/filter.state_dim] * n_robots  # G,C,S,V,I
#     r_id = random.choice(range(n_robots))
#     robot_ids.append(r_id)
#     for h_step in experiment:
#         total_belief = filter.update_neighbors(beliefs, h_step, r_id, range_inference_th)
#         prediction = filter.predict(total_belief)
#         h_step[r_id]["pred_class"] = prediction
#
#         # print()
#         # for k,data in h_step[r_id]["graph"].items():
#         #     print(k, data["adj"])
#         # BabySitter().draw_h_step_graph(h_step,r_id)
#
#     experiments.append(experiment)
#
# print(robot_ids)
# #
# BabySitter().display_classification_metrics(experiments, robot_ids, range_score_th)
# # #BabySitter().draw_map_predicted_trajectories(experiments, robot_ids, range_score_th)
# BabySitter().save_map_h_step_graph([experiments[9]],[robot_ids[9]], range_score_th, graph_type="local")

# file_name = f"data/test/{n_robots}/experiments/experiments_debug.pickle"
# with open(file_name, "wb") as f:
#     pickle.dump(experiments, f)
# file_name = f"data/test/{n_robots}/experiments/robot_ids.pickle"
# with open(file_name, "wb") as f:
#     pickle.dump(robot_ids, f)

# range_th = 200
# n_robots = 5
# file_name = f"data/test/{n_robots}/experiments/experiments_debug.pickle"
# with open(file_name,"rb") as f:
#     experiments = pickle.load(f)
#
# file_name = f"data/test/{n_robots}/experiments/robot_ids.pickle"
# with open(file_name,"rb") as f:
#     robot_ids = pickle.load(f)
#


random.seed(3215)
range_inference_th = 50
n_iteration = 20
spy_robot_ids = []
experiments = []
experiments_graphs = []

for exp_idx in tqdm(range(n_experiments)):

    with open(f"data/test/{n_robots}/experiments/{str(exp_idx+1)}/experiment.pickle","rb") as f:
        experiment = pickle.load(f)

    spy_robot_id = random.choice(range(n_robots))
    experiment_graphs = []

    beliefs = [np.ones((5,1))*1.0/filter.state_dim] * n_robots  # G,C,S,V,I
    for h_step in experiment:
        spy_robot_belief, h_step_graphs = filter.update_loop(beliefs,
                                                             h_step,
                                                             spy_robot_id,
                                                             range_inference_th,
                                                             n_iteration)
        prediction = filter.predict(spy_robot_belief)
        h_step[spy_robot_id]["pred_class"] = prediction

        experiment_graphs.append(h_step_graphs)

        #BabySitter().draw_map_h_step_graph(h_step, spy_robot_id, h_step_graphs)

    spy_robot_ids.append(spy_robot_id)
    experiments.append(experiment)
    experiments_graphs.append(experiment_graphs)


BabySitter().display_classification_metrics(experiments, spy_robot_ids, experiments_graphs)
#BabySitter().draw_map_predicted_trajectories(experiments, spy_robot_ids)
#BabySitter().save_map_h_step_graph([experiments[0]], [spy_robot_ids[0]], [experiments_graphs[0]], belief_type="filter_belief")

# # ---------------------------
# levels = []
# for exp in experiments:
#     for h_step in exp:
#         for r in h_step:
#             pred = filter.classifier.id_to_classlbl[np.argmax(r["filter_belief"])]
#             true = r["true_class"]
#             if pred != true:
#                 levels.append(np.max(r["filter_belief"]))
#
# fig, ax = plt.subplots()
# ax.set(xticks=np.arange(0, 1.1, step=0.1))
# e = ax.hist(levels, bins=[0.0+i*0.1 for i in range(11)], edgecolor="white",
#             weights=[1.0/len(levels)]*len(levels))
#
# print(f"Totale numero errori: {len(levels)} su {n_robots*30000}")
# print("Proporzione: ", e[0])
# print("Frequenza: ", e[0]*len(levels))
#
# plt.show()
#
#
#
# #---------------------------
# n_total_steps = 30000
# n_graphs = 0
# graphs_n_neighbors = []
# graphs_neighbors_data = []
# for exp_idx, (r_id, exp_graphs) in enumerate(zip(spy_robot_ids, experiments_graphs)):
#     for h_step_idx, h_step_graphs in enumerate(exp_graphs):
#         for graph in h_step_graphs:
#             if r_id in graph["connectivity"]:
#                 n_graphs += 1
#                 graphs_n_neighbors.append(len(graph["connectivity"]) - 1)
#                 neighbors_data = []
#                 for k in list(graph["connectivity"].keys()):
#                     if k != r_id:
#                         neighbors_data.append({"belief": experiments[exp_idx][h_step_idx][k]["filter_belief"],
#                                                "true_class": experiments[exp_idx][h_step_idx][k]["true_class"]})
#                 graphs_neighbors_data.append(neighbors_data)
#
# assert n_graphs == len(graphs_n_neighbors) == len(graphs_neighbors_data)
#
# print(n_graphs, n_total_steps, n_graphs/n_total_steps)
#
# fig, ax = plt.subplots()
# c = Counter(graphs_n_neighbors)
# heights = [0]*max(graphs_n_neighbors)
# for k,v in c.items():
#     heights[k-1] = v
# ax.set(xticks=range(1,max(graphs_n_neighbors)+1,1))
# ax.bar(range(1,max(graphs_n_neighbors)+1,1), heights)
#
# print("Frequenza: ", heights)
# plt.show()
#
#
#
# #------------------------------
# neighbor_levels_wrong = []
# neighbor_levels_right = []
# for graph in graphs_neighbors_data:
#     for neighbor in graph:
#         if filter.classifier.id_to_classlbl[np.argmax(neighbor["belief"])] != neighbor["true_class"]:
#             neighbor_levels_wrong.append(np.max(neighbor["belief"]))
#         else:
#             neighbor_levels_right.append(np.max(neighbor["belief"]))
#
# assert sum(graphs_n_neighbors) == len(neighbor_levels_right)+len(neighbor_levels_wrong)
#
# print(f"Neighbor totali: {sum(graphs_n_neighbors)}")
# print(f"Neighbor belief corretti: {len(neighbor_levels_right)}")
# print(f"Neighbor belief sbagliati: {len(neighbor_levels_wrong)}")
#
#
# fig, ax = plt.subplots(1,2)
# ax[0].set(xticks=np.arange(0, 1.1, step=0.1))
# if len(neighbor_levels_wrong) > 0:
#     e = ax[0].hist(neighbor_levels_wrong, bins=[0.0+i*0.1 for i in range(11)], edgecolor="white",
#                 weights=[1.0/len(neighbor_levels_wrong)]*len(neighbor_levels_wrong))
#     print("Proporzione: ", e[0])
#     print("Frequenza: ", e[0]*len(neighbor_levels_wrong))
#
# ax[1].set(xticks=np.arange(0, 1.1, step=0.1))
# if len(neighbor_levels_right) > 0:
#     e = ax[1].hist(neighbor_levels_right, bins=[0.0+i*0.1 for i in range(11)], edgecolor="white",
#                      weights=[1.0/len(neighbor_levels_right)]*len(neighbor_levels_right))
#     print("Proporzione: ", e[0])
#     print("Frequenza: ", e[0]*len(neighbor_levels_right))
#
# plt.show()

cases = []
for exp_idx, (r_id, exp_graphs) in enumerate(zip(spy_robot_ids, experiments_graphs)):
    for h_step_idx, h_step_graphs in enumerate(exp_graphs):
        for graph_idx, graph in enumerate(h_step_graphs):
            if r_id in graph["connectivity"]:
                cases.append((exp_idx,h_step_idx,r_id,graph_idx))

print(len(cases))

separate = {}
for c in cases:
    if c[0] not in separate:
        separate[c[0]] = {"h_steps":[],
                          "h_steps_o_idx":[],
                          "r_id":c[2],
                          "h_step_graphs":[]}

    separate[c[0]]["h_steps"].append(experiments[c[0]][c[1]])
    separate[c[0]]["h_steps_o_idx"].append([c[1]])
    separate[c[0]]["h_step_graphs"].append(experiments_graphs[c[0]][c[1]])

a1,a2,a3,a4 = [],[],[],[]
for i in range(30):
    a1.append(separate[i]["h_steps"])
    a2.append(separate[i]["r_id"])
    a3.append(separate[i]["h_step_graphs"])
    a4.append(separate[i]["h_steps_o_idx"])

print(len(a1), len(a2), len(a3), len(a4))
print(type(a2[0]))

na1,na3,na4 = [],[],[]
for _ in a1:
    na1.append(len(_))
for _ in a3:
    na3.append(len(_))
for _ in a4:
    na4.append(len(_))
assert na1 == na3 == na4
print(sum(na1),sum(na3),sum(na4))

#BabySitter().save_map_h_step_graph(a1, a2, a3, a4, belief_type="total_belief")
BabySitter().draw_map_predicted_trajectories(a1,a2,a4)

y_true,y_pred = [],[]
for exp_idx,h_step_idx,r_id,_ in cases:
    y_true.append(experiments[exp_idx][h_step_idx][r_id]["true_class"])
    y_pred.append(experiments[exp_idx][h_step_idx][r_id]["pred_class"])

print(classification_report(y_true,
                            y_pred,
                            zero_division=0,
                            output_dict=False,
                            labels=list(filter.classifier.classlbl_to_id.keys())))

print(confusion_matrix(y_true,
                       y_pred,
                       labels=list(filter.classifier.classlbl_to_id.keys())))


random.seed(3215)
range_inference_th = 10
n_iteration = 20
spy_robot_ids = []
experiments = []
experiments_graphs = []

for exp_idx in tqdm(range(n_experiments)):

    with open(f"data/test/{n_robots}/experiments/{str(exp_idx+1)}/experiment.pickle","rb") as f:
        experiment = pickle.load(f)

    spy_robot_id = random.choice(range(n_robots))
    experiment_graphs = []

    beliefs = [np.ones((5,1))*1.0/filter.state_dim] * n_robots  # G,C,S,V,I
    for h_step in experiment:
        spy_robot_belief, h_step_graphs = filter.update_loop(beliefs,
                                                             h_step,
                                                             spy_robot_id,
                                                             range_inference_th,
                                                             n_iteration)
        prediction = filter.predict(spy_robot_belief)
        h_step[spy_robot_id]["pred_class"] = prediction

        experiment_graphs.append(h_step_graphs)

        #BabySitter().draw_map_h_step_graph(h_step, spy_robot_id, h_step_graphs)

    spy_robot_ids.append(spy_robot_id)
    experiments.append(experiment)
    experiments_graphs.append(experiment_graphs)

y_true,y_pred = [],[]
for exp_idx,h_step_idx,r_id in cases:
    y_true.append(experiments[exp_idx][h_step_idx][r_id]["true_class"])
    y_pred.append(experiments[exp_idx][h_step_idx][r_id]["pred_class"])

print(classification_report(y_true,
                            y_pred,
                            zero_division=0,
                            output_dict=False,
                            labels=list(filter.classifier.classlbl_to_id.keys())))

print(confusion_matrix(y_true,
                       y_pred,
                       labels=list(filter.classifier.classlbl_to_id.keys())))