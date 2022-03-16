from pprint import pprint
from scipy.spatial.distance import pdist, squareform
from utils import HybridFilter, BabySitter, Point
from itertools import product
import numpy as np

side = 200
range_bearing_th = 40
range_inference_th = 12
n_iteration = 100
spy_robot_id = 150

h_step = []


# posiziona i robot in modo random
#r_positions = np.random.random((r_num, 2)) * side
np.random.seed(3)
grid = np.mgrid[0:side:10, 0:side:10]
r_positions = np.vstack([grid[0].ravel(), grid[1].ravel()]).T
noise = np.random.random(r_positions.shape)
r_positions = noise + r_positions.astype('float64')
r_num = len(r_positions)
print(f"Robots: {r_num}")

for i in range(r_num):
    h_step.append({"fb_id": f"fb_{i}",
                   "x": r_positions[i, 0],
                   "y": r_positions[i, 1],
                   "rab_readings": []})

# calcola i rab_readings
d = pdist(r_positions)
r_distances = squareform(d)

for row in range(r_num):
    for col in range(r_num):
        if r_distances[row,col] <= range_bearing_th and r_distances[row,col] > 0.0:
            h_step[row]["rab_readings"].append({"id": col,
                                                "range": r_distances[row,col]})

# inizializza il filtro
filter = HybridFilter()

# inizializza i belief
beliefs = []  # G,C,S,V,I
for i in range(r_num):
    b = np.ones((5,1))
    if r_positions[i][0] <= 100:

        if np.random.random() <= 0.60:
            w = 1000
            b[0,0] = w
        else:
            w = 1000
            b[1,0] = w
    else:
        if np.random.random() <= 0.60:
            w = 1000
            b[1,0] = w
        else:
            w = 1000
            b[0,0] = w

    b /= np.sum(b)
    beliefs.append(b)

print(beliefs[spy_robot_id])


# manually saving the filter_belief
for i in range(r_num):
    h_step[i]["belief"] = beliefs[i]

# fai inferenza update_loop
spy_robot_belief, h_step_graphs = filter._update_loop(beliefs, h_step, spy_robot_id, range_inference_th, n_iteration, order_type="sequential")

print(beliefs[spy_robot_id])

# for g in h_step_graphs:
#     pprint(g["connectivity"][spy_robot_id]["total_belief_history"][-1])
#     for k,v in g["connectivity"].items():
#         print(k, v["adj"])



# brute-force MAP of the joint distro and marginal of the spy-robot variable
# resetta i belief
# beliefs = []  # G,C,S,V,I
# for i in range(r_num):
#     b = np.ones((5,1))
#     if i == 0:
#         w = 1000
#         b[1,0] = w
#         b /= np.sum(b)
#     else:
#         w = 1000
#         b[3,0] = w
#         b /= np.sum(b)
#
#     beliefs.append(b)
#
#
# def prob(multivariable_value, edges):
#     p = 1.0
#     for bel, variable_value in zip(beliefs, multivariable_value):
#         p *= bel[variable_value,0]
#     for e in edges:
#         p *= filter.topology_model[multivariable_value[e[0]],multivariable_value[e[1]]]
#
#     assert p > 0.0, "underflow!"
#
#     return p


# MAP of the joint distro
# multivariable_values = product(range(5), repeat=len(r_positions))
# edges = h_step_graphs[0]["edges"]
#
# max, argmax = 0.0, None
# cum_sum = 0.0
# for multivariable_value in multivariable_values:
#     v = prob(multivariable_value, edges)
#     cum_sum += v
#     if v > max:
#         max = v
#         argmax = multivariable_value
#
# max /= cum_sum
#
# print(f"joint_argmax: {argmax} joint_max: {max}")

# spy-robot marginal
# p_x = {0:0.0,1:0.0,2:0.0,3:0.0,4:0.0}
# for v, a in zip(values, multivariable_values):
#     p_x[a[spy_robot_id]] +=v
#
# pprint(p_x)

# rendering
BabySitter()._draw_map_h_step_graph(h_step, spy_robot_id, h_step_graphs, type="belief")
BabySitter()._draw_map_h_step_graph(h_step, spy_robot_id, h_step_graphs, type="total")