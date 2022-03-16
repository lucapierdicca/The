from pprint import pprint

import matplotlib.pyplot as plt
import pickle
from utils import Point, GaussianFilter
import numpy as np


def draw_map(map_path, wall_path, name):
    with open(map_path, "rb") as f:
        map_ground_truth = pickle.load(f)

    with open(wall_path, "rb") as f:
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

    # plt.axis('scaled')
    plt.axis('equal')
    plt.axis('off')
    plt.savefig(name)
    # plt.show()


def draw_map_test_trajectories(robot_position, y_pred):
    with open("data/test/test_map_ground_truth.pickle", "rb") as f:
        map_ground_truth = pickle.load(f)

    with open("data/test/test_map_wall_boundary_vertices.pickle", "rb") as f:
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

    for rp, y in zip(robot_position, y_pred):
        c = map_ground_truth[y]["color"]
        circle = plt.Circle(rp, radius=0.05, edgecolor=None, facecolor=(c[0], c[1], c[2], 0.1))
        ax.add_patch(circle)

    # plt.axis('scaled')
    plt.axis('equal')
    plt.axis('off')
    # plt.savefig('test_map_traj_30.svg')
    plt.show()


def draw_map_train_trajectories(robot_position, y_pred):
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

    for rp, y in zip(robot_position, y_pred):
        c = map_ground_truth[y]["color"]
        circle = plt.Circle(rp, radius=0.1, edgecolor=None, facecolor=(c[0], c[1], c[2]), alpha=0.3)
        ax.add_patch(circle)

    # plt.axis('scaled')
    plt.axis('equal')
    plt.axis('off')
    # fig.savefig('plotcircles.png')
    plt.show()


def draw_belief_evo():
    with(open("data/test/1/exp_metrics_linear_geometricB_s.pickle", "rb")) as f:
        metrics = pickle.load(f)

    fig, ax = plt.subplots()

    w, h = 50, 50
    for i, simulation_step in enumerate(metrics[0]["belief_evo"][:1000]):
        p = [50, 50 + i * 2 * h]
        for j, belief_step in enumerate(simulation_step):
            for k, proba in enumerate(belief_step):
                ax.add_patch(plt.Rectangle(p, w, h, edgecolor=None, fill=True, facecolor=(1.0, 0.0, 0.0, proba)))
                p[0] = p[0] + w
            p[0] = p[0] + w

    # plt.axis('scaled')
    plt.autoscale(False)
    # plt.axis('square')
    plt.axis('equal')
    plt.axis('off')
    # plt.ylim([0,10000])
    plt.show()


def draw_obs_evo(run_metrics):
    all_y = []
    for run in run_metrics:
        x, y = [], []
        for i, step in enumerate(run["obs_evo"]):
            y += step[0]
            x += [i, i + 0.25, i + 0.50, i + 0.75]

        all_y.append(y)
    print(len(all_y))

    all_y = np.array(all_y)
    all_y = np.mean(all_y, axis=0)

    fig, ax = plt.subplots()
    plt.ylim([0.0, 1.0])
    ax.scatter(x, all_y, s=0.1)
    plt.show()


def draw_belief_evo_in_time(run_metric):
    x, y = [], []

    w = 0
    for i, b in enumerate(run_metric["belief_evo"]):

        y += b
        x += [w, w + 2.5, w + 5.0, w + 7.5]

        w = i*50

    print(len(y))

    fig, ax = plt.subplots()
    plt.ylim([0.0, 1.0])
    ax.scatter(x, y, s=0.5)
    plt.show()

def draw_pred_evo_in_time(run_metric):
    x, y = [], []

    w = 0
    for i, y_p in enumerate(run_metric["y_pred"]):

        y += [y_p]
        x += [i]

    print(len(y))

    fig, ax = plt.subplots()
    #plt.ylim([0.0, 1.0])
    ax.scatter(x, y, s=0.5)
    plt.show()

def draw_map_test_outlier_obs(robot_positions, robot_orientations, measurements, features):
    with open("data/test/test_map_ground_truth.pickle", "rb") as f:
        map_ground_truth = pickle.load(f)

    with open("data/test/test_map_wall_boundary_vertices.pickle", "rb") as f:
        map_wall_boundary_vertices = pickle.load(f)

    fig, ax = plt.subplots()

    origin = plt.Circle([0.0,0.0], radius=0.1, facecolor="black")
    ax.add_patch(origin)

    for boundary_id, vertices in map_wall_boundary_vertices.items():
        vertices = [[v[0], v[1]] for v in vertices]
        boundary = plt.Polygon(vertices, closed=True, fill=None, edgecolor='black')

        ax.add_patch(boundary)

    for class_lbl, poly_data in map_ground_truth.items():
        for vertices in poly_data["poly_boundary_vertices"]:
            boundary = plt.Polygon(vertices, closed=True, fill=True, edgecolor=None,
                                   facecolor=poly_data["color"], alpha=0.3)
            ax.add_patch(boundary)

    for rp,ro,z,f in zip(robot_positions, robot_orientations, measurements, features):

        # pprint(rp)
        # pprint(ro)
        # pprint(z)
        # pprint(f)

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




if __name__ == '__main__':
    from utils import loadDataset

    with(open("data/test/1/exp_metrics_linear_geometricB_s.pickle", "rb")) as f:
        metrics = pickle.load(f)

    # draw_pred_evo_in_time(metrics[0])

    # trajectory,y_pred = [],[]
    # for run in metrics:
    #     trajectory += [[x[0], x[1] ]for x in run["x_pred"]]
    #     y_pred += run["y_pred"]
    #
    # print(len(y_pred))
    # draw_map_test_trajectories(trajectory ,y_pred)


    # path = "data/test/1/exp_metrics_linear_geometricB_s.pickle"
    # with(open(path, "rb")) as f:
    #     run_metrics = pickle.load(f)

    robot_positions, robot_orientations, measurements, features = [],[],[],[]

    for run_metric in metrics:
        for o in run_metric["outliers_data"]:
            #print(o["feature"])
            robot_positions.append(o["robot_position"])
            robot_orientations.append(o["robot_orientation"])
            measurements.append(o["z"])
            features.append(o["feature"])

    print(len(robot_positions))
    s, e = 0,2000
    draw_map_test_outlier_obs(robot_positions[s:e],
                              robot_orientations[s:e],
                              measurements[s:e],
                              features[s:e])

