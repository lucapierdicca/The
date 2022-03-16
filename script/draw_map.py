from pprint import pprint

import matplotlib.pyplot as plt
import pickle
from utils import Point, GaussianFilter, Classifier
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
            r_p = np.array([[d["distance"]/100*np.cos(angle)],
                            [d["distance"]/100*np.sin(angle)]])
            w_p = np.array([[rp[0]],[rp[1]]]) + w_R_r@r_p
            ray = plt.Line2D([rp[0],w_p[0]],[rp[1],w_p[1]],
                             linewidth=0.2,
                             antialiased=True,
                             color=(1.0,0.0,0.0,1.0))
            #ax.add_line(ray)

    # plt.axis('scaled')
    plt.axis('equal')
    plt.axis('off')
    # plt.savefig(name)
    plt.show()



if __name__ == '__main__':
    from utils import loadDataset

    #draw_map_templates()

    with(open("data/test/1/exp_metrics_linear_geometricB.pickle", "rb")) as f:
        metrics = pickle.load(f)

    draw_pred_evo_in_time(metrics[0])

    trajectory,y_pred = [],[]
    for run in metrics:
        trajectory += [[x[0], x[1] ]for x in run["x_pred"]]
        y_pred += run["y_pred"]

    print(len(y_pred))
    draw_map_test_trajectories(trajectory ,y_pred)


