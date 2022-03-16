from pprint import pprint

import matplotlib.pyplot as plt
import pickle
import numpy as np

from script.map import Map
from script.utils import Point


def draw_map(map):
    fig, ax = plt.subplots()

    for boundary_id, vertices in map.wall_boundary_vertices.items():
        vertices = [[v[0], v[1]] for v in vertices]
        boundary = plt.Polygon(vertices, closed=True, fill=None, edgecolor='black')

        ax.add_patch(boundary)

    for class_lbl, poly_data in map.ground_truth.items():
        for vertices in poly_data["poly_boundary_vertices"]:
            boundary = plt.Polygon(vertices, closed=True, fill=True, edgecolor=None,
                                   facecolor=poly_data["color"], alpha=0.3)
            ax.add_patch(boundary)

    # plt.axis('scaled')
    plt.axis('equal')
    plt.axis('off')
    #plt.savefig(name)
    plt.show()

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

def draw_map_step(dataset, map=None, interval=None):
    fig, ax = plt.subplots()

    if map is not None:
        for boundary_id, vertices in map.wall_boundary_vertices.items():
            vertices = [[v[0], v[1]] for v in vertices]
            boundary = plt.Polygon(vertices, closed=True, fill=None, edgecolor='black')

            ax.add_patch(boundary)

        for class_lbl, poly_data in map.ground_truth.items():
            for vertices in poly_data["poly_boundary_vertices"]:
                boundary = plt.Polygon(vertices, closed=True, fill=True, edgecolor=None,
                                       facecolor=poly_data["color"], alpha=0.3)
                ax.add_patch(boundary)


    if interval is None:
        my_range = range(0,len(dataset),1)
    else:
        my_range = range(*interval)

    robot_positions, robot_orientations, measurements = [], [], []
    for i in my_range:
        robot_positions.append([dataset[i]["x"], dataset[i]["y"]])
        robot_orientations.append(dataset[i]["theta"])
        measurements.append(dataset[i]["world_model_long"])


    origin = plt.Circle((0.0,0.0), radius=0.02, edgecolor=None, facecolor="black", zorder=3)
    for rp,ro,z in zip(robot_positions, robot_orientations, measurements):
        w_R_r = np.array([[np.cos(ro), -np.sin(ro)],
                          [np.sin(ro), np.cos(ro)]])

        r_face = np.array([[rp[0]],[rp[1]]]) + w_R_r@np.array([[0.12],
                                                               [0.0]])
        face = plt.Circle(r_face, radius=0.02, edgecolor=None,
                          facecolor="black", zorder=3)

        body = plt.Circle(rp, radius=0.12, edgecolor=None,
                          facecolor=(0.0, 0.0, 1.0, 1.0), zorder=2)


        for angle,d in z.items():
            r_p = np.array([[(d["distance"]+12)/100*np.cos(angle)],
                            [(d["distance"]+12)/100*np.sin(angle)]])
            w_p = np.array([[rp[0]],[rp[1]]]) + w_R_r@r_p
            ray = plt.Line2D([rp[0],w_p[0]],[rp[1],w_p[1]],
                             linewidth=0.2,
                             antialiased=True,
                             color=(1.0,0.0,0.0,1.0) if d["occluded"] == 0 else (0.0,0.0,1.0,1.0), zorder=1)
            ax.add_line(ray)

            if "occluding_robot_data" in d:
                r_p2 = np.array([[(d["occluding_robot_data"][1]+24)/100*np.cos(d["occluding_robot_data"][0])],
                                [(d["occluding_robot_data"][1]+24)/100*np.sin(d["occluding_robot_data"][0])]])
                w_p2 = np.array([[rp[0]],[rp[1]]]) + w_R_r@r_p2
                ax.add_patch(plt.Circle(w_p2, radius=0.12, edgecolor="black",
                                        facecolor=(1.0,0.0,0.0,0.0), zorder=3, linewidth=0.1))

                w_p3 = np.array([[rp[0]],[rp[1]]]) + w_R_r@d["occluding_robot_data"][2]*1/100
                ray2 = plt.Line2D([rp[0],w_p3[0]],[rp[1],w_p3[1]],
                                 linewidth=0.8,
                                 antialiased=True,
                                 color=(1.0,0.0,1.0,1.0), zorder=1)
                ax.add_line(ray2)

        ax.add_patch(origin)
        ax.add_patch(body)
        ax.add_patch(face)

    #plt.axis('scaled')
    plt.axis('equal')
    plt.axis('off')
    # plt.savefig(name)
    plt.show()

def draw_map_templates(robot_positions, robot_orientations, measurements):
    with open("data/train/train_map_2_ground_truth.pickle", "rb") as f:
        map_ground_truth = pickle.load(f)

    with open("data/train/train_map_2_wall_boundary_vertices.pickle", "rb") as f:
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
            ax.add_line(ray)

    # plt.axis('scaled')
    plt.axis('equal')
    plt.axis('off')
    # plt.savefig(name)
    plt.show()



if __name__ == '__main__':
    pass



