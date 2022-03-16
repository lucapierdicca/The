from pprint import pprint

import matplotlib.pyplot as plt
import pickle
import numpy as np

#from script.map import Map
from utils import Point, Classifier, load_dataset


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

def draw_map_test_trajectories(experiments, orientation=False, color=None, prediction=False, neighbors=False):

    if color not in ["belief","prediction","filter"]:
        raise ValueError(f"Color cannot be {color}")

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
            
    circle_radius = 0.02
    face_radius = 0.008
    circles = []

    if "belief_pr_seq" in experiments[0]:
        for e in experiments:
            for yt,xp,yp,bpr,like,bup,fil in zip(e["y_true"],e["position"],e["y_pred"],e["belief_pr_seq"],e["likelihood_seq"],e["belief_up_seq"],e["filter"]):
                if fil == 1:

                    c = map_ground_truth[yp]["color"] if yp != -1 else [0.0,0.0,0.0,0.5]
                    if color == "belief":
                        circle = plt.Circle(xp, radius=circle_radius, edgecolor=None, facecolor=(c[0], c[1], c[2], 0.5) if max(bup)>= 0.6 else (0.0,0.0,0.0,1.0))
                    elif color == "prediction" :
                        circle = plt.Circle(xp, radius=circle_radius, edgecolor=None, facecolor=(c[0], c[1], c[2], 0.5))
                    elif color == "filter":
                        y_true_id = classifier.classlbl_to_id[yt]
                        max_bpr, max_like, max_bup = bpr.index(max(bpr)), like.index(max(like)),bup.index(max(bup))
                        pattern = [1 if max_bpr == y_true_id else 0,
                                   1 if max_like == y_true_id else 0,
                                   1 if max_bup == y_true_id else 0]

                        if pattern == [1,0,1]:
                            f = "yellow"
                        elif pattern == [1,0,0]:
                            f = "orange"
                        elif pattern == [0,0,0]:
                            f = "red"
                        elif pattern == [0,1,0]:
                            f = "blue"
                        elif pattern == [0,1,1]:
                            f = "cyan"
                        elif pattern == [0,0,1] or pattern == [1,1,0]:
                            f = "black"
                        else:
                            f = "green"
                        circle = plt.Circle(xp, radius=circle_radius, edgecolor=None, facecolor=f)
                    circles.append(circle)
    else:
        for e in experiments:
            for yt,xp,yp,fil in zip(e["y_true"],e["position"],e["y_pred"],e["filter"]):
                if fil == 1:
                    c = map_ground_truth[yp]["color"]
                    if color == "prediction" :
                        circle = plt.Circle(xp, radius=circle_radius, edgecolor=None, facecolor=(c[0], c[1], c[2], 0.5))
                    circles.append(circle)


    faces = []
    for e in experiments:
        for xp,o,fil in zip(e["position"],e["orientation"],e["filter"]):
            if fil == 1:
                w_R_r = np.array([[np.cos(o), -np.sin(o)],
                                  [np.sin(o), np.cos(o)]])

                r_face = np.array([[xp[0]],[xp[1]]]) + w_R_r@np.array([[0.02],
                                                                       [0.0]])
                face = plt.Circle(r_face, radius=face_radius, edgecolor=None,
                                  facecolor="black")
                faces.append(face)


    for c in circles:
        ax.add_patch(c)

    if orientation == True:
        for f in faces:
            ax.add_patch(f)

    if prediction == True:
        for e in experiments:
            for xp,yp,fil in zip(e["position"],e["y_pred"],e["filter"]):
                if fil == 1:
                    ax.text(*xp, yp, ha='center', va='center', size=5.0)

    if neighbors == True:
        for e in experiments:
            for xp,nei,ty,fil in zip(e["position"],e["rab_readings"],e["type"],e["filter"]):
                if fil == 1:
                    s = "-1"
                    sorted_neighbors = sorted(nei, key=lambda d: d["range"])
                    if len(sorted_neighbors) != 0:
                        s = str(sorted_neighbors[0]["range"])+"_"+str(len(sorted_neighbors))+"_"+ty
                    ax.text(*xp, s, ha='center', va='center', size=5.0)


    # plt.axis('scaled')
    plt.axis('equal')
    plt.axis('off')
    #plt.savefig('filter_test_1_customtrmo_errors_orie.svg')
    plt.show()

def draw_map_train_trajectories():
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

    # for step in dataset:
    #     c = map_ground_truth[step["true_class"]]["color"]
    #     circle = plt.Circle((step["x"],step["y"]), radius=0.1, edgecolor=None, facecolor=(c[0], c[1], c[2]), alpha=0.3)
    #     ax.add_patch(circle)

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

def draw_test_scatter_belief_sequence(y_pred, y_true, belief_pr=None, likelihood=None, belief_up=None):
    with open("data/test/test_map_ground_truth.pickle", "rb") as f:
        map_ground_truth = pickle.load(f)

    fig, ax = plt.subplots()

    print(len(y_pred))

    xp = [1.0+(100*i)for i in range(len(y_pred))]
    yp = [2.5]*len(y_pred)
    cp = [map_ground_truth[y]["color"] for y in y_pred]

    xt = [1.0+(100*i)for i in range(len(y_true))]
    yt = [0]*len(y_true)
    ct = [map_ground_truth[y]["color"] for y in y_true]

    xd = [1.0+(100*i)for i in range(len(y_true))]
    yd = [-2.5]*len(y_true)
    cd = ["g" if yp==yt else "r" for yp,yt in zip(y_pred,y_true)]

    ax.autoscale(enable=None)
    ax.set(xlim=(0.0,20000), ylim=(-10,10))
    ax.scatter(xp,yp,s=1.5,c=cp)
    ax.scatter(xt,yt,s=1.5,c=ct)
    ax.scatter(xd,yd,s=1.5,c=cd)
    
    for posx,posy,bp,li,bu in zip(xd,yd,belief_pr, likelihood,belief_up):
        ax.text(posx,posy-2, f"[{bp[0]:.2f} {bp[1]:.2f} {bp[2]:.2f} {bp[3]:.2f} {bp[4]:.2f}]", ha='center', va='center', size=3.0)
        ax.text(posx,posy-2.2, f"[{li[0]:.2f} {li[1]:.2f} {li[2]:.2f} {li[3]:.2f} {li[4]:.2f}]", ha='center', va='center', size=3.0)
        ax.text(posx,posy-2.4, f"[{bu[0]:.2f} {bu[1]:.2f} {bu[2]:.2f} {bu[3]:.2f} {bu[4]:.2f}]", ha='center', va='center', size=3.0)

    print(ax.get_autoscale_on())


    # plt.savefig('test_map_traj_30.svg')
    plt.show()

def draw_scatter_test_filtervsnofilter_sequences(experiments):
    filter_scatter_sequences  = []
    for i,e in enumerate(experiments):
        x = [1.0 + (20*j) for j in range(len(e["y_true"]))]
        y = [1.0 + (2*i)]*len(e["y_true"])
        p = list(e["x_pred"])
        c = []

        for k,(yt,bpr,like,bup) in enumerate(zip(e["y_true"],e["belief_pr_seq"],e["likelihood_seq"],e["belief_up_seq"])):
            y_true_id = classifier.classlbl_to_id[yt]
            max_bpr, max_like, max_bup = bpr.index(max(bpr)), like.index(max(like)),bup.index(max(bup))
            pattern = [1 if max_bpr == y_true_id else 0,
                       1 if max_like == y_true_id else 0,
                       1 if max_bup == y_true_id else 0]
            if pattern == [1,0,1]:
                c.append("yellow")
            elif pattern == [1,0,0]:
                c.append("orange")
            elif pattern == [0,0,0]:
                c.append("red")
            elif pattern == [0,1,0]:
                c.append("blue")
            elif pattern == [0,1,1]:
                c.append("cyan")
            elif pattern == [0,0,1] or pattern == [1,1,0]:
                c.append("black")
            else:
                c.append("green")

        filter_scatter_sequences.append(dict(x=x,y=y,c=c))


    nonfilter_scatter_sequences = []
    for i,e in enumerate(experiments):
        x = [1.0 + (20*j) for j in range(len(e["y_true"]))]
        y = [1.0 + ((2*i)-0.5)]*len(e["y_true"])
        c = []
        for k,(yt,pos) in enumerate(zip(e["y_true"],e["posterior_seq"])):
            y_true_id = classifier.classlbl_to_id[yt]
            max_pos = pos.index(max(pos))
            pattern = 1 if max_pos == y_true_id else 0

            if pattern == 1:
                c.append("green")
            else:
                c.append("red")

        nonfilter_scatter_sequences.append(dict(x=x,y=y,c=c))


    nfred, fyellow, fcyan, fgreen = 0,0,0,0
    for fs, nfs in zip(filter_scatter_sequences, nonfilter_scatter_sequences):
        for cfs, cnfs in zip(fs["c"],nfs["c"]):
            if cnfs ==  "red" and cfs in ["yellow", "cyan", "green"]:
                nfred+=1
                if cfs == "yellow":
                    fyellow+=1
                if cfs == "cyan":
                    fcyan+=1
                if cfs == "green":
                    fgreen+=1

    print(f"tot: {((fyellow+fcyan+fgreen)/nfred):.2f} - rwr: {(fyellow/nfred):.2f} - wrr: {(fcyan/nfred):.2f} - rrr: {(fgreen/nfred):.2f}")


    nfgreen, forange, fred, fblue = 0,0,0,0
    for fs, nfs in zip(filter_scatter_sequences, nonfilter_scatter_sequences):
        for cfs, cnfs in zip(fs["c"],nfs["c"]):
            if cnfs ==  "green" and cfs in ["orange","red","blue"]:
                nfgreen+=1
                if cfs == "orange":
                    forange+=1
                if cfs == "red":
                    fred+=1
                if cfs == "blue":
                    fblue+=1


    print(f"tot: {((forange+fred+fblue)/nfgreen):.2f} - rww: {(forange/nfgreen):.2f} - wrw: {(fblue/nfgreen):.2f} - www: {(fred/nfgreen):.2f}")

    fig, ax = plt.subplots()

    for ss in filter_scatter_sequences:
        ax.scatter(ss["x"], ss["y"], s=5.5, c=ss["c"])

    for ss in nonfilter_scatter_sequences:
        ax.scatter(ss["x"], ss["y"], s=5.5, c=ss["c"])

    ax.autoscale(enable=None)
    ax.set(xlim=(0.0,10000), ylim=(0,70))

    plt.show()

if __name__ == '__main__':

    classifier = Classifier()

    n_robots = 20
    file_name = "exp_metrics_svc.pickle"

    with open(f"data/test/{n_robots}/{file_name}","rb") as f:
        experiments = pickle.load(f)

    experiments_copy = list(experiments)
    for e in experiments_copy:
        e["filter"] = [1]*len(e["y_true"])

    #filter
    for e in experiments_copy:
        e["filter"] = []
        for yt,yp in zip(e["y_true"], e["y_pred"]):
            if yp != yt:
                e["filter"].append(1)
            else:
                e["filter"].append(0)

    draw_map_test_trajectories(experiments_copy,
                               orientation=False,
                               neighbors=True,
                               prediction=False,
                               color="prediction")

    draw_scatter_test_filtervsnofilter_sequences(experiments)

    #draw_map_train_trajectories()








