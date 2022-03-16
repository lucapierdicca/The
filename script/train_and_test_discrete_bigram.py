import os
import pickle
import random
from bisect import bisect_left
from tqdm import tqdm
from utils import DiscreteFilter_bigram, load_dataset, Point
from pprint import pprint
import numpy as np
from matplotlib import pyplot as plt
from draw_plots import plot_test_metrics_compound



def produce_test_metrics(filter, n_robots=[1]):

    # testing loop (fake navigation simulation from argos collected data)
    for r in n_robots:
        exp_metrics = []
        print(f"Robots: {r}")
        exps = [_ for _ in os.listdir(f"data/test/{r}/") if ".csv" in _]
        for j,exp in enumerate(exps):
            id = random.choice(range(r))
            # fb_id = f"fb_{id}"
            test = load_dataset(f"data/test/{r}/{exp}",
                               "data/test/test_map_ground_truth.pickle",
                               row_range=[id * (10000 - 9), id * (10000 - 9) + (10000 - 9)])
            belief = [0.25,0.25,0.25,0.25]  # parte in C [I C V G]
            y_pred, x_pred, y_true, belief_evo, obs_evo, outliers_data = [], [], [], [], [], []
            for index_in_test, step in enumerate(tqdm(test)):
                if step['clock'] % 10 == 0:
                    x = [step['x'], step['y']]
                    x_pred.append(x)
                    y_true.append(step['true_class'])
                    z = filter.classifier.preProcess(step['world_model_long'], 3)
                    feature = filter.extractFeature(z)
                    if step["clock"] == 10 and j == 0: print(len(feature))
                    belief = filter.update(belief, feature)
                    prediction = filter.predict(belief)
                    y_pred.append(prediction)

            report = filter.classifier.classification_report_(y_true, y_pred, output_dict=True)
            confusion = filter.classifier.confusion_matrix_(y_true, y_pred)

            # store metrics
            metrics = {"fb_id": test[0]["fb_id"],
                       "exp": exp,
                       "report": report,
                       "confusion": confusion,
                       "y_true": y_true,
                       "y_pred": y_pred,
                       "x_pred": x_pred}

            exp_metrics.append(metrics)

        # dump test metrics
        file_name = f"data/test/{r}/exp_metrics_discrete_bigram.pickle"
        with open(file_name, 'wb') as f:
            pickle.dump(exp_metrics, f)

def compute_transition_model_stationary_distro():
    train = load_dataset("data/train/unstructured.csv",
                        "data/train/train_map_ground_truth.pickle")

    filter = DiscreteFilter_bigram()
    print("Estimate transition model")
    filter.estimateTransitionModel(train)
    print("Done")

    w,v = np.linalg.eig(filter.transition_model)
    print(w)
    print(v)
    stationary_distro = v[:,0]/np.sum(v[:,0]) # the eve with eva=1 normalized -> sum(eve) = 1
    print(stationary_distro)

    belief_old = [0.0,0.0,0.0,1.0]
    belief_old = np.array(belief_old).reshape((4,1))

    i,max_iter = 0,100

    while True:
        belief = filter.transition_model@belief_old
        if np.linalg.norm(belief_old-belief) <= 0.000001 or i == max_iter:
            break

        belief_old = belief
        i+=1

    print(i)
    print(belief)

def inspect_observation_model_bigrams_distro_per_class(filter):
    for k,v in filter.observation_model.items():
        print(k,np.sum(v, axis=0))

    fig, axs = plt.subplots(1, 4)
    for i,(k,v) in enumerate(filter.classid_to_beamunigrams.items()):
        axs[i].hist2d(np.array(v)[:,0],
                      np.array(v)[:,1],
                      bins=edges,
                      density=True)
        axs[i].set(title=filter.classifier.id_to_classlbl[k])

    plt.show()

def inspect_measurements_polygons_per_class(train, filter):
    fig, axs = plt.subplots(1, 4)

    classid_to_polygon_vertices = {"C":[],"I":[],"V":[],"G":[]}
    for step in train:
        class_id = step["true_class"]
        z = filter.classifier.preProcess(step['world_model_long'], 3)
        polygon_vertices = [[d/10*np.cos(angle),d/10*np.sin(angle)] for angle,d in z.items()]
        classid_to_polygon_vertices[class_id].append(polygon_vertices)

    for i,(k,v) in enumerate(classid_to_polygon_vertices.items()):
        print(k, len(v))
        for vertices in random.choices(v, k=1000):
            p = plt.Polygon(vertices, closed=True, fill=None, edgecolor=(0.0,0.0,1.0,0.02))
            axs[i].add_patch(p)
        axs[i].set(title=k, xlim=(-16,16), ylim=(-16,16))

    plt.show()



if __name__ == "__main__":

    # filter = DiscreteFilter()
    # filter.filter_min = 10
    # filter.filter_max = 150
    # filter.gap = 1
    # edges = range(10,160,10)
    # bu = filter.extractFeature({1:11, 2:5, 3:6, 4:3, 8:56.7, 9:144.5, 10:12, 11:11, 12:150, 13:123})
    #
    # print(bu)
    #
    # classid_to_beamunigrams = {1:bu}
    #
    # conditional_dict = {}
    # for classid, beam_unigrams in classid_to_beamunigrams.items():
    #     conditional = np.ones((len(edges), len(edges)))
    #     for bu in beam_unigrams:
    #         prev_beam_bin_index = bisect_left(edges, bu[0]) - 1
    #         next_beam_bin_index = bisect_left(edges, bu[1]) - 1
    #
    #         print(prev_beam_bin_index, next_beam_bin_index)
    #
    #         conditional[next_beam_bin_index, prev_beam_bin_index] += 1.0
    #
    #         #conditional = conditional/ np.sum(conditional, axis=0) # to obtain conditional probabilities
    #
    #     conditional_dict[classid] = conditional
    #
    # pprint(conditional_dict)

    print("Load dataset")
    train = load_dataset("data/train/unstructured.csv",
                        "data/train/train_map_ground_truth.pickle")

    filter = DiscreteFilter_bigram()


    print("Estimate transition model")
    filter.estimateTransitionModel(train)
    print("Done")


    filter_min, filter_max = 10, 160

    for g in [10]:
        for bs in [30]:

            print(f"----->{g}")
            print(f"----->{bs}")

            gap = g
            edges = range(10,160,bs)
            print(list(edges))

            print("Estimate observation model")
            filter.estimateObservationModel(train, edges, gap, filter_min, filter_max)
            print("Done")

            produce_test_metrics(filter)
            #plot_test_metrics_compound()
            #draw_transition_model_distro(filter)



