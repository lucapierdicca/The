import os
import pickle
import random
from bisect import bisect_left
from tqdm import tqdm
from utils import DiscreteFilter_bigram, DiscreteFilter_count, load_dataset, Point
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
            y_pred, x_pred, y_true, belief_evo = [], [], [], [belief]
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

                    belief_evo.append(belief)
                    y_pred.append(prediction)

            report = filter.classifier.classification_report_(y_true, y_pred, output_dict=True)
            confusion = filter.classifier.confusion_matrix_(y_true, y_pred)

            # store metrics
            metrics = {"fb_id": test[0]["fb_id"],
                       "exp": exp,
                       "report": report,
                       "confusion": confusion,
                       "x_pred": x_pred,
                       "y_true": y_true,
                       "belief_evo":belief_evo,
                       "y_pred": y_pred}

            exp_metrics.append(metrics)

        # dump test metrics
        file_name = f"data/test/{r}/exp_metrics_discrete_count.pickle"
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







if __name__ == "__main__":

    print("Load dataset")
    train = load_dataset("data/train/unstructured.csv",
                        "data/train/train_map_ground_truth.pickle")

    filter = DiscreteFilter_count()


    print("Estimate transition model")
    filter.estimateTransitionModel(train)
    print("Done")

    pprint(filter.transition_model)


    filter_min, filter_max = 10, 160

    for g in [10]:
        for bs in [20]:

            edges = range(10,160,bs)

            print(f"----->{g}")
            print(f"----->{bs}")

            filter.set_params(edges, g, filter_min, filter_max)
            print(list(edges))

            print("Estimate observation model")
            filter.estimateObservationModel(train)
            print("Done")

            produce_test_metrics(filter)
            plot_test_metrics_compound()
            #draw_transition_model_distro(filter)



