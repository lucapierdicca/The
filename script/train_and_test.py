import random
import pickle
from tqdm import tqdm
from utils import Classifier, GaussianFilter, loadDataset, Point
import os
from pprint import pprint


def produce_test_metrics(classifier_type, feature_type, n_robots=[1], load_model=True, sequential=True):
    # training and/or loading models for the HMM
    train = loadDataset("data/train/unstructured.csv",
                        "data/train/train_map_ground_truth.pickle")
    print("train: ", len(train))

    classifier = Classifier()

    if feature_type == "template":
        template = loadDataset("data/train/template.csv",
                               "data/train/train_map_ground_truth.pickle")
        print("template: ", len(template))
        filter = GaussianFilter(classifier_type, feature_type, template)
    else:
        filter = GaussianFilter(classifier_type, feature_type)

    print("Estimate transition model")
    filter.estimateTransitionModel(train)
    print("Done")

    pprint(filter.transition_model)

    if load_model == True:
        print("Load observation model")
    else:
        print("Estimate observation model")
    filter.estimateObservationModel(dataset=train, load_model=load_model)
    print("Done")

    print(filter.clf.coef_)
    print(filter.clf.coef_.shape)
    print(filter.clf.intercept_)
    print(filter.clf.intercept_.shape)

    #testing loop (fake navigation simulation from argos collected data)
    for r in n_robots:
        exp_metrics = []
        print(f"Robots: {r}")
        exps = [_ for _ in os.listdir(f"data/test/{r}/") if ".csv" in _]
        for exp in exps:
            id = random.choice(range(r))
            # fb_id = f"fb_{id}"
            test = loadDataset(f"data/test/{r}/{exp}",
                               "data/test/test_map_ground_truth.pickle",
                               row_range=[id * (10000 - 9), id * (10000 - 9) + (10000 - 9)])
            belief = [0.25, 0.25, 0.25, 0.25]  # [V C I G]
            y_pred, x_pred, y_true, belief_evo, obs_evo, outliers_data = [], [], [], [], [], []
            for index_in_test, step in enumerate(tqdm(test)):
                if step['clock'] % 10 == 0:
                    x = [step['x'], step['y']]
                    x_pred.append(x)
                    y_true.append(step['true_class'])
                    z = classifier.preProcess(step['world_model_long'], 3)
                    feature = filter.extractFeature(z)
                    feature_std = filter.scaler.transform([feature])  # standardize

                    if sequential == True:
                        b = []
                        b.append(belief)
                        pdfs, mahalas, bt_t_1, belief = filter.update(belief, feature_std[0], log=False)
                        b += [bt_t_1, belief]
                        #print(bt_t_1, belief)
                        prediction = filter.predict(belief)

                    else:
                        prediction = filter.predict_non_sequential(feature_std)

                    obs_evo.append([pdfs, mahalas])
                    belief_evo.append(b)
                    y_pred.append(prediction)
                    if (len(set(pdfs)) == 1 and pdfs[0] == 0.01):
                        print("outlier")
                        outliers_data.append({"index_in_test": index_in_test,
                                              "clock": step["clock"],
											  "robot_position":x,
                                              "robot_orientation":step["theta"],
											  "mahalas":mahalas,
                                              "z": z,
                                              "feature": feature,
                                              "feature_std": feature_std})

            report = classifier.classification_report_(y_true, y_pred, output_dict=True)
            confusion = classifier.confusion_matrix_(y_true, y_pred)

            # store metrics
            metrics = {"fb_id": test[0]["fb_id"],
                       "exp": exp,
                       "report": report,
                       "confusion": confusion,
                       "y_true": y_true,
                       "y_pred": y_pred,
                       "x_pred": x_pred,
                       "belief_evo": belief_evo,
                       "obs_evo": obs_evo,
                       "classifier_type": classifier_type,
                       "feature_type": feature_type,
                       "sequential": "s" if sequential else "ns",
                       "outliers_data": outliers_data}

            exp_metrics.append(metrics)

        # dump test metrics
        if sequential == True:
            file_name = f"data/test/{r}/exp_metrics_{classifier_type}_{feature_type}_s.pickle"
        else:
            file_name = f"data/test/{r}/exp_metrics_{classifier_type}_{feature_type}_ns.pickle"
        with open(file_name, 'wb') as f:
            pickle.dump(exp_metrics, f)


if __name__ == '__main__':
    classifier_type = ["linear"]  # ["linear", "quadratic"]
    feature_type = ["geometricB"]  # ["template", "geometricP", "geometricB"]
    load_model = False
    sequential = True
    n_robots = [1]

    for c in classifier_type:
        for f in feature_type:
            print(c, f)
            produce_test_metrics(c, f, load_model=load_model,
                                 n_robots=n_robots, sequential=sequential)
