import random
import pickle
from tqdm import tqdm
from utils import Classifier, GaussianFilter, load_dataset, Point
import os
from pprint import pprint


def produce_test_metrics(classifier_type,
                         feature_type,
                         n_robots=[1],
                         load_model=True,
                         handle_occlusion=False,
                         unique=False):
    # training and/or loading models for the HMM
    train = load_dataset("data/train/unstructured_occluded_3.csv",
                         "data/train/train_map_2_ground_truth.pickle")
    print("train: ", len(train))

    if feature_type == "template":
        template = load_dataset("data/train/template2.csv",
                               "data/train/train_map_ground_truth.pickle")
        template = [step for step in template if step["clock"] == 10]
        print("template: ", len(template))
        filter = GaussianFilter(classifier_type, feature_type, template)
    else:
        filter = GaussianFilter(classifier_type, feature_type)

    print(filter.classifier.classlbl_to_id)

    print("Estimate transition model")
    filter.estimateTransitionModel(train, unique=unique)
    print("Done")

    pprint(filter.transition_model)

    if load_model == True:
        print("Load observation model")
    else:
        print("Estimate observation model")
    filter.estimateObservationModel(dataset=train, load_model=load_model)
    print("Done")

    print(filter.clf.classes_)
    print(filter.clf.coef_)
    print(filter.clf.coef_.shape)
    print(filter.clf.intercept_)
    print(filter.clf.intercept_.shape)


    #testing loop (fake navigation simulation from argos collected data)
    robot_ids = []
    for r in n_robots:
        exp_metrics = []
        print(f"Robots: {r}")
        exps = [_ for _ in os.listdir(f"data/test/{r}/") if ".csv" in _]
        for exp in exps:
            id = random.choice(range(r))
            robot_ids.append(id)
            # fb_id = f"fb_{id}"
            test = load_dataset(f"data/test/{r}/{exp}",
                               "data/test/test_map_ground_truth.pickle",
                               row_range=[id * (10000 - 9), id * (10000 - 9) + (10000 - 9)])

            belief = [0.25]*filter.state_dim # G,C,S,V,I
            y_pred, valid_pred, x_pred, y_true = [], [], [], []
            prediction = "C"
            for step in tqdm(test):
                if step['clock'] % 10 == 0:
                    x_pred.append([step['x'], step['y']])
                    y_true.append(step['true_class'])

                    occlusion_data = [data["occluded"] for data in list(step["world_model_long"].values())]

                    if handle_occlusion == True:
                        valid = False
                        occlusion_ratio = occlusion_data.count(True)/len(occlusion_data)
                        if(occlusion_ratio < 0.7):
                            z = filter.classifier.preProcess(step['world_model_long'], 3)
                            feature = filter.extractFeature(z, handle_occlusion=handle_occlusion)
                            feature_std = filter.scaler.transform([feature])  # standardize
                            belief = filter.update(belief, feature_std[0], weight=None, log=True)
                            prediction = filter.predict(belief)
                            valid = True

                    else:
                        z = filter.classifier.preProcess(step['world_model_long'], 3)
                        feature = filter.extractFeature(z, handle_occlusion=handle_occlusion)
                        feature_std = filter.scaler.transform([feature])  # standardize
                        belief = filter.update(belief, feature_std[0], log=True)
                        prediction = filter.predict(belief)
                        valid = True

                    y_pred.append(prediction)
                    valid_pred.append(valid)


            report = filter.classifier.classification_report_(y_true, y_pred, output_dict=True)
            confusion = filter.classifier.confusion_matrix_(y_true, y_pred)

            # store metrics
            metrics = {"fb_id": test[0]["fb_id"],
                       "exp": exp,
                       "report": report,
                       "confusion": confusion,
                       "y_true": y_true,
                       "y_pred": y_pred,
                       "valid_pred":valid_pred,
                       "x_pred": x_pred,
                       "classifier_type": classifier_type,
                       "feature_type": feature_type}

            exp_metrics.append(metrics)

        # dump test metrics
        file_name = f"data/test/{r}/exp_metrics_{classifier_type}_{feature_type}.pickle"
        with open(file_name, 'wb') as f:
            pickle.dump(exp_metrics, f)

        print(robot_ids)


if __name__ == '__main__':
    classifier_type = ["linear"]  # ["linear", "quadratic"]
    feature_type = ["geometricB"]  # ["template", "geometricP", "geometricB"]
    n_robots = [20]
    load_model = True
    handle_occlusion = True
    unique = True

    random.seed(3215)

    for c in classifier_type:
        for f in feature_type:
            print(c, f)
            produce_test_metrics(c, f,
                                 load_model=load_model,
                                 n_robots=n_robots,
                                 handle_occlusion=handle_occlusion,
                                 unique=unique)
