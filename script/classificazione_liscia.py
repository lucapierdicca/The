import os
import pickle
import random
from collections import namedtuple
from sklearn import preprocessing
from sklearn.svm import SVC, LinearSVC
from tqdm import tqdm
from utils import Classifier, load_dataset, GaussianFilter
import numpy as np


Point = namedtuple('Point','x y')

n_robots = [1]
feature_type = "geometricB"  # ["template", "geometricP", "geometricB"]
handle_occlusion = False
load_model = False

train = load_dataset("data/train/unstructured_occluded_3.csv",
                    "data/train/train_map_2_ground_truth.pickle")

print("train: ", len(train))

if feature_type == "template":
    template = load_dataset("data/train/template2.csv",
                           "data/train/train_map_2_ground_truth.pickle")
    #template = [step for step in template if step["clock"] == 10]
    classlbl_to_template = {}
    for step in template:
        if step["clock"] == 10:
            if step["true_class"] not in classlbl_to_template:
                classlbl_to_template[step["true_class"]] = [step]
            else:
                if len(classlbl_to_template[step["true_class"]]) < 9:
                    classlbl_to_template[step["true_class"]].append(step)

    template = []
    for v in classlbl_to_template.values():
        template += v
    print("template: ", len(template))
else:
    template = None
    print("template: ", template)


filter = GaussianFilter("linear", feature_type, template_dataset=template)


def train_test():
    #train
    X_train, y_train = [],[]

    if load_model == False:
        for step in tqdm(train):
            z = Classifier().preProcess(step["world_model_long"], 3)
            X_train.append(filter.extractFeature(z, handle_occlusion=False))
            y_train.append(step["true_class"])

        # for i in tqdm(range(0, len(train)-10, 10)):
        #     z1 = Classifier().preProcess(train[i]["world_model_long"], 3)
        #     z2 = Classifier().preProcess(train[i+10]["world_model_long"], 3)
        #     f1 = filter.extractFeature(z1, handle_occlusion=False)
        #     f2 = filter.extractFeature(z2, handle_occlusion=False)
        #     X_train.append([f2[i]-f1[i] for i in range(len(f1))])
        #     X_train[-1] += f2
        #     y_train.append(train[i+10]["true_class"])

        X_train = np.array(X_train)
        print(X_train.shape)

        scaler = preprocessing.StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)

        clf = SVC(probability=True)
        clf.fit(X_train,y_train)

        with open(f"model_svc_{feature_type}.pickle",'wb') as f:
            pickle.dump(clf, f)
        with open(f"scaler_svc_{feature_type}.pickle",'wb') as f:
            pickle.dump(scaler, f)

    else:
        with open(f"model_svc_{feature_type}.pickle","rb") as f:
            clf = pickle.load(f)
        with open(f"scaler_svc_{feature_type}.pickle","rb") as f:
            scaler = pickle.load(f)

    print(clf.classes_)
    # print(clf.coef_)
    # print(clf.covariance_)

    #test
    for r in n_robots:
        exp_metrics = []
        print(f"Robots: {r}")
        exps = [_ for _ in os.listdir(f"data/test/{r}/") if ".csv" in _]
        for j,exp in enumerate(exps):
            id = random.choice(range(r))

            test = load_dataset(f"data/test/{r}/{exp}",
                               "data/test/test_map_ground_truth.pickle",
                               row_range=[id * (10000 - 9), id * (10000 - 9) + (10000 - 9)])

            y_pred, valid_pred, x_pred, ori_pred, y_true = [], [], [], [], []

            for step in tqdm(test):
                if step['clock'] % 10 == 0:
                    x_pred.append([step['x'], step['y']])
                    ori_pred.append(step["theta"])
                    y_true.append(step['true_class'])

                    if handle_occlusion == True:
                        occlusion_data = [data["occluded"] for data in list(step["world_model_long"].values())]
                        if(occlusion_data.count(True)/len(occlusion_data) < 0.7):
                            z = Classifier().preProcess(step['world_model_long'], 3)
                            feature = filter.extractFeature(z, handle_occlusion=handle_occlusion)
                            feature_std = scaler.transform([feature])  # standardize
                            y_pred_proba = clf.predict_proba(feature_std)
                            prediction = clf.classes_[np.argmax(y_pred_proba)]
                        else:
                            print("INVALID")
                    else:
                        z = Classifier().preProcess(step['world_model_long'], 3)
                        feature = filter.extractFeature(z, handle_occlusion=handle_occlusion)
                        feature_std = scaler.transform([feature])  # standardize
                        y_pred_proba = clf.predict_proba(feature_std)
                        prediction = clf.classes_[np.argmax(y_pred_proba)]

                    y_pred.append(prediction)

            # store metrics
            metrics = {"fb_id": test[0]["fb_id"],
                       "exp": exp,
                       "y_true": y_true,
                       "y_pred": y_pred,
                       "x_pred": x_pred,
                       "ori_pred": ori_pred}

            exp_metrics.append(metrics)

        # dump test metrics
        file_name = f"data/test/{r}/exp_metrics_svc.pickle"
        with open(file_name, 'wb') as f:
            pickle.dump(exp_metrics, f)

            # for i in range(0, len(test)-10, 10):
            #     x_pred.append([test[i+10]['x'], test[i+10]['y']])
            #     if handle_occlusion == True:
            #         occlusion_data = [data["occluded"] for data in list(test[i+10]["world_model_long"].values())]
            #         if(occlusion_data.count(True)/len(occlusion_data) < 0.7):
            #             z1 = Classifier().preProcess(test[i]['world_model_long'], 3)
            #             z2 = Classifier().preProcess(test[i+10]['world_model_long'], 3)
            #             f1 = filter.extractFeature(z1, handle_occlusion=handle_occlusion)
            #             f2 = filter.extractFeature(z2, handle_occlusion=handle_occlusion)
            #             X_test.append([f2[i]-f1[i] for i in range(len(f1))])
            #             X_test[-1] += f2
            #             y_test.append(test[i+10]['true_class'])
            #         else:
            #             print("INVALID")
            #     else:
            #         z1 = Classifier().preProcess(test[i]['world_model_long'], 3)
            #         z2 = Classifier().preProcess(test[i+10]['world_model_long'], 3)
            #         f1 = filter.extractFeature(z1, handle_occlusion=handle_occlusion)
            #         f2 = filter.extractFeature(z2, handle_occlusion=handle_occlusion)
            #         X_test.append([f2[i]-f1[i] for i in range(len(f1))])
            #         X_test[-1] += f2
            #         y_test.append(test[i+10]['true_class'])


random.seed(3215)
train_test()




