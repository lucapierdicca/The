import os
import pickle
import random
from collections import namedtuple
from pprint import pprint
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB
from bisect import bisect_left

from tqdm import tqdm

from draw_map import draw_map_test_trajectories
from utils import Classifier, loadDataset, GaussianFilter
import numpy as np



Point = namedtuple('Point','x y')

n_robots = [1]
feature_type = "geometricB"  # ["template", "geometricP", "geometricB"]
handle_occlusion = False
load_model = False

train = loadDataset("data/train/unstructured_occluded_3.csv",
                    "data/train/train_map_2_ground_truth.pickle")

print("train: ", len(train))

if feature_type == "template":
    template = loadDataset("data/train/template2.csv",
                           "data/train/train_map_2_ground_truth.pickle")
    #template = [step for step in template if step["clock"] == 10]
    classlbl_to_template = {}
    for step in template:
        if step["clock"] == 10 and step["true_class"] not in classlbl_to_template:
            classlbl_to_template[step["true_class"]] = step
    template = list(classlbl_to_template.values())
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

        X_train = np.array(X_train)
        print(X_train.shape)

        scaler = preprocessing.StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)

        clf = SVC()
        clf.fit(X_train,y_train)

        # clf = LinearDiscriminantAnalysis(store_covariance=True)
        # clf.fit(X_train, y_train)

        with open(f"model_svc_template_center_{feature_type}.pickle",'wb') as f:
            pickle.dump(clf, f)
        with open(f"scaler_svc_template_center_{feature_type}.pickle",'wb') as f:
            pickle.dump(scaler, f)

    else:
        with open(f"model_svc_template_center_{feature_type}.pickle","rb") as f:
            clf = pickle.load(f)
        with open(f"scaler_svc_template_center_{feature_type}.pickle","rb") as f:
            scaler = pickle.load(f)

    print(clf.classes_)
    # print(clf.coef_)
    # print(clf.covariance_)

    #test
    X_test, y_test, x_pred = [],[],[]

    for r in n_robots:
        print(f"Robots: {r}")
        exps = [_ for _ in os.listdir(f"data/test/{r}/") if ".csv" in _]
        for j,exp in enumerate(exps):
            id = random.choice(range(r))

            test = loadDataset(f"data/test/{r}/{exp}",
                               "data/test/test_map_ground_truth.pickle",
                               row_range=[id * (10000 - 9), id * (10000 - 9) + (10000 - 9)])


            for step in tqdm(test):
                if step['clock'] % 10 == 0:
                    x_pred.append([step['x'], step['y']])
                    if handle_occlusion == True:
                        occlusion_data = [data["occluded"] for data in list(step["world_model_long"].values())]
                        if(occlusion_data.count(True)/len(occlusion_data) < 0.5):
                            y_test.append(step['true_class'])
                            z = Classifier().preProcess(step['world_model_long'], 3)
                            X_test.append(filter.extractFeature(z, handle_occlusion=handle_occlusion))
                        else:
                            print("INVALID")
                    else:
                        y_test.append(step['true_class'])
                        z = Classifier().preProcess(step['world_model_long'], 3)
                        X_test.append(filter.extractFeature(z, handle_occlusion=handle_occlusion))

    X_test = np.array(X_test)
    X_test = scaler.transform(X_test)  # standardize
    print(X_test.shape)

    y_pred = clf.predict(X_test)

    print(Classifier().classification_report_(y_test, y_pred))
    print(Classifier().confusion_matrix_(y_test, y_pred))

    draw_map_test_trajectories(x_pred ,y_pred)

train_test()





# <editor-fold desc="MultinomialNB with counts">
def MultinomialNB_traintest():
    edges = range(10,160,20)
    print(list(edges))
    gap = 10
    filter_min, filter_max = 10, 160

    #train
    X_train, y_train = [],[]

    for step in train:
        z = Classifier().preProcess(step["world_model_long"], 3)
        X_train.append(extractFeatureCounts(z, filter_min, filter_max, edges, gap))
        y_train.append(step["true_class"])

    X_train = np.array(X_train)
    print(X_train.shape)

    clf = MultinomialNB()
    clf.fit(X_train,y_train)

    print(clf.classes_)
    pprint(np.exp(clf.feature_log_prob_))

    #test
    X_test, y_test = [],[]

    for r in n_robots:
        exp_metrics = []
        print(f"Robots: {r}")
        exps = [_ for _ in os.listdir(f"data/test/{r}/") if ".csv" in _]
        for j,exp in enumerate(exps):
            id = random.choice(range(r))

            test = loadDataset(f"data/test/{r}/{exp}",
                               "data/test/test_map_ground_truth.pickle",
                               row_range=[id * (10000 - 9), id * (10000 - 9) + (10000 - 9)])

            for step in test:
                if step['clock'] % 10 == 0:
                    y_test.append(step['true_class'])
                    z = Classifier().preProcess(step['world_model_long'], 3)
                    X_test.append(extractFeatureCounts(z, filter_min, filter_max, edges, gap))

    X_test = np.array(X_test)
    print(X_test.shape)

    y_pred = clf.predict(X_test)

    print(Classifier().classification_report_(y_test, y_pred))
    print(Classifier().confusion_matrix_(y_test, y_pred))
# </editor-fold>

# <editor-fold desc="SVM with geometricP">
def SVMGeometricB_traintest():
    #train
    X_train, y_train = [],[]

    for step in train:
        z = Classifier().preProcess(step["world_model_long"], 3)
        X_train.append(extractFeatureGeometricB(z))
        y_train.append(step["true_class"])


    X_train = np.array(X_train)
    print(X_train.shape)

    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    clf = SVC()
    clf.fit(X_train,y_train)

    print(clf.classes_)

    #test
    X_test, y_test = [],[]

    for r in n_robots:
        exp_metrics = []
        print(f"Robots: {r}")
        exps = [_ for _ in os.listdir(f"data/test/{r}/") if ".csv" in _]
        for j,exp in enumerate(exps):
            id = random.choice(range(r))

            test = loadDataset(f"data/test/{r}/{exp}",
                               "data/test/test_map_ground_truth.pickle",
                               row_range=[id * (10000 - 9), id * (10000 - 9) + (10000 - 9)])

            for step in test:
                if step['clock'] % 10 == 0:
                    y_test.append(step['true_class'])
                    z = Classifier().preProcess(step['world_model_long'], 3)
                    X_test.append(extractFeatureGeometricB(z))

    X_test = np.array(X_test)
    print(X_test.shape)

    X_test = scaler.transform(X_test)

    y_pred = clf.predict(X_test)

    print(Classifier().classification_report_(y_test, y_pred))
    print(Classifier().confusion_matrix_(y_test, y_pred))
# </editor-fold>



