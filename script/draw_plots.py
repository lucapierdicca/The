from pprint import pprint

import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
from utils import Classifier, Point

def plot_test_metrics():
    fig, ax = plt.subplots()
    # per raccogliere i nomi dei metrics .pickle (non importa da dove, quì ho scelto 1)
    metrics_files = [f_name for f_name in os.listdir("data/test/1/") if "metrics" in f_name]

    # average run accuracy vs n of robots
    for f_name in metrics_files:
        x,y = [],[]
        for r in [1,2,5,10,20,30]:
            avg_run_acc, n_valid_run = 0.0, 0
            with open(f"data/test/{r}/{f_name}","rb") as f:
                exp_run_metrics = pickle.load(f)

            for run_metrics in exp_run_metrics:
                if 'accuracy' in run_metrics['report']:
                    n_valid_run+=1
                    avg_run_acc+=run_metrics['report']['weighted']


            print(r, n_valid_run)

            avg_run_acc /= n_valid_run

            x.append(r)
            y.append(avg_run_acc)

        ax.plot(x, y, 'o--', linewidth=0.5, markersize=3,
                label=exp_run_metrics[0]["classifier_type"] + " " + exp_run_metrics[0]["feature_type"])
        # ax.errorbar(x, y, 0.1, fmt='o', linewidth=2, capsize=6,
        #             label=exp_run_metrics[0]["classifier_type"] + " " + exp_run_metrics[0]["feature_type"])

    ax.set(xlim=(0, 31), ylim=(0.5, 1), xlabel= "number of robots", ylabel="avg accuracy")
    ax.legend()
    ax.grid(linewidth=0.5)
    plt.show()

def plot_test_metrics_compound():

    classifier = Classifier()
    fig, ax = plt.subplots()
    # per raccogliere i nomi dei metrics .pickle (non importa da dove, quì ho scelto 1)
    metrics_files = [f_name for f_name in os.listdir("data/test/1/") if "linear_geometricB" in f_name]

    # average run accuracy vs n of robots
    table = {}
    for f_name in metrics_files:
        x,y,err = [],[],[]
        for r in [1]:
            run_acc = []
            with open(f"data/test/{r}/{f_name}","rb") as f:
                exp_run_metrics = pickle.load(f)

            y_true, y_pred = [],[]
            for run_metrics in exp_run_metrics:
                y_true += run_metrics['y_true']
                y_pred += run_metrics['y_pred']

            report = classifier.classification_report_(y_true, y_pred, output_dict=True)


            x.append(r)
            y.append(report['weighted avg']['f1-score'])
            #y.append(report['accuracy'])

        table[f_name] = list(y)

        print(classifier.classification_report_(y_true, y_pred))
        print(classifier.confusion_matrix_(y_true,y_pred))

        # ax.plot(x, y, 'o--', linewidth=0.5, markersize=3,
        #         label=exp_run_metrics[0]["classifier_type"] + " "
        #               + exp_run_metrics[0]["feature_type"] + " "
        #               + (exp_run_metrics[0]["sequential"] if "sequential" in exp_run_metrics[0] else ""))
        # ax.errorbar(x, y, 0.1, fmt='o', linewidth=2, capsize=6,
        #             label=exp_run_metrics[0]["classifier_type"] + " " + exp_run_metrics[0]["feature_type"])
        ax.plot(x, y, 'o--', linewidth=0.5, markersize=3)
    ax.set(xlim=(0, 31), ylim=(0.2, 1), xlabel= "number of robots", ylabel="accuracy")
    ax.legend()
    ax.grid(linewidth=0.5)
    #plt.savefig("plot.svg")
    #plt.show()

    #pprint(table)

if __name__ == '__main__':
    plot_test_metrics_compound()