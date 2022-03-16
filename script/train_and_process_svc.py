import random
import pickle
from tqdm import tqdm
from utils import HybridFilter, load_dataset, Point
import os
from pprint import pprint


n_robots = [20]
load_model = True
handle_occlusion = False

filter = HybridFilter()

train_dataset = None
if load_model == False:
    print("Loading training dataset...")
    train_dataset = load_dataset("data/train/unstructured_occluded_3.csv",
                                 "data/train/train_map_2_ground_truth.pickle")

print("Training models...") if load_model == False \
     else print("Loading models...")
filter.estimateTransitionModel(dataset=train_dataset)
filter.estimateObservationModel(dataset=train_dataset)
print("Done")
print()
pprint(filter.transition_model)
print(filter.clf.classes_)

random.seed(3215)

#testing loop (fake navigation simulation from argos collected data)
for r in n_robots:

    print(f"Robots: {r}")
    experiments_names = [_ for _ in os.listdir(f"data/test/{r}/") if ".csv" in _]
    experiments = []
    for exp in experiments_names:

        id = random.choice(range(r))
        test = load_dataset(f"data/test/{r}/{exp}",
                            "data/test/test_map_ground_truth.pickle",
                            row_range=[id * (10000 - 9), id * (10000 - 9) + (10000 - 9)])

        #y_pred, feature_std_seq, valid_pred, position, orientation, y_true, rab_readings = [], [], [], [], [], [], []
        #belief_pr_seq, likelihood_seq, belief_up_seq, posterior_seq = [], [], [], []

        #belief_up = [0.25]*filter.state_dim # G,C,S,V,I
        #belief_pr = [0.0]*filter.state_dim
        #prediction = "C"
        experiment_data = []
        for step in tqdm(test):
            if step['clock'] % 10 == 0:

                exp_step = {'exp': exp,
                            'fb_id': id,
                            'clock': step['clock'],
                            'x': step['x'], 'y': step['y'],
                            'theta': step['theta'],
                            'true_class':step['true_class'],
                            'feature_std': None,
                            'rab_readings': []}

                occlusion_data = [data["occluded"] for data in list(step["world_model_long"].values())]

                occlusion_ratio = 0.0
                if handle_occlusion == True:
                    valid = False
                    occlusion_ratio = occlusion_data.count(True)/len(occlusion_data)
                    if occlusion_ratio < 0.7:
                        z = filter.classifier.preProcess(step['world_model_long'], 3)
                        feature = filter.extractFeature(z, handle_occlusion=handle_occlusion)
                        feature_std = filter.scaler.transform([feature])  # standardize

                        # belief_pr, likelihood, belief_up, posterior = filter.update(belief_up,
                        #                                                             feature_std,
                        #                                                             occlusion_ratio,
                        #                                                             use_weight=handle_occlusion)
                        #prediction = filter.predict(belief_up, step["rab"], use_neighbors=use_neighbors)
                        valid = True

                else:
                    z = filter.classifier.preProcess(step['world_model_long'], 3)
                    feature = filter.extractFeature(z, handle_occlusion=handle_occlusion)
                    feature_std = filter.scaler.transform([feature])  # standardize
                    # belief_pr, likelihood, belief_up, posterior = filter.update(belief_up,
                    #                                                             feature_std,
                    #                                                             occlusion_ratio,
                    #                                                             use_weight=handle_occlusion)
                    #prediction = filter.predict(belief_up, step["rab"], use_neighbors=use_neighbors)
                    valid = True

                exp_step["feature_std"] = feature_std

                exp_step["rab_readings"] = step["rab_readings"].copy()
                for n in exp_step["rab_readings"]:
                    if "world_model_long" in n:
                        z = filter.classifier.preProcess(n['world_model_long'], 3)
                        feature = filter.extractFeature(z, handle_occlusion=handle_occlusion)
                        feature_std = filter.scaler.transform([feature])

                        n["feature_std"] = feature_std

                experiment_data.append(exp_step)


                # y_pred.append(prediction)
                # valid_pred.append(valid)
                # belief_pr_seq.append(belief_pr)
                # belief_up_seq.append(belief_up)
                # likelihood_seq.append(likelihood)
                # posterior_seq.append(posterior)

        experiments.append(experiment_data)

    # dump experiments
    file_name = f"data/test/{r}/experiments.pickle"
    with open(file_name, 'wb') as f:
        pickle.dump(experiments, f)




