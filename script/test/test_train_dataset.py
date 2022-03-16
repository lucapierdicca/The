from script.draw_map import draw_map_step
from script.map import Map
from script.utils import load_dataset
from collections import Counter
from script.utils import Point
from random import random, choice
from math import pi as PI
import numpy as np

def test_info():
    train = load_dataset("data/train/unstructured_occluded_3.csv",
                        "data/train/train_map_2_ground_truth.pickle")
    print("train: ", len(train))

    y_true = [step["true_class"] for step in train]

    print(Counter(y_true))

def simulate_occlusion(dataset):
    fb_radius = 12.0

    rnd_a = choice(list(dataset[0]["world_model_long"].keys())) # scegli random un angolo (un raggio)
    rnd_bb_real_d = (150  * random()) # scegli random la distanza bordo-bordo reale

    rnd_a = PI/2
    rnd_bb_real_d = 0
    
    rnd_bb_measured_d = max(10.0, rnd_bb_real_d) # distanza bordo-bordo misurata (simula saturazione)
    cc_d = rnd_bb_real_d + fb_radius*2 # distanza centro-centro
    
    print(rnd_a, rnd_bb_real_d, rnd_bb_measured_d, cc_d)

    cc_v = np.array([[cc_d*np.cos(rnd_a)],
                     [cc_d*np.sin(rnd_a)]])
    dir_cc_v = cc_v/cc_d # normalize
    shift_v = fb_radius * np.array([-dir_cc_v[1],
                                    dir_cc_v[0]]) # perpendicularize ccwise
    border_v = cc_v + shift_v
    border_a = np.arctan2(border_v[1], border_v[0])

    delta_a =  np.abs(np.arctan2(np.sin(border_a - rnd_a), np.cos(border_a - rnd_a)))


    for k,v in dataset[0]["world_model_long"].items():
        if k <= rnd_a+delta_a and k >= rnd_a-delta_a:
            dataset[0]["world_model_long"][k]["distance"] = rnd_bb_measured_d
            dataset[0]["world_model_long"][k]["occluded"] = 1
            dataset[0]["world_model_long"][k]["occluding_robot_data"] = [rnd_a, rnd_bb_real_d, border_v, border_a]




def test_draw_map_step():

    toy_dataset = [{'x':0,'y':0,'theta':0,
                    'world_model_long':{-PI+i*(3*2*PI/360):{'distance': 150,
                                                            'occluded':0}
                                        for i in range(1,121)}}]
    simulate_occlusion(toy_dataset)
    draw_map_step(toy_dataset)

    # train_map = Map(type="train")
    # train_dataset = load_dataset("data/train/unstructured_occluded_3.csv",
    #                              "data/train/train_map_2_ground_truth.pickle")
    #
    # draw_map_step(train_dataset, map=train_map, interval=(999,1000,1))

if __name__ == '__main__':

    test_info()

