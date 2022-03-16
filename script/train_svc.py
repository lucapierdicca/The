from utils import HybridFilter, load_dataset, Point
from pprint import pprint

load_model = True
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
filter.estimateSpatialModel(dataset=train_dataset)
print("Done")
print()
#pprint(filter.transition_model)
#print(filter.clf.classes_)