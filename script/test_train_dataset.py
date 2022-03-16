from utils import loadDataset
from collections import Counter

def test_train_info():
    train = loadDataset("data/train/unstructured_occluded_2.csv",
                        "data/train/train_map_ground_truth.pickle")
    print("train: ", len(train))

    y_true = [step["true_class"] for step in train]

    print(Counter(y_true))

if __name__ == '__main__':
    test_train_info()

