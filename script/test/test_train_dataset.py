from script.utils import load_dataset
from collections import Counter
from script.utils import Point

def test_info():
    train = load_dataset("data/train/unstructured_occluded_3.csv",
                        "data/train/train_map_2_ground_truth.pickle")
    print("train: ", len(train))

    y_true = [step["true_class"] for step in train]

    print(Counter(y_true))

if __name__ == '__main__':
    test_info()

