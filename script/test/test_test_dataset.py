import os
import random
from collections import Counter
from script.utils import load_dataset
from script.utils import Point

def test_info(n_robot):

		exps = [_ for _ in os.listdir(f"data/test/{n_robot}/") if ".csv" in _ and "2734" in _]

		for exp in exps:
			print(exp)
			id = random.choice(range(n_robot))
			test = load_dataset(f"data/test/{n_robot}/{exp}",
								"data/test/test_map_ground_truth.pickle",
								row_range=[id * (10000 - 9), id * (10000 - 9) + (10000 - 9)])

			print("test: ", len(test))

			y_true = [step["true_class"] for step in test]

			print(Counter(y_true))

if __name__ == '__main__':
	test_info(1)