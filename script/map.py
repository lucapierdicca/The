import pickle


class Map:
	paths = {
		"train_map_ground_truth":"data/train/train_map_2_ground_truth.pickle",
		"train_map_wall_boundary_vertices":"data/train/train_map_2_wall_boundary_vertices.pickle",
		"test_map_ground_truth":"data/train/test_map_ground_truth.pickle",
		"test_map_wall_boundary_vertices":"data/train/test_map_wall_boundary_vertices.pickle"
	}

	def __init__(self, type="train"):
		if type not in ["train", "test"]:
			raise ValueError(f"argument type cannot be \"{type}\"")

		self.type = type
		self.ground_truth = None
		self.wall_boundary_vertices = None

		self._load()

	def _load(self):

		path = None
		if self.type == "train":
			path = Map.paths["train_map_ground_truth"]
		elif self.type == "test":
			path = Map.paths["test_map_ground_truth"]

		with open(path, "rb") as f:
			self.ground_truth = pickle.load(f)

		if self.type == "train":
			path = Map.paths["train_map_wall_boundary_vertices"]
		elif self.type == "test":
			path = Map.paths["test_map_wall_boundary_vertices"]

		with open(path, "rb") as f:
			self.wall_boundary_vertices = pickle.load(f)




