from matplotlib import gridspec

from utils import BabySitter
from utils import Point

import matplotlib.pyplot as plt
import numpy as np
input_dirs = ["script/video_local_graph",
              "script/video_global_graph"]
BabySitter().stack_map_h_step_graphs_frame(input_dirs)


# fig = plt.figure()
# gs = gridspec.GridSpec(2, 2)
#
# ax_left = fig.add_subplot(gs[:,0])
# ax_ru = fig.add_subplot(gs[0,1])
# ax_rb = fig.add_subplot(gs[1,1])
#
# ys = np.random.random((4,5))
# ax_left.plot(ys.T, "-o")
# ax_ru.plot(ys.T, "-o")
# ax_rb.plot(ys.T, "-o")
# #ax.xticks(np.arange(ys.T.shape[0]), np.arange(1,ys.T.shape[0]+1))
#
#
# plt.show()