import numpy as np
import matplotlib.pyplot as plt
from math import pi, cos, sin
from scipy.optimize import dual_annealing



class Solver():
    def __init__(self, path_length=15, step_size=0.03, forest=(0, 1)):
        self.rng = np.random.default_rng()
        self.path = np.zeros(shape=(2, path_length))
        self.path_length = path_length
        self.step_size = step_size
        self.forest = forest

        for point_idx in range(path_length - 1):
            theta = 2 * pi * self.rng.random()
            offset = np.array([cos(theta), sin(theta)])
            self.path[:, point_idx + 1] = offset

    def objective(self, x):
        x = x.reshape(2, self.path_length)
        #random rotation
        t = 2 * pi * self.rng.random()
        R = np.array([[cos(t), -sin(t)], [sin(t), cos(t)]])
        x = R @ x
        #random translation
        translation = np.array([[self.rng.random()*100, self.rng.random() - x[1, 0]]]).T
        x += translation
        point_y = x[1].cumsum()
        upper_bound = self.forest[1]
        lower_bound = self.forest[0]
        if first := (np.nonzero((point_y > upper_bound) | (point_y < lower_bound))):
            if first[0].size == 0:
                return 100
            escape_path = x[:, :first[0][0] + 1]
            return np.linalg.norm(escape_path)

    def optimize(self):
        options = {"method": "Nelder-Mead", "options": {"disp": True, "maxiter":100000}}
        ret = dual_annealing(self.objective, [(-0.1, 0.1) for _ in range(self.path_length * 2)], x0=self.path.reshape(-1), local_search_options=options, initial_temp=1, maxfun=10000)
        self.show_plot(ret.x.reshape(2, self.path_length))
        print(ret.x.reshape(2, self.path_length), ret.success)

    def show_plot(self, x):
        plt.plot(x[0].cumsum(), x[1].cumsum())
        plt.show()


x = Solver()
x.optimize()