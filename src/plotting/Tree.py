#!/usr/bin/env python3

# ==== Global behavior ==== #
import graphviz
import matplotlib.pyplot as plt
import numpy as np
import sklearn.tree as tree

from GetData import create_data, create_meshgrid
from joblib import load
from matplotlib import rc

rc('text', usetex=True)
dir_of_file = lambda x: "../../article/figs/" + x
save = lambda x: plt.savefig(dir_of_file(x), bbox_inches='tight')

# ==== Tree ==== #
class Tree:
    def __init__(self, tree_joblib, feature_names, clas_names):
        self.tree = load(tree_joblib)
        self.feature_names = feature_names
        self.clas_names = clas_names

    def plotmemaybe(self, n=10):
        prefix = "tree-"

        # Plot information
        self.plot_tree(prefix + "graph")
        self.plot_contour(prefix + "contour-", n)


    def plot_tree(self, name_file):
        dot_data = tree.export_graphviz(self.tree,
                                        out_file=None,
                                        feature_names=self.feature_names,
                                        class_names=self.clas_names,
                                        filled=True, rounded=True,
                                        special_characters=True)
        graph = graphviz.Source(dot_data)
        graph.render(dir_of_file(name_file))

    def plot_contour(self, name_file, n):
        datax, datay = create_data("../data/num-data.csv", "../data/indexes.csv",
                                   sep=False)
        datay = np.resize(datay, datay.shape[0])

        # Meshgrid
        ranges = []
        for i in range(datax.shape[1]):
            ranges.append(np.linspace(datax[:, i].min(), datax[:, i].max(), n))
        mesh = np.meshgrid(*ranges)

        # Evaluate tree
        mesh_fl = list(map(lambda x : x.flatten(), mesh))
        mesh_col = list(map(lambda x : np.reshape(x, (x.size, 1)), mesh_fl))
        mesh_h = np.hstack(mesh_col)
        output = tree.DecisionTreeClassifier.predict(self.tree, mesh_h)

        # Plot each contour
        for i in range(3):
            for j in range(3, len(self.feature_names)):
                xx, yy, zz = Tree.project_mesh(mesh_h, output, i, j, datax, n)
                fig = plt.figure()
                ax = plt.axes()
                pcm = ax.contourf(xx, yy, zz, cmap = "Wistia")
                fig.colorbar(pcm, ax=ax)


                # Scatter
                ax.scatter(datax[datay == 0, i], datax[datay == 0, j], color='k', label="Class 0")
                ax.scatter(datax[datay == 1, i], datax[datay == 1, j], color='r', label="Class 1")
                ax.legend()
                ax.set_xlabel(self.feature_names[i])
                ax.set_ylabel(self.feature_names[j])
                save(name_file + str(i) + "-" + str(j) + ".pdf")
                plt.clf()

    @staticmethod
    def project_mesh(mesh, output, index_i, index_j, datax, n):
        x_range = np.linspace(datax[:, index_i].min(), datax[:, index_i].max(), n)
        y_range = np.linspace(datax[:, index_j].min(), datax[:, index_j].max(), n)
        xx, yy = np.meshgrid(x_range, y_range)

        # Output
        zz = np.zeros(xx.shape)
        for j in range(xx.shape[0]):
            for k in range(xx.shape[1]):
                acum = 0
                count = 0
                for i in range(mesh.shape[0]):
                    if np.array_equal(mesh[i, [index_i, index_j]], [xx[j, k], yy[j, k]]):
                        acum += output[i]
                        count += 1
                zz[j, k] = acum / count
        return xx, yy, zz


# ==== Tree plotters ==== #
dt = Tree("../../results/tree.joblib", ["Level of Attention", "Academic Perfomance",
                                        "Emotional Socialization", "Depression",
                                        "Anxiety", "Hyperactivity"], ["0", "1"])
dt.plotmemaybe(n = 4)
