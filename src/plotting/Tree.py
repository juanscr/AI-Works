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

    def plotmemaybe(self, data_file, aux_prefix="", n=10):
        prefix = "tree-"
        if len(aux_prefix) != 0:
            prefix += aux_prefix + "-"

        # Plot information
        self.plot_tree(prefix + "graph")
        self.plot_contour(prefix + "contour-", data_file, n)


    def plot_tree(self, name_file):
        dot_data = tree.export_graphviz(self.tree,
                                        out_file=None,
                                        feature_names=self.feature_names,
                                        class_names=self.clas_names,
                                        filled=True, rounded=True,
                                        special_characters=True)
        graph = graphviz.Source(dot_data)
        graph.render(dir_of_file(name_file))

    def plot_contour(self, name_file, data_file, n):
        datax, datay = create_data(data_file, "../data/indexes.csv",
                                   sep=False)
        datay = np.resize(datay, datay.shape[0])

        if datax.shape[1] > 2:
            self.contour_hd(datax, datay, name_file, n)
        else:
            self.contour_2d(datax, datay, name_file, n)

    def contour_2d(self, datax, datay, name_file, n):
        x_range = np.linspace(datax[:, 0].min(), datax[:, 0].max(), n)
        y_range = np.linspace(datax[:, 1].min(), datax[:, 1].max(), n)

        # Mesh
        xx, yy = np.meshgrid(x_range, y_range)

        # Evaluate tree
        x_r = np.reshape(xx.flatten(), (xx.size, 1))
        y_r = np.reshape(yy.flatten(), (yy.size, 1))
        mesh_mat = np.hstack((x_r, y_r))
        output = tree.DecisionTreeClassifier.predict(self.tree, mesh_mat)
        zz = np.reshape(output, xx.shape)

        # Contour
        self.plot_contour_formatted(xx, yy, zz, datax, datay, 0, 1, name_file)

    def contour_hd(self, datax, datay, name_file, n):
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
                self.plot_contour_formatted(xx, yy, zz, datax, datay, i, j,
                                            name_file)

    def plot_contour_formatted(self, xx, yy, zz, datax, datay, i, j, name_file):
        fig = plt.figure()
        ax = plt.axes()
        pcm = ax.contourf(xx, yy, zz, cmap = "Wistia")
        fig.colorbar(pcm, ax=ax)

        # Scatter
        ax.scatter(datax[datay == 0, i], datax[datay == 0, j], color='k',
                    label="Class 0")
        ax.scatter(datax[datay == 1, i], datax[datay == 1, j], color='r',
                    label="Class 1")
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
                    if np.array_equal(mesh[i, [index_i, index_j]],
                                      [xx[j, k], yy[j, k]]):
                        acum += output[i]
                        count += 1
                zz[j, k] = acum / count
        return xx, yy, zz


# ==== Tree plotters ==== #
# High dimensional dataset
dt = Tree("../../results/tree.joblib", ["Level of Attention", "Academic Perfomance",
                                        "Emotional Socialization", "Depression",
                                        "Anxiety", "Hyperactivity"], ["0", "1"])
dt.plotmemaybe("../data/num-data.csv", n=4)

# Embedded dataset
dte = Tree("../../results/tree-emb.joblib", ["X0", "X1"], ["0", "1"])
dte.plotmemaybe("../data/embedded-data.csv", n=4, aux_prefix="emb")
