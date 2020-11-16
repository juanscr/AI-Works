#!/usr/bin/env python3

# ==== Global behavior ==== #
import graphviz
import matplotlib.pyplot as plt
import numpy as np
import sklearn.tree as tree

from GetData import create_data
from joblib import load
from matplotlib import rc

rc('text', usetex=True)
save = lambda x: plt.savefig("../../article/figs/" + x, bbox_inches='tight')

# ==== Tree ==== #
class Tree:
    def __init__(self, clas, output, tree_joblib, feature_names, clas_names):
        self.clas = clas
        self.output = output
        self.tree = load(tree_joblib)
        self.feature_names = feature_names
        self.clas_names = clas_names

    @staticmethod
    def read_tree(name_file):
        print("holi")

    def plotmemaybe(self):
        prefix = "tree-"

        # Plot information
        self.plot_tree(prefix + "graph.pdf")


    def plot_tree(self, name_file):
        dot_data = tree.export_graphviz(self.tree,
                                        out_file=None,
                                        feature_names=self.feature_names,
                                        class_names=self.clas_names,
                                        filled=True, rounded=True,
                                        special_characters=True)
        graph = graphviz.Source(dot_data)
        graph.render(name_file)
