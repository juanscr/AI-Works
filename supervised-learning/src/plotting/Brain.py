#!/usr/bin/env python3

# ==== Global behavior ==== #
from GetData import create_data, create_sens_spec
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rc
from sklearn.metrics import roc_curve, auc

rc('text', usetex=True)
save = lambda x: plt.savefig("../../article/figs/" + x, bbox_inches='tight')

# ==== Brain ==== #
class Brain:
    def __init__(self, brain_data):
        self.eta = 0
        self.ls = []
        self.grads = []
        self.err = []
        self.out = []
        self.get_brain_from_string(brain_data)

    def get_brain_from_string(self, brain_data):
        # Eta
        self.eta = float(brain_data[0][1:])

        # Number of neurons
        self.ls = list(map(int, brain_data[1].split(",")[1:-1]))

        # Gradients
        for i in range(3, 53):
            self.grads.append(brain_data[i].split(",")[:-1])
        self.grads = np.array(self.grads, dtype=float)

        # Error
        for i in range(54, 104):
            self.err.append(brain_data[i][:-1])
        self.err = np.array(self.err, dtype=float)

        # Output
        for i in range(105, len(brain_data) - 1):
            self.out.append(brain_data[i][:-1])
        self.out = np.array(self.out, dtype=float)

    @staticmethod
    def read_brains(name_file):
        brain_csv = open(name_file).read().split("eta")
        brain_csv = list(map(lambda x: x.split("\n"), brain_csv))[1:]

        # Create each brain
        brains = []
        for brain_data in brain_csv:
            brains.append(Brain(brain_data))

        return brains

    def plotmemaybe(self, aux_prefix=""):
        prefix = ""
        for l in self.ls:
            prefix += str(l) + "-"
        prefix += str(self.eta) + "-"
        if len(aux_prefix) != 0:
            prefix += aux_prefix + "-"

        # Plot information
        self.plot_grads(prefix + "gradients.pdf")
        self.plot_err(prefix + "error.pdf")
        self.plot_roc(prefix + "roc.pdf")

    def plot_grads(self, name_file):
        epocs = range(1, 51)
        for i in range(self.grads.shape[1]):
            plt.plot(epocs, self.grads[:, i], label="$l={}$".format(i + 1))
        plt.legend()
        plt.xlabel("Epocs")
        plt.ylabel("$$\sum_i \delta_i^l$$")
        save(name_file)
        plt.clf()

    def plot_err(self, name_file):
        epocs = range(1, 51)
        plt.plot(epocs, self.err)
        plt.xlabel("Epocs")
        plt.ylabel("$\\xi_{av}$")
        save(name_file)
        plt.clf()

    def plot_roc(self, name_file):
        _, datay = create_data("../data/num-data.csv", "../data/indexes.csv",
                                   sep=False)
        datay = np.resize(datay, datay.shape[0])

        # Roc curve
        rocx, rocy, _ = roc_curve(datay, self.out)
        area = auc(rocx, rocy)

        # Plot
        plt.plot(rocx, rocy, label="$A = %0.2f$" % area)
        plt.plot([0, 1], [0, 1], 'k', lw=0.5, linestyle='--')
        plt.legend()
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        save(name_file)
        plt.clf()

# ==== Plotting ==== #
# Data creation
brains = Brain.read_brains("../../results/nn-results.csv")
for brain in brains:
    brain.plotmemaybe()

# Sensitivity and specificity
outs = list(map(lambda x: x.out, brains))
sep_info = list(map(lambda x: {"eta": x.eta, "ls": x.ls}, brains))
create_sens_spec(outs, "../../results/nn-specs.csv", sep_info,
                 "../data/num-data.csv", "../data/indexes.csv")

# Data creation embedded
brains = Brain.read_brains("../../results/nn-results-emb.csv")
for brain in brains:
    brain.plotmemaybe("emb")

# Sensitivity and specificity
outs = list(map(lambda x: x.out, brains))
sep_info = list(map(lambda x: {"eta": x.eta, "ls": x.ls}, brains))
create_sens_spec(outs, "../../results/nn-emb-specs.csv", sep_info,
                 "../data/num-data.csv", "../data/indexes.csv")
