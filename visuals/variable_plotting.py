from __future__ import print_function
import math
import matplotlib.pyplot as plt
from collections import defaultdict

class VariablePlotter():
    def __init__(self):
        plt.ion()
        self.variables = defaultdict(list)
        self.constants = defaultdict(lambda: 0.0)
        self.domains = defaultdict(list)
        self.varorders = []
        self.mindomain = float('inf')
        self.maxdomain = float('-inf')
        self.plots = defaultdict(lambda:None)
        self.lastscatters = []
        self.lastvarplots = []
        self.lastconstplots = []

    def clear(self):
        plt.clf()

    def plot_var(self, name, x, y, style, scaletype='symlog', linthreshy=0.000000000001, figure=0):
        plt.figure(figure)
        self.variables[name].append(y)
        self.domains[name].append(x)
        if x < self.mindomain:
            self.mindomain = x
        if x > self.maxdomain:
            self.maxdomain = x
        if not name in self.varorders:
            self.varorders.append(name)
        plt.yscale(scaletype, linthreshy=linthreshy)
        var = plt.plot(self.domains[name], self.variables[name], style, label=name)
        #self.lastvarplots.append(var)


    def plot_constant(self, name, y, style, scaletype='symlog', linthreshy=1.0, figure=0):
        plt.figure(figure)
        self.constants[name] = y
        if not name in self.varorders:
            self.varorders.append(name)
        plt.yscale(scaletype, linthreshy=linthreshy)
        const = plt.plot([self.mindomain, self.maxdomain], [self.constants[name],self.constants[name]], style, label=name)
        #self.lastconstplots.append(const)

    def refresh(self, figure=0):
        plt.figure(figure)
        plt.legend(self.varorders, prop={'size':6})
        plt.pause(0.0001)
        #for plot in self.lastvarplots:
        #    plot.remove()
        #for plot in self.lastconstplots:
        #    plot.remove()

    def save(self, name, figure=0):
        plt.figure(figure)
        plt.savefig(name)

    def plot_points(self, xs, ys, labels, colors, sizes, alpha, figure=1, xlim=[-10,10], ylim=[-10,10]):
        plt.figure(figure)
        uniqlabels = sorted(set(labels))
        legs = []
        for scatter in self.lastscatters:
            scatter.remove()
            #break
        self.lastscatters = []
        for label in uniqlabels:
            x = [xs[i] for i in range(len(xs)) if labels[i] == label]
            for xi in x:
                if xi < xlim[0]:
                    xlim[0] = xi
                if xi > xlim[1]:
                    xlim[1] = xi
            y = [ys[i] for i in range(len(ys)) if labels[i] == label]
            for yi in y:
                if yi < ylim[0]:
                    ylim[0] = yi
                if yi > ylim[1]:
                    ylim[1] = yi
            c = [colors[i] for i in range(len(colors)) if labels[i] == label]
            s = 10 #[sizes[i] for i in range(len(sizes)) if labels[i] == label]
            #a = [alphas[i] for i in range(len(alphas)) if labels[i] == label]
            l = plt.scatter(x, y, s=s, c=c, alpha=1, edgecolors='none')
            self.lastscatters.append(l)
            legs.append(l)

        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        plt.legend(legs, uniqlabels, scatterpoints=1, prop={'size':6})
