"""
Utpal Kumar
Institute of Earth Sciences, Academia Sinica
This is a wrapper for the dtaidistance package for the DTW computation
See https://dtaidistance.readthedocs.io/en/latest/usage/dtw.html for details
"""

from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

# default matplotlib parameters
import matplotlib
fontsize = 22
font = {'family': 'Times',
        'weight': 'bold',
        'size': fontsize}

matplotlib.rc('font', **font)
plt.rcParams["figure.figsize"] = (12, 6)
plt.style.use('ggplot')


class dtw_signal_pairs:
    def __init__(self, s1, s2, labels=['s1', 's2']):
        '''
        Analyze the DTW between a pair of signal
        '''
        self.s1 = np.array(s1, dtype=np.double)
        self.s2 = np.array(s2, dtype=np.double)
        self.labels = labels

    def plot_signals(self, figname=None):
        '''
        Plot the signals
        '''
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(self.s1, color="C0", lw=1)
        ax[0].set_ylabel(self.labels[0], fontsize=fontsize)

        ax[1].plot(self.s2, color="C1", lw=1)
        ax[1].set_ylabel(self.labels[1], fontsize=fontsize)

        if figname:
            plt.savefig(figname, bbox_inches='tight', dpi=300)
            plt.close()
            fig, ax = None, None

        return fig, ax

    def compute_distance(self, pruning=True, best_path=False):
        '''
        Returns the DTW distance
        '''
        if not best_path:
            if pruning:
                # prunes computations by setting max_dist to the Euclidean upper bound
                distance = dtw.distance_fast(
                    self.s1, self.s2, use_pruning=True)
            else:
                distance = dtw.distance(self.s1, self.s2)
        else:
            _, path = dtw.warping_paths(
                self.s1, self.s2, window=None, psi=None)
            best_path = dtw.best_path(path)
            distance = path[best_path[-1][0], best_path[-1][1]]

        return distance

    def compute_warping_path(self, windowfrac=None, psi=None, fullmatrix=False):
        '''
        Returns the DTW path
        :param windowfrac: Fraction of the signal length. Only allow for shifts up to this amount away from the two diagonals.
        :param psi: Up to psi number of start and end points of a sequence can be ignored if this would lead to a lower distance
        :param full_matrix: The full matrix of all warping paths (or accumulated cost matrix) is built 
        '''
        if fullmatrix:
            if windowfrac:
                window = int(windowfrac * np.min([len(self.s1), len(self.s2)]))
            else:
                window = None
            d, path = dtw.warping_paths(
                self.s1, self.s2, window=window, psi=psi)
        else:
            path = dtw.warping_path(self.s1, self.s2)

        return path

    def plot_warping_path(self, figname=None):
        '''
        Plot the signals with the warping paths
        '''

        distance = self.compute_distance()

        fig, ax = dtwvis.plot_warping(
            self.s1, self.s2, self.compute_warping_path())
        ax[0].set_ylabel(self.labels[0], fontsize=fontsize)
        ax[1].set_ylabel(self.labels[1], fontsize=fontsize)

        if figname:
            plt.savefig(figname, bbox_inches='tight')
            plt.close()
            fig, ax = None, None
        return distance, fig, ax

    def plot_matrix(self, windowfrac=0.2, plot_best_path=True, psi=None, figname=None, shownumbers=False, showlegend=True):
        '''
        Plot the signals with the DTW matrix
        '''
        path = self.compute_warping_path(
            windowfrac=windowfrac, psi=psi, fullmatrix=True)

        if plot_best_path:
            best_path = dtw.best_path(path)
        else:
            best_path = None

        fig, ax = dtwvis.plot_warpingpaths(
            self.s1, self.s2, path, best_path, shownumbers=shownumbers, showlegend=showlegend)
        ax[0].set_ylabel(self.labels[0], fontsize=fontsize)
        ax[1].set_ylabel(self.labels[1], fontsize=fontsize)
        if figname:
            plt.savefig(figname, bbox_inches='tight')
            plt.close()
            fig, ax = None, None
        return fig, ax


class dtw_clustering:
    def __init__(self, matrix):
        self.matrix = matrix
