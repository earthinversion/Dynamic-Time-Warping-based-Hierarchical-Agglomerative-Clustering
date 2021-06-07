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
from scipy.cluster import hierarchy

from dtaidistance.clustering.hierarchical import HierarchicalTree

# default matplotlib parameters
import matplotlib
fontsize = 26
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

    def plot_matrix(self, windowfrac=0.2, psi=None, figname=None, shownumbers=False, showlegend=True):
        '''
        Plot the signals with the DTW matrix
        '''
        path = self.compute_warping_path(
            windowfrac=windowfrac, psi=psi, fullmatrix=True)

        fig, ax = dtwvis.plot_warpingpaths(
            self.s1, self.s2, path, shownumbers=shownumbers, showlegend=showlegend)
        ax[0].set_ylabel(self.labels[0], fontsize=fontsize)
        ax[1].set_ylabel(self.labels[1], fontsize=fontsize)
        if figname:
            plt.savefig(figname, bbox_inches='tight')
            plt.close()
            fig, ax = None, None
        return fig, ax


class dtw_clustering:
    def __init__(self, matrix):
        '''
        param matrix: matrix of type numpy array
        '''
        self.matrix = matrix
        self.ind_rand_perm = None
        self.shuffled_matrix = []

    def plot_signals(self, figname=None, figsize=(10, 6)):
        fig, ax = plt.subplots(
            nrows=self.matrix.shape[0], sharex=True, figsize=figsize)
        for i in range(self.matrix.shape[0]):
            ax[i].plot(self.matrix[i, :], color=f"C{i}")
            ax[i].set_ylabel(f"S{i}", fontsize=fontsize)
        if figname:
            plt.savefig(figname, bbox_inches='tight')
            plt.close()
            fig, ax = None, None
        return fig, ax

    def reshuffle_signals(self, plot_signals=False, figsize=(10, 12), figname=None):
        fig, ax = None, None
        self.ind_rand_perm = np.random.permutation(self.matrix.shape[0])
        self.shuffled_matrix = self.matrix[self.ind_rand_perm, :]

        if plot_signals:
            fig, ax = plt.subplots(
                nrows=self.matrix.shape[0], sharex=True, figsize=figsize)
            for i, randidx in enumerate(self.ind_rand_perm):
                ax[i].plot(self.shuffled_matrix[i, :], color=f"C{i}")
                ax[i].set_ylabel(f"S{randidx}", fontsize=fontsize)

            if figname:
                plt.savefig(figname, bbox_inches='tight')
                plt.close()

        return self.shuffled_matrix, fig, ax

    def compute_distance_matrix(self, compact=True, block=None):
        ds = dtw.distance_matrix_fast(
            self.matrix, compact=compact, block=block)
        return ds

    def compute_cluster(self, clusterMatrix=None):
        if clusterMatrix is None:
            clusterMatrix = self.matrix

        model = HierarchicalTree(
            dists_fun=dtw.distance_matrix_fast, dists_options={}, show_progress=True)
        cluster_idx = model.fit(clusterMatrix)
        return model, cluster_idx

    def plot_cluster(self, figname=None, figsize=(20, 8), color_thresh=None):
        if len(self.shuffled_matrix):
            model, cluster_idx = self.compute_cluster(
                clusterMatrix=self.shuffled_matrix)
        else:
            model, cluster_idx = self.compute_cluster()
        linkage_list = model.linkage
        R = hierarchy.dendrogram(linkage_list, no_plot=True)
        if not color_thresh:
            color_thresh = np.amax(R['dcoord'])
        stn_sort = []
        if len(self.shuffled_matrix):
            for stn_idx in R["leaves"]:
                stn_sort.append(self.ind_rand_perm[stn_idx])
        else:
            for stn_idx in R["leaves"]:
                stn_sort.append(stn_idx)

        fig, ax = plt.subplots(1, 1, figsize=figsize, sharey=True)
        dn = hierarchy.dendrogram(
            linkage_list, color_threshold=color_thresh, ax=ax)
        ax.set_xticks(np.arange(5, len(R["ivl"]) * 10 + 5, 10))
        ax.set_xticklabels(stn_sort, fontsize=fontsize)
        ax.set_ylabel("DTW Distance", fontsize=fontsize)
        ax.set_xlabel("3-D Stations", fontsize=fontsize)
        dist_sort, lchild, rchild = [], [], []
        for lnk in linkage_list:
            left_child = lnk[0]
            right_child = lnk[1]
            dist_left_right = lnk[2]
            lchild.append(left_child)
            rchild.append(right_child)
            dist_sort.append(dist_left_right)

        if figname:
            plt.savefig(figname, bbox_inches='tight')
            plt.close()
            fig, ax = None, None
        return fig, ax
