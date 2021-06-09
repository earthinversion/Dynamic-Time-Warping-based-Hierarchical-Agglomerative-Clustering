"""
dtw.dtw_analysis
----------------
This module is built around the dtaidistance package for the DTW computation and scipy.cluster

:author: Utpal Kumar, Institute of Earth Sciences, Academia Sinica
:note: See https://dtaidistance.readthedocs.io/en/latest/usage/dtw.html for details on dtaidistance package
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

# to edit text in Illustrator
matplotlib.rcParams['pdf.fonttype'] = 42


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


def plot_signals(matrix, labels=[], figname=None, figsize=(10, 6), fontsize=fontsize, plotpdf=True):
    fig, ax = plt.subplots(
        nrows=matrix.shape[0], sharex=True, figsize=figsize)
    _labels = []
    for i in range(matrix.shape[0]):
        ax[i].plot(matrix[i, :], color=f"C{i}")
        if len(labels) == 0:
            lab = f"S{i}"
            ax[i].set_ylabel(lab, fontsize=fontsize)
            _labels.append(lab)
    if len(labels):
        try:
            for iaxx, axx in enumerate(ax):
                axx.set_ylabel(f"{labels[iaxx]}", fontsize=fontsize)
        except Exception as e:
            print(e)
            for i in range(matrix.shape[0]):
                if len(labels) == 0:
                    ax[i].set_ylabel(f"S{i}", fontsize=fontsize)
    if figname:
        plt.savefig(figname, bbox_inches='tight')
        if plotpdf:
            figname_pdf = ".".join(figname.split(".")[:-1])+".pdf"
            ax.figure.savefig(figname_pdf,
                              bbox_inches='tight')
        plt.close()
        fig, ax = None, None

    if len(labels) == 0:
        return fig, ax, np.array(_labels)

    return fig, ax


def plot_cluster(lons, lats, figname=None, figsize=(10, 6), fontsize=fontsize, plotpdf=True):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    labels = []
    clusterIdx = 0
    for ilonlat, (loncluster, latcluster) in enumerate(zip(lons, lats)):
        lab = f'Cluster {clusterIdx}'
        if lab in labels:
            ax.plot(loncluster, latcluster, 'o', color=f"C{clusterIdx}", ms=20)
        else:
            ax.plot(loncluster, latcluster, 'o',
                    color=f"C{clusterIdx}", ms=20, label=lab)
            labels.append(lab)
        if (ilonlat+1) % 3 == 0:
            clusterIdx += 1

    plt.legend(fontsize=26, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    if figname:
        plt.savefig(figname, bbox_inches='tight')
        if plotpdf:
            figname_pdf = ".".join(figname.split(".")[:-1])+".pdf"
            ax.figure.savefig(figname_pdf,
                              bbox_inches='tight')
        plt.close()
        fig, ax = None, None
    return fig, ax


def shuffle_signals(matrix, labels=[], plot_signals=False, figsize=(10, 12), figname=None, plotpdf=True):
    ind_rand_perm = np.random.permutation(matrix.shape[0])
    shuffled_matrix = matrix[ind_rand_perm, :]

    if plot_signals:
        fig, ax = None, None
        fig, ax = plt.subplots(
            nrows=matrix.shape[0], sharex=True, figsize=figsize)
        for i, randidx in enumerate(ind_rand_perm):
            ax[i].plot(shuffled_matrix[i, :], color=f"C{i}")
            ax[i].set_ylabel(f"S{randidx}", fontsize=fontsize)

        if figname:
            plt.savefig(figname, bbox_inches='tight')
            if plotpdf:
                figname_pdf = ".".join(figname.split(".")[:-1])+".pdf"
                ax.figure.savefig(figname_pdf,
                                  bbox_inches='tight')
            plt.close()

        return ind_rand_perm, shuffled_matrix, fig, ax
    return ind_rand_perm, shuffled_matrix


class dtw_clustering:
    def __init__(self, matrix, labels=[], longitudes=[], latitudes=[]):
        '''
        :param matrix: matrix of type numpy array, shaped with row as different signals, and column as the values
        :param labels: Labels for each signals in the matrix
        :param longitudes: geographical x location of the signals in the matrix
        :param latitudes: geographical y location of the signals in the matrix
        '''
        self.matrix = matrix
        self.ind_rand_perm = None
        self.shuffled_matrix = []
        self.labels = labels
        if len(longitudes) > 0 and len(latitudes) > 0 and len(longitudes) == len(latitudes) == self.matrix.shape[0]:
            self.longitudes = longitudes
            self.latitudes = latitudes
        else:
            self.longitudes = None
            self.latitudes = None

    def plot_signals(self, figname=None, figsize=(10, 6), fontsize=fontsize):
        labels = self.labels
        fig, ax = plt.subplots(
            nrows=self.matrix.shape[0], sharex=True, figsize=figsize)
        for i in range(self.matrix.shape[0]):
            ax[i].plot(self.matrix[i, :], color=f"C{i}")
            if len(labels) == 0:
                ax[i].set_ylabel(f"S{i}", fontsize=fontsize)
        if len(labels):
            try:
                for iaxx, axx in enumerate(ax):
                    axx.set_ylabel(f"{labels[iaxx]}", fontsize=fontsize)
            except Exception as e:
                print(e)
                for i in range(self.matrix.shape[0]):
                    if len(labels) == 0:
                        ax[i].set_ylabel(f"S{i}", fontsize=fontsize)
        if figname:
            plt.savefig(figname, bbox_inches='tight')
            plt.close()
            fig, ax = None, None
        return fig, ax

    def compute_distance_matrix(self, compact=True, block=None):
        ds = dtw.distance_matrix_fast(
            self.matrix, compact=compact, block=block)
        return ds

    def compute_cluster(self, clusterMatrix=None):
        if clusterMatrix is None:
            clusterMatrix = self.matrix

        model = HierarchicalTree(
            dists_fun=dtw.distance_matrix_fast, dists_options={}, show_progress=False)
        cluster_idx = model.fit(clusterMatrix)
        return model, cluster_idx

    def _put_labels(self, R, addlabels=False):
        stn_sort = []
        if addlabels:
            for stn_idx in R["leaves"]:
                stn_sort.append(self.labels[stn_idx])
        else:
            for stn_idx in R["leaves"]:
                stn_sort.append(f"S{stn_idx}")
        return stn_sort

    def get_linkage(self):
        if len(self.shuffled_matrix):
            model, cluster_idx = self.compute_cluster(
                clusterMatrix=self.shuffled_matrix)
        else:
            model, cluster_idx = self.compute_cluster()
        linkage_matrix = model.linkage
        return linkage_matrix

    def plot_dendrogram(self,
                        figname=None,
                        figsize=(20, 8),
                        xtickfontsize=fontsize,
                        labelfontsize=fontsize,
                        xlabel="3-D Stations",
                        ylabel="DTW Distance",
                        truncate_p=None,
                        max_d=None,
                        annotate_above=10,
                        plotpdf=True):
        '''
        :param truncate_p: show only last truncate_p out of all merged branches
        '''
        if max_d:
            color_thresh = max_d

        else:
            color_thresh = None

        # R = hierarchy.dendrogram(linkage_matrix, no_plot=True)
        linkage_matrix = self.get_linkage()
        fig, ax = plt.subplots(1, 1, figsize=figsize, sharey=True)
        truncate_args = {}
        if truncate_p:
            if truncate_p > len(linkage_matrix)-1:
                truncate_p = len(linkage_matrix)-1
            truncate_args = {"truncate_mode": 'lastp',
                             "p": truncate_p, "show_contracted": True,
                             'show_leaf_counts': False}
        # highest color allowd

        R = hierarchy.dendrogram(
            linkage_matrix, color_threshold=color_thresh, ax=ax, **truncate_args)

        for i, d, c in zip(R['icoord'], R['dcoord'], R['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                ax.plot(x, y, 'o', c=c)
                ax.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                            textcoords='offset points',
                            va='top', ha='center', fontsize=xtickfontsize)

        if not truncate_p:
            if len(self.labels):  # if labels for the nodes are given
                try:
                    stn_sort = self._put_labels(R, addlabels=True)
                except:
                    stn_sort = self._put_labels(R, addlabels=False)
            else:
                stn_sort = self._put_labels(R, addlabels=False)
            ax.set_xticks(np.arange(5, len(R["ivl"]) * 10 + 5, 10))
            ax.set_xticklabels(stn_sort, fontsize=xtickfontsize)
        ax.set_ylabel(ylabel, fontsize=labelfontsize)
        ax.set_xlabel(xlabel, fontsize=labelfontsize)

        if max_d:
            ax.axhline(y=max_d, c='k')

        if figname:
            plt.savefig(figname, bbox_inches='tight')
            if plotpdf:
                figname_pdf = ".".join(figname.split(".")[:-1])+".pdf"
                ax.figure.savefig(figname_pdf,
                                  bbox_inches='tight')
                # plt.savefig(figname_ps, bbox_inches='tight')
            plt.close()
            fig, ax = None, None
        return fig, ax

    def compute_cut_off_inconsistency(self, t=None, depth=2, criterion='inconsistent', return_cluster=False):
        '''
        Calculate inconsistency statistics on a linkage matrix following scipy.cluster.hierarchy.inconsistent
        It compares each cluster merge's height h to the average avg and normalize it by the standard deviation std formed over the depth previous levels
        :param t: threshold to apply when forming flat clusters. See scipy.cluster.hierarchy.fcluster for details
        :type t: scalar
        :param depth: The maximum depth to perform the inconsistency calculation
        :type depth: int
        :return: maximum inconsistency coefficient for each non-singleton cluster and its children; the inconsistency matrix (matrix with rows of avg, std, count, inconsistency); cluster
        '''
        from scipy.cluster.hierarchy import inconsistent, maxinconsts
        from scipy.cluster.hierarchy import fcluster
        linkage_matrix = self.get_linkage()
        incons = inconsistent(linkage_matrix, depth)
        maxincons = maxinconsts(linkage_matrix, incons)
        cluster = None
        if return_cluster:
            if t is None:
                t = np.median(maxincons)

            cluster = fcluster(linkage_matrix, t=t, criterion=criterion)
        return maxincons, incons, cluster

    def plot_hac_iteration(self,
                           figname=None,
                           figsize=(10, 8),
                           xtickfontsize=fontsize,
                           labelfontsize=fontsize,
                           xlabel="Iteration #",
                           ylabel="DTW Distance",
                           plot_color="C0",
                           plotpdf=True):

        linkage_matrix = self.get_linkage()
        xvals = np.arange(0, len(linkage_matrix))
        distance_vals = [row[2] for row in linkage_matrix]
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(xvals, distance_vals, '--.', color=plot_color)
        ax.set_ylabel(ylabel, fontsize=labelfontsize)
        ax.set_xlabel(xlabel, fontsize=labelfontsize)

        if figname:
            plt.savefig(figname, bbox_inches='tight')
            if plotpdf:
                figname_pdf = ".".join(figname.split(".")[:-1])+".pdf"
                ax.figure.savefig(figname_pdf,
                                  bbox_inches='tight')
            plt.close()
            fig, ax = None, None
        return fig, ax
