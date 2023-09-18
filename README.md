# Dynamic Time Warping based Hierarchical Agglomerative Clustering for continuous GPS displacements of Taiwan
- For details, see the research paper: [On analyzing GNSS displacement field variability of Taiwan: Hierarchical Agglomerative Clustering based on Dynamic Time Warping technique](https://doi.org/10.1016/j.cageo.2022.105243)
- Codes to perform Dynamic Time Warping Based Hierarchical Agglomerative Clustering of GPS data

## You can also open the notebooks in the Binder application online...

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/earthinversion/Dynamic-Time-Warping-based-Hierarchical-Agglomerative-Clustering/master)

## Uses the `dtwhaclustering` package developed for this study
See Documentation: [dtwhaclustering](https://dtwhaclustering.readthedocs.io/en/latest/)

```
pip install dtwhaclustering
```

## Details

This package include codes for processing the GPS displacement data including least-square modelling for trend, co-seismic jumps, 
seasonal and tidal signals. Finally, it can be used to cluster the GPS displacements based on the similarity of the waveforms. The
similarity among the waveforms will be obtained using the DTW distance.

## Notebooks to reproduce the results:
1.	`part1_dtw_hac_east_coast_gps_taiwan`: Read and preprocess the continuous GPS displacements to select best stations with least amount of missing data
2.	`part2_remove_trend_seasonality_least_square_modeling`: Least square model for the seasonal, tidal, trend, and co-seismic jumps, remove the seasonality and tidal parts to focus on the tectonic features
3.	`part3_slope_analysis`: Visualize the variations of linear trend across the selected stations.
4.	`part4_DTW_intro`: Introduction to the concepts of DTW distance
5.	`part5_dtw_clustering_example`: Test the DTW clustering algorithm (and dtwhaclustering package) on the synthetic data. Also compare the results with the Euclidean based HAC method.
6.	`part6_dtw_clustering_gps_displacements`: Apply the DTW distance-based HAC clustering on the modeled continuous GPS displacement residuals of Taiwan
7.	`part7significance_test`: Perform Monte-Carlo based simulations to quantify the significance of the outcome of the clustering using DTW. We randomly shuffle the individual time-series data, and perform the HAC clustering. Randomly shuffled time-series lead to the optimal number of clusters equals to as many clusters as we started with.

## Please cite this work as
1. Kumar, U., Cédric. P. Legendre, Jian-Cheng Lee, Li Zhao, Benjamin Fong Chao (2022) On analyzing GNSS displacement field variability of Taiwan: Hierarchical Agglomerative Clustering based on Dynamic Time Warping technique Computers & Geosciences, 105243. https://doi.org/10.1016/j.cageo.2022.105243
2. Kumar, U., Legendre, C.P. (2022) Crust-mantle decoupling beneath Afar revealed by Rayleigh-wave tomography Sci Rep 12, 17036 https://doi.org/10.1038/s41598-022-20890-5
