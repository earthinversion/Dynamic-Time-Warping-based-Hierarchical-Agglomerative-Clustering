# Dynamic Time Warping based Hierarchical Agglomerative Clustering for continuous GPS displacements of Taiwan
Under review in [Computers and Geosciences](https://www.journals.elsevier.com/computers-and-geosciences)

Codes to perform Dynamic Time Warping Based Hierarchical Agglomerative Clustering of GPS data

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
