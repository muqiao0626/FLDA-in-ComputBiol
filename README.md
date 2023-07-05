# FLDA-in-ComputBiol
FLDA for "Factorized linear discriminant analysis and its application in computational biology"

## Overview
This repository contains Python code for implementing a two-dimensional factorized linear discriminant analysis (FLDA) algorithm, as detailed in the research paper: https://arxiv.org/pdf/2010.02171.pdf. The code utilizes Numpy and scikit-learn libraries. This repository also includes metrics to assess FLDA and to compare it with alternative methods.

## FLDA_2d_complete Class
The FLDA_2d_complete class includes the following methods:

init(): Initializes the FLDA_2d_complete class with the number of components to extract from each dimension.

fit(): Fits the model with the data and calculates the eigenvectors and eigenvalues of the covariance matrix of each dimension. It returns the eigenvectors and the eigenvalues.

sparse_fit(): Fits the model with the data and calculates the sparse eigenvectors and eigenvalues of the covariance matrix of each dimension. This method can be used for feature selection on high-dimensional data.

## Usage
The FLDA_2d_complete class is particularly useful for datasets where the data is partitioned into two classes. The fit method calculates the eigenvectors and eigenvalues of the covariance matrix of each dimension. The sparse_fit method performs the same operation but with sparse eigenvectors, which can be used to select important features from the dataset.

To use this code, import it into your Python script and create an instance of the FLDA_2d_complete class. Then call either the fit or sparse_fit method with the appropriate input data.

## Metrics
This repository also provides a set of evaluation metrics for analyzing the performance and efficiency of the FLDA implementation. Detailed definitions of these metrics are available in the research paper, and their implementation can be found in the metrics.py file. These metrics include:

Signal-to-Noise Ratio (SNR): Evaluates the efficacy of each discriminant axis in distinguishing distinct cell types. It can be calculated using the SNR_2d() function.

Explained Variance (EV): Assesses how much of a feature's variance is captured by a discriminant axis. Use the EV_2d() function for this calculation.

Mutual Information (MI): Measures the association between each discriminant axis and each feature. It's computed using the MI_2d() function.

Modularity Score: Checks whether an axis is majorly dependent on a single feature, indicating successful disentanglement of features. You can compute it using the Modularity() function.