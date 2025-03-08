###########
# Imports #
###########

import numpy as np
import pandas as pd

from joblib import (
    Parallel,
    delayed,
)

from scipy.cluster.hierarchy import (
    ward,
    fcluster,
)
from scipy.spatial.distance import squareform

from ..utils import (
    spearman_scorer,
    kendall_scorer,
    union_lists,
)

from sklearn.feature_selection import SelectKBest
#############
# Functions #
#############


class FeatureClustering:
    def __init__(self, X, colinearity_threshold=0.95):
        self.clusters_of_features = {}
        self.X = X
        self.colinearity_threshold = colinearity_threshold

    def dictionnarise(self, X):
        """
        Convert the features into a dictionary of clusters.

        Returns:
            dict: A dictionary where the keys are cluster IDs and the values are lists of feature IDs belonging to each cluster.
        """
        # Initialize an empty dictionary to hold the clusters
        clusters_tmp = {}

        # Iterate over the columns of the DataFrame (i.e., the feature IDs)
        for cluster_id in X.columns:
            # If the current cluster ID is not already a key in the dictionary, add it with an empty list as its value
            if f"{cluster_id}" not in clusters_tmp:
                clusters_tmp[f"{cluster_id}"] = []
            # Append the current feature ID to the list of feature IDs for the current cluster ID
            clusters_tmp[f"{cluster_id}"].append(cluster_id)

        # Update the clusters_of_features attribute with the newly created dictionary
        self.clusters_of_features.update(clusters_tmp)

        # Return the updated clusters_of_features attribute
        return self.clusters_of_features

    def update_clusters_of_features(self, outputs):
        """
        Update the clusters of features based on the given outputs.

        Args:
            outputs (list): A list of dictionaries representing the outputs.

        Returns:
            dict: The updated clusters of features.

        """
        # Initialize an empty dictionary to hold the results
        results = {}

        # Update the results dictionary with each output dictionary
        for result in outputs:
            results.update(result)

        # Iterate over each cluster in the results
        for cluster in results.copy():
            # Copy the list of features for the current cluster
            list_tmp = results[cluster].copy()

            # Iterate over each feature in the copied list
            for key in list_tmp.copy():
                # If the feature is already a key in the clusters_of_features dictionary
                if key in self.clusters_of_features:
                    # Union the current list with the list from the clusters_of_features dictionary
                    list_tmp = union_lists(
                        list_tmp,
                        self.clusters_of_features[key],
                    )
                # Remove the feature from the clusters_of_features dictionary
                self.clusters_of_features.pop(key)

            # Get the representative feature for the current cluster
            representative_feature = list_tmp[0]

            # Update the clusters_of_features dictionary with the representative feature and the updated list
            self.clusters_of_features[representative_feature] = list_tmp

        # Return the updated clusters_of_features dictionary
        return self.clusters_of_features

    def compute_distance_linkage(self, X):
        """
        Compute the distance linkage and correlation matrix.

        Parameters:
        X (DataFrame): The input data.

        Returns:
        distance_linkage (ndarray): The distance linkage matrix.
        correlation_matrix (DataFrame): The correlation matrix.
        """

        # Compute the correlation matrix using Spearman method
        correlation_matrix = X.corr(method="spearman")

        # Ensure the correlation matrix is symmetric by adding it to its transpose and dividing by 2
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2

        # Compute the distance matrix by subtracting the absolute value of the correlation matrix from 1 and taking the absolute value of the result
        distance_matrix = abs(1 - abs(correlation_matrix))

        # Compute the distance linkage matrix using the Ward method on the distance matrix
        distance_linkage = ward(squareform(distance_matrix))

        # Return the distance linkage matrix and the correlation matrix
        return distance_linkage, correlation_matrix

    def correlation_based_clustering(self, X):
        """
        Perform correlation-based clustering on the input data.

        Args:
            X (pandas.DataFrame): The input data.

        Returns:
            dict: A dictionary containing the clusters of features. The keys are the representative features
                and the values are lists of features belonging to each cluster.
        """

        # Initialize an empty dictionary to hold the temporary clusters
        clusters_tmp = {}

        # Compute the distance linkage matrix and ignore the correlation matrix
        distance_linkage, _ = self.compute_distance_linkage(X)

        # Perform hierarchical clustering on the distance linkage matrix
        cluster_ids = fcluster(
            Z=distance_linkage,
            criterion="distance",
            t=1 - self.colinearity_threshold,
        )

        # Iterate over the cluster IDs
        for idx, cluster_id in enumerate(cluster_ids):
            # If the cluster ID is not already a key in the temporary clusters dictionary, add it
            if f"{cluster_id}" not in clusters_tmp:
                clusters_tmp[f"{cluster_id}"] = []
            # Append the current feature to the list of features for the current cluster
            clusters_tmp[f"{cluster_id}"].append(X.columns[idx])

        # Get the list of clusters
        clusters = list(clusters_tmp.keys())
        # Iterate over the clusters
        for cluster in clusters:
            # Copy the list of features for the current cluster
            list_tmp = clusters_tmp[cluster].copy()

            # Iterate over the features in the copied list
            for key in list_tmp.copy():
                # If the feature is already a key in the clusters_of_features dictionary, union the lists
                if key in self.clusters_of_features:
                    list_tmp = union_lists(
                        list_tmp,
                        self.clusters_of_features[key],
                    )

            # Get the representative feature for the current cluster
            representative_feature = list_tmp[0]

            # Update the key for the current cluster in the temporary clusters dictionary to be the representative feature
            clusters_tmp[representative_feature] = clusters_tmp.pop(cluster)

        # Return the temporary clusters dictionary
        return clusters_tmp

    def process_gene(self, gene, X):
        """
        Process a gene and perform correlation-based clustering on the features.

        Parameters:
            gene (str): The name of the gene to process.
            X (pandas.DataFrame): The input data containing the features.

        Returns:
            dict: A dictionary containing the clusters of features.

        """
        # Initialize an empty dictionary to hold the clusters
        clusters = {}

        # Create a copy of the input data, keeping only the columns that are keys in clusters_of_features and contain the gene name
        X_tmp = X[self.clusters_of_features.keys()].filter(like=gene).copy()

        # If there is more than one column in the subsetted data, perform correlation-based clustering
        if len(X_tmp.columns) > 1:
            clusters = self.correlation_based_clustering(X_tmp)

        # Return the clusters
        return clusters

    def clustering_by_genes(self, X, gene_list):
        """
        Perform clustering based on genes.

        This method extracts gene names from the clusters of features and performs clustering
        on the corresponding gene data. It filters the input data based on each gene and
        applies a correlation-based clustering algorithm if there are more than one columns
        of gene data available.

        Returns:
            None
        """
        # If clusters_of_features is empty, initialize it using the dictionnarise method
        if not self.clusters_of_features:
            self.dictionnarise(X)

        # Execute the process_gene function in parallel for each gene in gene_list
        outputs = Parallel(n_jobs=-1, verbose=10)(
            delayed(self.process_gene)(gene, X) for gene in gene_list
        )

        # Update clusters_of_features with the results of the processing and return it
        return self.update_clusters_of_features(outputs)

    def process_group(self, group, X):
        """
        Process a group of features.

        Args:
            group (pandas.Index): The index of the features in the group.
            X (pandas.DataFrame): The input data.

        Returns:
            dict: A dictionary containing the clusters of features.

        """
        # Subset the input data to include only the features in the group
        X_tmp = X[group.index]

        # Perform correlation-based clustering on the subset of data
        clusters = self.correlation_based_clustering(X_tmp)

        # Return the clusters
        return clusters

    def redondant_features_clustering(self, X, digits=3):
        """
        Clusters redundant features based on correlation analysis.

        Parameters:
            X (pandas.DataFrame): The input dataset.
            digits (int, optional): Number of decimal places to round the correlation values to. Defaults to 3.

        Returns:
            dict: A dictionary containing the updated clusters of features.

        """
        # If clusters_of_features is empty, initialize it using the dictionnarise method
        if not self.clusters_of_features:
            self.dictionnarise(X)

        # Create a copy of the input data, keeping only the columns that are keys in clusters_of_features
        X_tmp = X[self.clusters_of_features.keys()].copy()

        # Compute the absolute value of the Spearman correlation of the data, round it to the specified number of decimal places, and convert it to a pandas Series
        correlation = abs(
            pd.Series(
                spearman_scorer(np.array(X_tmp), X_tmp.sample(axis="columns", random_state=42))[0],
                index=X_tmp.columns,
            )
        ).round(digits)

        # Count the number of occurrences of each correlation value
        counts = correlation.value_counts()

        # Process each group of features that have the same correlation value in parallel
        outputs = Parallel(n_jobs=-1, verbose=10)(
            delayed(self.process_group)(correlation[correlation == count], X_tmp)
            for count in counts[counts > 1].index
        )

        # Update clusters_of_features with the results of the processing and return it
        return self.update_clusters_of_features(outputs)

    def order_clusters(self, X, y):
        """
        Orders the clusters of features based on the best representative feature in each cluster.

        Args:
            X (array-like): The input feature matrix.
            y (array-like): The target variable.

        Returns:
            dict: A dictionary containing the ordered clusters of features, where the key is the representative feature
                and the value is the list of features in the cluster.
        """

        # Get the list of clusters
        clusters = list(self.clusters_of_features.keys())

        # Iterate over the clusters
        for cluster in clusters:
            # Copy the list of features for the current cluster
            list_tmp = self.clusters_of_features[cluster].copy()

            # Select the best feature using kendall_scorer
            kendall_selector = SelectKBest(
                score_func=kendall_scorer,
                k=1,
            ).fit(
                X[list_tmp],
                y,
            )
            # Get the name of the representative feature
            representative_feature = kendall_selector.get_feature_names_out()[0]

            # Update the key for the current cluster in clusters_of_features to be the representative feature
            self.clusters_of_features[representative_feature] = self.clusters_of_features.pop(
                cluster
            )

        # Return clusters_of_features
        return self.clusters_of_features

    def get_keys_for_value(self, value):
        """
        Return a list of keys for which the corresponding value contains `value`.

        Parameters:
        - value: The value to search for in the dictionary values.

        Returns:
        - list: A list of keys whose corresponding value contains `value`.
        """
        return [k for k, v in self.clusters_of_features.items() if value in v]
