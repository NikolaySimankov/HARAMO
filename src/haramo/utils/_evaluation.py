###########
# Imports #
###########

import pandas as pd
import numpy as np

from scipy.stats import (
    pointbiserialr,
    pearsonr,
    kendalltau,
    spearmanr,
)

from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    cohen_kappa_score,
    matthews_corrcoef,
)

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

#############
# Functions #
#############


def pearson_scorer(X, y):
    """
    Compute the Pearson correlation coefficient and p-value for each feature in X with respect to y.

    Args:
        X (numpy.ndarray): A 2D array where each column represents a feature.
        y (array-like): An array of target values.

    Returns:
        tuple: A tuple containing two numpy arrays:
            - scores: The Pearson correlation coefficient for each feature.
            - pvalue: The p-value for the hypothesis test for each feature.
    """

    # Calculate Pearson correlation coefficients for each feature with respect to y
    scores = X.apply(lambda x: pearsonr(x, y).statistic)

    # Calculate p-values for the hypothesis test for each feature with respect to y
    pvalue = X.apply(lambda x: pearsonr(x, y).pvalue)

    # Return the scores and p-values as a tuple
    return (scores, pvalue)


def kendall_scorer(X, y):
    """
    Compute the Kendall Tau correlation coefficient and p-value for each feature in X with respect to y.

    Args:
        X (numpy.ndarray): A 2D array where each column represents a feature.
        y (array-like): An array of target values.

    Returns:
        tuple: A tuple containing two numpy arrays:
            - scores: The Kendall Tau correlation coefficient for each feature.
            - pvalue: The p-value for the hypothesis test for each feature.
    """

    # Calculate Kendall Tau correlation coefficients for each feature with respect to y
    scores = X.apply(lambda x: kendalltau(x, y).statistic)

    # Calculate p-values for the hypothesis test for each feature with respect to y
    pvalue = X.apply(lambda x: kendalltau(x, y).pvalue)

    # Return the scores and p-values as a tuple
    return (scores, pvalue)


def spearman_scorer(X, y):
    """
    Compute the Spearman rank-order correlation coefficient and p-value for each feature in X with respect to y.

    Args:
        X (numpy.ndarray): A 2D array where each column represents a feature.
        y (array-like): An array of target values.

    Returns:
        tuple: A tuple containing two numpy arrays:
            - scores: The Spearman rank-order correlation coefficient for each feature.
            - pvalue: The p-value for the hypothesis test for each feature.
    """

    # Calculate Spearman rank-order correlation coefficients for each feature with respect to y
    scores = X.apply(lambda x: spearmanr(x, y).statistic)

    # Calculate p-values for the hypothesis test for each feature with respect to y
    pvalue = X.apply(lambda x: spearmanr(x, y).pvalue)

    # Return the scores and p-values as a tuple
    return (scores, pvalue)


def biserial_scorer(X, y):
    """
    Compute the Point-Biserial correlation coefficient and p-value for each feature in X with respect to y.

    Args:
        X (numpy.ndarray): A 2D array where each column represents a feature.
        y (array-like): An array of binary target values.

    Returns:
        tuple: A tuple containing two numpy arrays:
            - scores: The Point-Biserial correlation coefficient for each feature.
            - pvalue: The p-value for the hypothesis test for each feature.
    """

    # Calculate Point-Biserial rank-order correlation coefficients for each feature with respect to y
    scores = X.apply(lambda x: pointbiserialr(x, y).statistic)

    # Calculate p-values for the hypothesis test for each feature with respect to y
    pvalue = X.apply(lambda x: pointbiserialr(x, y).pvalue)

    # Return the scores and p-values as a tuple
    return (scores, pvalue)


def classification_report(true_value, predicted_value):
    """

    Parameters
    ----------
    true_value : TYPE
        DESCRIPTION.
    predicted_value : TYPE
        DESCRIPTION.
    Returns
    -------
    table : TYPE
        DESCRIPTION.
    """
    table = {}

    table["MCC"] = matthews_corrcoef(
        true_value,
        predicted_value,
    )

    table["F1-score"] = f1_score(
        true_value,
        predicted_value,
    )

    table["Kappa"] = cohen_kappa_score(
        true_value,
        predicted_value,
    )

    table["Bal. Acc."] = balanced_accuracy_score(
        true_value,
        predicted_value,
    )

    table["Precision"] = precision_score(
        true_value,
        predicted_value,
        average="binary",
    )

    table["Sensitivity"] = recall_score(
        true_value,
        predicted_value,
        average="binary",
    )

    table["Selectivity"] = recall_score(
        true_value,
        predicted_value,
        pos_label=0,
        average="binary",
    )

    return pd.Series(table)


def plot_confusion_matrix(true_values, predicted_values, title):
    """
    Plot and save a stacked 5 fold Cross-Validation confusion matrix as a heatmap.

    Args:
        true_values (array-like): Array of true values.
        predicted_values (array-like): Array of predicted values.
        title (str): Title for the plot.

    Returns:
        None
    """

    # Define the path for saving the plots
    path = Path(".")
    plots = path / "plots"
    plots.mkdir(exist_ok=True)  # Create the directory if it doesn't exist

    # Get unique labels from true values and convert to float type
    labels = pd.Series(np.unique(true_values)).astype("float")

    # Calculate confusion matrix without normalization
    cm = confusion_matrix(true_values, predicted_values, labels=labels, normalize=None)
    cm = cm[~np.all(cm == 0, axis=1)]  # Remove rows that are all zeros
    cm_sum = np.sum(
        cm, axis=1, keepdims=True
    )  # Sum of each row for percentage calculation
    cm_perc = cm / cm_sum.astype(float) * 100  # Convert to percentages

    # Initialize an empty array for annotations
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]  # Count value
            p = cm_perc[i, j]  # Percentage value
            if i == j:
                s = cm_sum[i][0]  # Sum of the row
                annot[i, j] = (
                    f"{p:.1f}%\n{c:d}/{s:d}"  # Annotate with percentage and count/sum for diagonal
                )
            elif c == 0:
                annot[i, j] = ""  # Leave empty if count is zero
            else:
                annot[i, j] = (
                    f"{p:.1f}%\n{c:d}"  # Annotate with percentage and count for off-diagonal
                )

    # Plotting the heatmap
    fig, ax = plt.subplots(figsize=(12.8, 9.6))
    sns.heatmap(cm_perc, annot=annot, fmt="", ax=ax, vmin=0, vmax=100, cmap="Blues")
    ax.set_xlabel("Predicted value")
    ax.set_ylabel("True value")

    # Set title and labels
    title = f"{title}_stackedCV_confusion_matrix"
    ax.set_title(title)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save the plot as a PDF file
    plt.savefig(plots / f"{title}.pdf")
    # plt.show()  # Uncomment to show the plot
    plt.close()  # Close the plot to free memory
