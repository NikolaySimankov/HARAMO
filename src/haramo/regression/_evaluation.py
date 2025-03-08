###########
# Imports #
###########

import pandas as pd
import numpy as np

from scipy.stats import (
    kendalltau,
    spearmanr,
    pointbiserialr,
    pearsonr,
)

from sklearn.metrics import (
    r2_score,
    root_mean_squared_error,
    root_mean_squared_log_error,
    confusion_matrix,
)

from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


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
    scores = np.array([pearsonr(X[:, i], y).statistic for i in range(X.shape[1])])

    # Calculate p-values for the hypothesis test for each feature with respect to y
    pvalue = np.array([pearsonr(X[:, i], y).pvalue for i in range(X.shape[1])])

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
    scores = np.array([kendalltau(X[:, i], y).statistic for i in range(X.shape[1])])

    # Calculate p-values for the hypothesis test for each feature with respect to y
    pvalue = np.array([kendalltau(X[:, i], y).pvalue for i in range(X.shape[1])])

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
    scores = np.array([spearmanr(X[:, i], y).statistic for i in range(X.shape[1])])

    # Calculate p-values for the hypothesis test for each feature with respect to y
    pvalue = np.array([spearmanr(X[:, i], y).pvalue for i in range(X.shape[1])])

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
    # Calculate Point-Biserial correlation coefficients for each feature with respect to y
    scores = np.array([pointbiserialr(X[:, i], y).statistic for i in range(X.shape[1])])

    # Calculate p-values for the hypothesis test for each feature with respect to y
    pvalue = np.array([pointbiserialr(X[:, i], y).pvalue for i in range(X.shape[1])])

    # Return the scores and p-values as a tuple
    return (scores, pvalue)


def approx_accuracy(y_true, y_pred):
    """
    Compute the approximate accuracy (+/- 1 category) of a classification model. (for Ordinal or discrete data)

    Parameters:
    - y_true (array-like): The true labels.
    - y_pred (array-like): The predicted labels.

    Returns:
    - float: The approximate accuracy of the model.
    """

    # Compute the confusion matrix
    cm = confusion_matrix(y_true.round(), y_pred.round())

    # Initialize the count of correct predictions
    correct_predictions = 0

    # Iterate over the confusion matrix
    for i in range(cm.shape[0]):
        # Count the correct predictions (diagonal elements)
        correct_predictions += cm[i, i]

        # Count the correct predictions considering the neighboring class
        if i > 0:
            correct_predictions += cm[i, i - 1]
        if i < cm.shape[0] - 1:
            correct_predictions += cm[i, i + 1]

    # Compute the accuracy
    accuracy = correct_predictions / np.sum(cm)

    return accuracy


def plot_confusion_matrix(true_values, predicted_values, title=None, path=None):
    """
    Plot and save a stacked 5 fold Cross-Validation confusion matrix as a heatmap.

    Args:
        true_values (array-like): Array of true values.
        predicted_values (array-like): Array of predicted values.
        title (str): Title for the plot.

    Returns:
        None
    """

    if title is None:
        title = datetime.now().strftime("%d%m%Y-%H%M")

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
    title = f"{title}_confusion_matrix"
    ax.set_title(title)
    ax.set_xticklabels((2**labels).round(3).astype("str"))
    ax.set_yticklabels((2**labels).round(3).astype("str"))
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save the plot as a PDF file
    # Display the plot when path is not provided
    if path is not None:
        plt.savefig(path / f"{title}.pdf")
    else:
        plt.show()


def regression_report(true_value, predicted_value):
    """
    Generate a regression report based on the true and predicted values.

    Parameters:
    - true_value (array-like): The true values of the regression.
    - predicted_value (array-like): The predicted values of the regression.
    - population_threshold (float): The threshold used to define populations.

    Returns:
    - pd.Series: A pandas Series containing various metrics and statistics of the regression.

    Note:
    - The function calculates the confusion matrix, initializes counters for each diagonal category,
      and aggregates data to generate the regression report.
    - Some additional metrics such as ±1 Dil. Acc., Sens., Spec., R², RMSE, RMSLE, Bal. Acc, Bal. MAE,
      Bal. RMSE, and Bal. RMSLE may also be included in the report if they can be calculated.

    """
    # Drop indices with negative values in y_pred
    indices = true_value.index
    indices = indices[predicted_value[indices] >= 0]

    # Calculate confusion matrix
    cm = confusion_matrix(true_value.round(), predicted_value.round())

    # Initialize counters for each diagonal category
    table = {
        ">= -2": 0,
        "-2": 0,
        "-1": 0,
        "0": 0,
        "+1": 0,
        "+2": 0,
        "> +2": 0,
    }

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            diff = j - i  # Difference between predicted and true value

            if diff == 0:
                table["0"] += cm[i, j]
            elif diff == -1:
                table["-1"] += cm[i, j]
            elif diff == -2:
                table["-2"] += cm[i, j]
            elif diff == 1:
                table["+1"] += cm[i, j]
            elif diff == 2:
                table["+2"] += cm[i, j]
            elif diff < -2:
                table[">= -2"] += cm[i, j]
            elif diff > 2:
                table["> +2"] += cm[i, j]

    table["±1 2-fold dil. Acc."] = approx_accuracy(true_value, predicted_value)

    table["R²"] = r2_score(true_value, predicted_value)

    table["RMSE"] = root_mean_squared_error(true_value, predicted_value)

    table["RMSLE"] = root_mean_squared_log_error(
        true_value[indices], predicted_value[indices]
    )

    return pd.Series(table)


def mean_report(table):
    # get only the follwing columns
    table = table[
        [
            "Antibiotic",
            "Model",
            "±1 2-fold dil. Acc.",
            "R²",
            "RMSE",
            "RMSLE",
        ]
    ]

    # compute mean and std in the following format "mean ± std" based on the columns "antibio" and "Model"
    table = (
        table.groupby(["Antibiotic", "Model"])
        .agg(
            {
                "±1 2-fold dil. Acc.": ["mean", "std"],
                "R²": ["mean", "std"],
                "RMSE": ["mean", "std"],
                "RMSLE": ["mean", "std"],
            }
        )
        .reset_index()
    )

    # create a new dataframe and merge each mean and std into a single column with the format "mean ± std"
    new_table = pd.DataFrame()
    # add columns "antibio" and "Model"
    new_table = table[["Antibiotic", "Model"]]

    metrics = [
        "±1 2-fold dil. Acc.",
        "R²",
        "RMSE",
        "RMSLE",
    ]

    for metric in metrics:
        mean_values = table[(metric, "mean")].round(3).astype(str)
        std_values = table[(metric, "std")].round(3).astype(str)
        new_table[metric] = mean_values + " ± " + std_values

    # sort by antibio ascending then by Bal. MAE ascending
    new_table = new_table.sort_values(
        by=["Antibiotic", "RMSLE"], ascending=[True, True]
    )

    return new_table
