# haramo

HARAMO: Holistic AutoML-driven Robust pipeline for Applied Multi-Omics (Nikolay Simankov & Helene Soyeurt)

## Overview

HARAMO is a comprehensive AutoML-driven pipeline designed for applied multi-omics data analysis. It leverages the power of automated machine learning (AutoML) to streamline the process of feature selection, scaling, and model training, making it easier to derive meaningful insights from complex multi-omics datasets.

## Purpose

The primary purpose of HARAMO is to provide a robust and flexible framework for multi-omics data analysis. It aims to automate the tedious and time-consuming tasks involved in data preprocessing, feature selection, and model optimization, allowing researchers to focus on interpreting the results and making scientific discoveries.

## How It Works

HARAMO operates through 5 well-defined steps: 

HARAMO leverages Optuna for hyperparameter optimization, allowing users to find the best pipeline configurations for their specific datasets. The Tools is designed to be run twice. The first run is a gridsearch designed to find the best combination of processes for a specific type of algorithm, ensuring that the entire pipeline is optimized for performance. The second run optimizes the hyperparameters of all steps within the defined pipeline to achieve the best results.

Optuna is an open-source hyperparameter optimization framework designed to automate the process of hyperparameter tuning. It uses state-of-the-art algorithms to efficiently search for the best hyperparameter configurations, making it highly relevant for our purpose.

1. **Data Preprocessing**: The input data is preprocessed using various scalers to ensure it is in the right format for machine learning models. The library provides a range of scalers, including standard scaling, min-max scaling, and robust scaling, to ensure that the data is appropriately transformed for machine learning models.

2. **Feature Selection**: The library applies different feature selection methods to identify the most relevant features for the analysis. HARAMO includes various feature selection methods such as variance thresholding, p-value filtering, and Boruta feature selection. It also supports combining multiple feature selection methods for enhanced performance.

3. **Model Training**: HARAMO supports a wide range of classifiers, including SVM classifiers with linear, RBF, or polynomial kernels, SGDClassifier, MLPClassifier, RandomForestClassifier, ExtraTreesClassifier, LGBMClassifier, XGBClassifier, KNeighborsClassifier, ElasticNetClassifier, LogisticRegression, RidgeClassifier, and LinearDiscriminantAnalysis. This diverse set of models ensures that users can find the best algorithm for their specific dataset and analysis needs. Additionally, class weights are balanced to handle class imbalance, ensuring that the models perform well even with imbalanced datasets.

4. **Nested Cross-Validation**: The trained models are evaluated using a nested 4-fold (75% training, 25% testing) cross-validation process. The inner cross-validation is used to find the best pipline (hyper)parameters, while the outer cross-validation is used to retrain and validate the defined pipeline, ensuring accurate evaluation.

5. **Model Selection**: The best-performing models are selected based on their outer cross-validation MCC(Matthews Correlation Coefficient) scores because it provides a balanced measure that takes into account true and false positives and negatives, making it particularly useful for imbalanced datasets and saved for further analysis.

## Installation

To install HARAMO, you can use the following command:

```bash
pip install haramo
```

## Usage

Here is a basic example of how to use HARAMO for multi-omics data analysis:

```python
import pandas as pd
from haramo.classification import magic_now

# Load your multi-omics dataset
X = pd.read_csv('path_to_features.csv')
y = pd.read_csv('path_to_labels.csv')

# Define the output directory
output_dir = 'path_to_output_directory'

# Run the HARAMO pipeline
validation, pipeline, studies = magic_now(
    X=X,
    y=y,
    scoring='balanced_accuracy',
    task='classification',
    feature_selector='combine',
    scaler='optimize',
    algorithm='optimize',
    hyperparameters='optimize',
    random_state=42,
    n_trials=100,
    output_dir=output_dir,
    tag='example_run'
)

# Print the validation results
print(validation)
```

## Contributing

Contributions to HARAMO are welcome! If you have any ideas for new features or improvements, please feel free to open an issue or submit a pull request.

## License

HARAMO is licensed under the MIT License. See the LICENSE file for more details.
