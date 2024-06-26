# Flood Predictor

The Flood Predictor is a machine learning application developed in Python, which aims to predict the occurrence of
floods based on historical weather data. This predictor utilizes several machine learning algorithms to evaluate and
compare their performance in predicting floods.

## Features

- **Multiple Machine Learning Models**: Logistic Regression, K-Nearest Neighbors, Support Vector Machines, Decision
  Trees, and Random Forest Classifier.
- **Feature Selection**: Selects the top predictive features for modeling using chi-squared tests.
- **Data Normalization**: Option to use MinMaxScaler for normalizing data to improve model performance.
- **Performance Evaluation**: Evaluates models based on accuracy, recall, and precision, and includes ROC curve
  plotting.

## Prerequisites

To run this project, you will need Python 3.x and the following libraries:

- pandas
- seaborn
- matplotlib
- scikit-learn

You can install these with pip using the following command:

```bash
pip install pandas seaborn matplotlib scikit-learn
```

## Usage

To run the predictor, execute the main script:

```bash
python main.py
```

## Outputs

The outputs include:

- Printed classification reports for each model.
- ROC curves for each model.
- Comparison plots for accuracy, recall, and precision of the models:
-

![Comparison for the ML models](floods/results.png)

