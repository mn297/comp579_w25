# Stock Price Prediction with Technical Indicators

This project implements a stock price prediction model using various technical indicators and an ensemble of machine learning algorithms. The model predicts the direction of price movements and provides price predictions with uncertainty bounds for the next 8 hours.

## Table of Contents
- [Overview](#overview)
- [Technical Indicators](#technical-indicators)
- [Lagged Returns](#lagged-returns)
- [Modeling](#modeling)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Evaluation Metrics](#evaluation-metrics)
- [Contributing](#contributing)
- [License](#license)

## Overview
The aim of this project is to predict stock prices based on historical data and technical indicators. It uses an ensemble of Random Forest and Gradient Boosting classifiers to predict the direction of price changes and provides price predictions with associated uncertainty.

## Technical Indicators
The following technical indicators are used as features for the prediction model:

1. **Simple Moving Average (SMA)**: 
   - SMA_10: 10-period moving average of the closing prices.
   - SMA_30: 30-period moving average of the closing prices.

2. **Relative Strength Index (RSI)**: 
   - Measures the speed and change of price movements, typically calculated over 14 periods.

3. **Bollinger Bands**: 
   - Consists of an upper band (BB_upper), a middle band (BB_middle), and a lower band (BB_lower). These bands are calculated based on the standard deviation and a moving average of the closing prices.

4. **Moving Average Convergence Divergence (MACD)**: 
   - MACD: The difference between a short-term and a long-term moving average.
   - MACD_Signal: The signal line which is a moving average of the MACD.

5. **Average True Range (ATR)**: 
   - Measures market volatility by decomposing the entire range of an asset price for a given period.

6. **Volume Moving Averages**: 
   - Volume_SMA_10: 10-period moving average of the volume.
   - Volume_SMA_30: 30-period moving average of the volume.

## Lagged Returns
**Lagged Returns** refer to the historical returns calculated for a stock or any financial asset for previous time periods. These are used as features in predictive models to capture the momentum and mean-reversion effects in financial time series data.

### Calculation
The lagged return for a given lag \( k \) is calculated as follows:

```math
Lagged Return k = (P_t - P_{t-k}) / P_{t-k}
```

Where:
- \( P_t \) is the price at time \( t \).
- \( P_{t-k} \) is the price \( k \) periods before time \( t \).


### Example
Suppose we have the following closing prices for a stock over 5 hours:
- Hour 1: $100
- Hour 2: $102
- Hour 3: $101
- Hour 4: $103
- Hour 5: $105

The lagged returns can be calculated as:
- **1-hour lagged return** at Hour 2: ({102 - 100}/{100} = 0.02) or 2%
- **1-hour lagged return** at Hour 3: ({101 - 102}/{102} = -0.0098) or -0.98%
- **2-hour lagged return** at Hour 3: ({101 - 100}/{100} = 0.01) or 1%
- And so on.

## Modeling
The model uses an ensemble of machine learning algorithms:
- **RandomForestClassifier**: A versatile classifier that uses multiple decision trees, training each tree on a different subset of the data.
- **GradientBoostingClassifier**: An ensemble method that builds trees sequentially, with each tree trying to correct errors made by the previous ones.

The ensemble is implemented using `VotingClassifier` from `scikit-learn`, which combines the predictions of the individual models to improve accuracy.

### Training the Model
1. **Handle Class Imbalance**: The dataset is resampled using SMOTE (Synthetic Minority Over-sampling Technique) to balance the classes.
2. **Hyperparameter Tuning**: GridSearchCV is used to find the best hyperparameters for the RandomForestClassifier.
3. **Model Training**: The RandomForestClassifier and GradientBoostingClassifier are trained with the best parameters and combined into an ensemble model.

## Installation
To run this project, you need Python 3.x installed on your machine. You can install the necessary dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Usage
1. Prepare your data file `nq024.csv` with the appropriate columns (Time, Last, High, Low, Close, Volume).
2. Run the script:

```bash
python stock_prediction.py
```

The script will:

1. Load and preprocess the data.
2. Add technical indicators.
3. Train the ensemble model.
4. Make predictions for the next 8 hours.
5. Output the predictions along with uncertainty bounds.

## Results
The model outputs the following metrics:
- **Precision**
- **Recall**
- **Accuracy**
- **Confusion Matrix**

It also provides predictions for the next 8 hours, including:
- Timestamp
- Predicted direction (Up/Down)
- Predicted price
- Uncertainty bounds for the predicted price

Example output:
``` bash
Best parameters for RandomForest:  {'max_depth': 30, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 300}
Precision: 0.7700381679389313
Recall: 0.7707736389684814
Accuracy: 0.7595
Confusion Matrix:
[[1424  482]
 [ 480 1614]]
Last closing price: 18570.75
Predicted prices for the next 8 hours:
Time: 2024-05-31 16:00:00, Direction: Up, Predicted Price: 18604.93 (18567.72 - 18642.14)
Time: 2024-05-31 17:00:00, Direction: Up, Predicted Price: 18642.64 (18605.35 - 18679.92)
Time: 2024-05-31 18:00:00, Direction: Up, Predicted Price: 18687.25 (18649.88 - 18724.62)
Time: 2024-05-31 19:00:00, Direction: Up, Predicted Price: 18684.96 (18647.59 - 18722.33)
Time: 2024-05-31 20:00:00, Direction: Down, Predicted Price: 18657.30 (18619.99 - 18694.62)
Time: 2024-05-31 21:00:00, Direction: Down, Predicted Price: 18651.11 (18613.81 - 18688.42)
Time: 2024-05-31 22:00:00, Direction: Up, Predicted Price: 18662.40 (18625.08 - 18699.73)
Time: 2024-05-31 23:00:00, Direction: Up, Predicted Price: 18726.61 (18689.15 - 18764.06)
```

## Evaluation Metrics
### Precision
**Precision** is the ratio of correctly predicted positive observations to the total predicted positives. It is a measure of the accuracy of the positive predictions.
```math
{Precision} = \frac{TP}{TP + FP} 
```
Where:
- \( TP \) (True Positives) are the positive instances correctly classified.
- \( FP \) (False Positives) are the negative instances incorrectly classified as positive.

### Recall
**Recall** (Sensitivity or True Positive Rate) is the ratio of correctly predicted positive observations to all observations in the actual class.

```math
{Recall} = \frac{TP}{TP + FN} 
```

Where:

- \( TP \) (True Positives) are the positive instances correctly classified.
- \( FN \) (False Negatives) are the positive instances incorrectly classified as negative.

### Accuracy
**Accuracy** is the ratio of correctly predicted observations to the total observations. It is the most intuitive performance measure and it is simply a ratio of correctly predicted observations to the total observations.

```math
{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} 
```
Where:
- \( TP \) (True Positives) are the positive instances correctly classified.
- \( TN \) (True Negatives) are the negative instances correctly classified.
- \( FP \) (False Positives) are the negative instances incorrectly classified as positive.
- \( FN \) (False Negatives) are the positive instances incorrectly classified as negative.


### Confusion Matrix
A **Confusion Matrix** is a summary of prediction results on a classification problem. The number of correct and incorrect predictions are summarized with count values and broken down by each class. This matrix gives insight not only into the errors being made by a classifier but more importantly the types of errors that are being made.

The confusion matrix shows the counts of:
- **True Positives (TP)**: Correctly predicted positive instances.
- **True Negatives (TN)**: Correctly predicted negative instances.
- **False Positives (FP)**: Incorrectly predicted positive instances.
- **False Negatives (FN)**: Incorrectly predicted negative instances.

### Interpretation of Results
- **Precision** indicates the accuracy of positive predictions; a higher precision means fewer false positives.
- **Recall** indicates how well the model can identify positive instances; a higher recall means fewer false negatives.
- **Accuracy** provides an overall measure of model performance.
- The **Confusion Matrix** helps in understanding the types of errors the model makes, which can be useful for further model tuning and improvement.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any changes or improvements.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
