# Industrial Time Series Forecasting: Boiler Drum Water TDS Prediction using LSTM

## Project Overview
This project focuses on developing a robust time series forecasting model to predict `Drum Water TDS` (Total Dissolved Solids) in an industrial boiler system. 
Developed for Forbes Marshall Pvt Ltd Company during my tenure as a Data Science Intern, this solution addresses a critical need for maintaining boiler efficiency, preventing scale formation, and ensuring operational reliability, making this a real-world application of predictive analytics in an industrial setting. The solution leverages advanced feature engineering, data transformation techniques, and Long Short-Term Memory (LSTM) neural networks, coupled with a crucial bias correction step, to achieve high-accuracy predictions.

## Problem Statement
Industrial boilers are complex systems where various parameters, including Drum Water TDS, need continuous monitoring. Fluctuations in TDS can lead to operational inefficiencies, increased maintenance costs, and potential equipment damage. This project aims to predict future Drum Water TDS values based on historical data of related parameters, providing an early warning system for operators.

## Data
The dataset comprises historical time series data from an industrial boiler system, including `Timestamp` and key operational parameters such as `Boiler Pressure P1`, `FeedWater TDS`, and the target variable `Drum Water TDS`. The data is split into training, validation, test, and forecast sets.

## Methodology

### 1. Data Loading and Initial Preprocessing
- Data for training, validation, testing, and forecasting is loaded from separate data files.
- Irrelevant columns are dropped, retaining `Timestamp`, `Boiler Pressure P1`, `FeedWater TDS`, and `Drum Water TDS`.
- Missing values are handled by dropping rows containing `NaN` values.

### 2. Feature Engineering
- **Rolling Averages:** To capture temporal dependencies and smooth out short-term fluctuations, 7-day rolling averages were calculated for `Boiler Pressure P1` and `FeedWater TDS`.

### 3. Outlier Handling
- **IQR Outlier Capping:** Outliers in all features (`Boiler Pressure P1`, `FeedWater TDS`, `Boiler Pressure P1 Rolling`, `FeedWater TDS Rolling`) and the target variable (`Drum Water TDS`) were capped using the Interquartile Range (IQR) method to prevent extreme values from skewing the model.

### 4. Data Transformation
- **Box-Cox Transformation:** The target variable (`Drum Water TDS`) was transformed using the Box-Cox method to achieve a more Gaussian-like distribution, which can improve model performance. A `shift` was applied to ensure all values were positive before transformation.
- **Min-Max Scaling:** All features and the transformed target variable were scaled to a range between 0 and 1 using `MinMaxScaler`. This standardization is crucial for neural network models.

### 5. Sequence Generation
- **Time Series Sequences:** The scaled data was transformed into sequences suitable for LSTM input. Each sequence consists of 15 time steps of features, with the target being the value at the next time step.

### 6. LSTM Model Architecture
An LSTM (Long Short-Term Memory) neural network was chosen for its ability to model sequential data and capture long-term dependencies. The model consists of:
- An `Input` layer defining the shape of the input sequences.
- An `LSTM` layer with 64 units, returning only the last output (not sequences), and `L2 regularization` (0.0005) to prevent overfitting.
- A `Dropout` layer (0.2) for further regularization.
- A `Dense` output layer with a single unit for regression.

The model was compiled with the `Adam` optimizer and `MAE` (Mean Absolute Error) as the loss function, and `MSE` (Mean Squared Error) as an additional metric.

### 7. Training and Callbacks
- The model was trained for 50 epochs with a `batch_size` of 64.
- **Early Stopping:** Training was stopped if the `val_loss` did not improve for 5 consecutive epochs, and the best weights were restored.
- **Model Checkpoint:** The best performing model weights (based on validation loss) were saved.

### 8. Prediction and Inverse Transformation
- Predictions were generated for both the test and forecast sets.
- The predictions were then subjected to inverse Min-Max scaling and inverse Box-Cox transformation (with the previously calculated `boxcox_lambda` and `shift`) to convert them back to the original scale of `Drum Water TDS`.

### 9. Bias Correction
- A critical step involved calculating and correcting the mean bias observed in the initial predictions. The bias was determined by averaging the difference between actual and predicted values on the test set and then subtracting this bias from both test and forecast predictions. This significantly improved the accuracy and reduced the systematic error of the model.

## Results and Evaluation
The model's performance was evaluated using standard regression metrics. The bias correction step notably enhanced the prediction accuracy.

### Test Set (After Bias Correction)
- **MAE:** 66.77
- **MAPE:** 17.20%
- **MSE:** 8194.70
- **RMSE:** 90.52

### Forecast Set (After Bias Correction)
- **MAE:** 50.98
- **MAPE:** 11.38%
- **MSE:** 4238.36
- **RMSE:** 65.10

### Visualizations
Below are plots illustrating the actual vs. predicted values for both the test and forecast sets, showcasing the effectiveness of the bias correction:

**Test Set: Original vs. Bias Corrected Prediction**
![Test Set Bias Correction Plot](test_bias_correction.png)

**Forecast Set: Original vs. Bias Corrected Prediction**
![Forecast Set Bias Correction Plot](forecast_bias_correction.png)


## Installation
To run this project, you will need the following Python libraries. It is recommended to create a virtual environment.

```bash
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow scipy openpyxl
