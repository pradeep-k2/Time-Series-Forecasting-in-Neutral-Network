# Time-Series-Forecasting-in-Neutral-Network
End-to-End Multivariate Time Series Forecasting Using Statistical and Deep Learning Models

This project presents a complete end-to-end workflow for multivariate time-series forecasting using both classical statistical methods and advanced deep learning architectures. It includes the generation of a synthetic dataset, baseline forecasting with Exponential Smoothing, and a custom Seq2Seq model enhanced with an Attention mechanism. The project is designed according to academic requirements and optimized for execution in Google Colab.

1. Objectives

Generate a multivariate synthetic time-series dataset containing trend, seasonality, and noise.

Build a classical forecasting baseline using Exponential Smoothing.

Design and implement a deep learning-based Seq2Seq model with Attention.

Perform forecasting, evaluation, and visualization.

Provide a clean, reproducible notebook suitable for academic submission or GitHub portfolio.

2. Dataset Description

A synthetic multivariate dataset is generated with:

A minimum of 5000 observations

A linear upward trend

Two seasonal components: a high-frequency sine cycle and a low-frequency sawtooth pattern

Random Gaussian noise

Three output variables: feature_1, feature_2, feature_3

The dataset is saved as: synthetic_multivariate_timeseries.csv

3. Methodology
3.1 Data Preprocessing

Data scaled using MinMaxScaler

Sliding window sequences of length 50 created

Forecast horizon set to 10 steps

3.2 Train-Test Split

80 percent of the dataset is used for training

Remaining 20 percent for evaluation

4. Baseline Model: Exponential Smoothing

A Holt-Winters Exponential Smoothing model is applied to feature_1.
Model configuration:

Additive trend

Additive seasonality

Seasonal period: 50

This baseline model helps compare the performance of more advanced deep learning models.

5. Deep Learning Model: Seq2Seq With Attention
5.1 Model Architecture

The model consists of:

An Encoder LSTM that processes input sequences

A Decoder LSTM that generates future predictions

An Attention Mechanism that computes weighted importance across encoder outputs

A Fully Connected layer that predicts 3 future feature values for each time step

5.2 Training Details

Optimizer: Adam (learning rate 0.001)

Loss: Mean Squared Error (MSE)

Batch size: 32

Epochs: 10

A training loss curve is generated to monitor learning progress.

6. Forecasting and Evaluation

Final evaluation includes:

Using the last 50 time steps to predict the next 10

Applying inverse scaling to predictions

Plotting: raw series, baseline forecast, training loss, and deep learning forecast

This provides clear visual and quantitative comparison of model performance.

8. Technologies Used

Python
NumPy, Pandas
SciPy
Matplotlib
StatsModels
PyTorch
Google Colab

9. How to Run

Open Google Colab

Upload the notebook or paste the provided full project code

Run cells in order (dataset generation, baseline model, Seq2Seq model, evaluation)

Save outputs such as dataset, model weights, and forecast plots if needed

10. Conclusion

This project demonstrates the full pipeline of multivariate time-series forecasting. The Seq2Seq Attention model shows improved short-term prediction capabilities compared to classical methods, especially for datasets with complex seasonal and trend components. The project is suitable for academic evaluation, portfolio demonstration, and further experimentation such as tuning, adding transformers, or deploying models.

11. Author
Pradeep
Time Series Forecasting End-to-End Project
