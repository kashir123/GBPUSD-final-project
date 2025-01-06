**Predicting the Price and Developing trading strategy for GBP/USD using Machine Learning Algorithm**

**Research Questions:**
1. Can Machine Learning Models accurately predict the price of GBP/USD using the historical
data and technical indicators?
2. Can Machine Learning Models used to create the profitable trading strategy?
**Project Objectives:**
1. Collect and preprocess the historical data and perform EDA on it and understand the
      patterns.
2. Build and compare the machine learning models (LSTM, GRU, ARIMA, Random Forest,
      XGBOOST) in predicting the price of GBP/USD.
3. Evaluate the performance of the model.
4. Build and analyze the trading strategy for different Models and evaluate them which model
is giving profitable results.

**Features**
- Predict GBP/USD exchange rates using machine learning.
- Implement deep learning models (LSTM, GRU) and ensemble models (Random Forest, XGBoost).
- Incorporates technical indicators such as:
   - Exponential Moving Average (EMA)
   - Simple Moving Average (SMA)
   - Relative Strength Index (RSI)
   - Moving Average Convergence Divergence (MACD)
- Data preprocessing includes normalization and sliding window techniques for sequential modeling.
- Backtesting to validate model predictions and assess profitability.

**Technologies Used**
- Programming Language: Python
- Libraries:
  - Pandas
  - NumPy
  - Matplotlib
  - Seaborn
  - TensorFlow/Keras
  - Scikit-Learn
  - Plotly
**Dataset**

- Source: https://uk.finance.yahoo.com/quote/GBPUSD%3DX/history/?guccounter=1
- Description: The dataset contains historical daily GBP/USD exchange rates, including Open, High, Low, and Close prices. Additional technical indicators are computed to enrich the feature set.
- Preprocessing: Missing values were handled, and data normalization (Min-Max Scaling) was applied.

**Model Architecture**

The project implements the following models for price prediction:
- LSTM (Long Short-Term Memory): Captures long-term dependencies in sequential data.
- GRU (Gated Recurrent Unit): A lightweight alternative to LSTM that reduces computational complexity.
- Random Forest: Ensemble-based model effective for regression tasks on structured data.
- XGBoost: Boosted decision tree model optimized for speed and accuracy.

**Evaluation Metrics**

The model’s performance is evaluated using the following metrics:
	- Mean Squared Error (MSE)
	- Mean Absolute Error (MAE)
	- Root Mean Squared Error (RMSE)
	- R-squared (R²)
**Results**
- The GRU model demonstrated the highest accuracy with an R² of 0.99 and MAE of 0.0041.
- LSTM followed closely with an R² of 0.85.
- Random Forest and XGBoost models performed well after hyperparameter tuning, achieving R² scores of 0.84 and 0.79, respectively.
**Backtesting and Strategy**
- Backtesting was conducted to simulate trading scenarios using model predictions.
- A starting capital of £10,000 was used, with risk-adjusted returns measured using the Sharpe ratio.
**Ethical Considerations**
- The dataset used in this project is publicly available on Yahoo Finance and does not contain personal or sensitive information.
- All data was collected from reputable sources, ensuring compliance with ethical research practices.
- No personal data or GDPR-sensitive information was processed during the project.
**Contact**
- Author: Kashir Waseem
- Email: kashir.waseem116@gmail.com
