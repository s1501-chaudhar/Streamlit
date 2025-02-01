import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# App title
st.title("Enhanced Stock Price Analysis and Forecasting App")
st.write("Upload a stock price dataset to analyze trends and forecast future prices.")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file with 'Date' and 'Close' columns", type="csv")

if uploaded_file:
    # Load the dataset
    data = pd.read_csv(uploaded_file)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    
    if 'Close' not in data.columns:
        st.error("Dataset must contain a 'Close' column.")
    else:
        st.write("### Dataset Preview")
        st.dataframe(data.head())
        
        # Candlestick Chart
        st.write("### Interactive Candlestick Chart")
        fig = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data['Open'] if 'Open' in data.columns else data['Close'],
            high=data['High'] if 'High' in data.columns else data['Close'],
            low=data['Low'] if 'Low' in data.columns else data['Close'],
            close=data['Close']
        )])
        fig.update_layout(title="Stock Price Candlestick Chart", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig)

        # Forecasting Options
        st.write("### Forecasting Options")
        forecasting_method = st.selectbox("Select Forecasting Method", ["ARIMA", "LSTM"])
        forecast_steps = st.slider("Select number of days to forecast", 1, 365, 30)
        
        # Data Preparation
        train_size = int(len(data) * 0.8)
        train, test = data['Close'][:train_size], data['Close'][train_size:]
        
        if forecasting_method == "ARIMA":
            try:
                model = ARIMA(train, order=(5, 1, 0))
                model_fit = model.fit()
                st.write("ARIMA Model Summary:")
                st.text(model_fit.summary())
                
                # Forecast
                forecast = model_fit.forecast(steps=forecast_steps)
                st.write("### ARIMA Forecasted Prices")
                fig, ax = plt.subplots()
                ax.plot(data.index, data['Close'], label="Historical Prices")
                ax.plot(pd.date_range(data.index[-1], periods=forecast_steps + 1, freq='D')[1:], forecast, label="Forecasted Prices", linestyle="--")
                ax.set_title("Historical and Forecasted Prices")
                ax.set_xlabel("Date")
                ax.set_ylabel("Price")
                ax.legend()
                st.pyplot(fig)
                
                # Forecast metrics
                predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)
                mse = mean_squared_error(test, predictions)
                st.write(f"Mean Squared Error on Test Set: {mse:.2f}")
            except Exception as e:
                st.error(f"Error while fitting ARIMA model: {e}")

        elif forecasting_method == "LSTM":
            try:
                # Scaling data
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
                
                # Prepare training data
                def create_dataset(data, time_step=60):
                    X, y = [], []
                    for i in range(len(data) - time_step - 1):
                        X.append(data[i:(i + time_step), 0])
                        y.append(data[i + time_step, 0])
                    return np.array(X), np.array(y)
                
                time_step = 60
                X_train, y_train = create_dataset(scaled_data[:train_size], time_step)
                X_test, y_test = create_dataset(scaled_data[train_size:], time_step)
                
                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                
                # LSTM model
                model = Sequential([
                    LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
                    LSTM(50, return_sequences=False),
                    Dense(25),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)
                
                # Forecast
                last_60_days = scaled_data[-time_step:]
                forecast_input = last_60_days.reshape(1, time_step, 1)
                forecast = []
                for _ in range(forecast_steps):
                    pred = model.predict(forecast_input, verbose=0)
                    forecast.append(pred[0, 0])
                    forecast_input = np.append(forecast_input[:, 1:, :], [[pred[0]]], axis=1)
                
                forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
                st.write("### LSTM Forecasted Prices")
                fig, ax = plt.subplots()
                ax.plot(data.index, data['Close'], label="Historical Prices")
                ax.plot(pd.date_range(data.index[-1], periods=forecast_steps + 1, freq='D')[1:], forecast, label="Forecasted Prices", linestyle="--")
                ax.set_title("Historical and Forecasted Prices")
                ax.set_xlabel("Date")
                ax.set_ylabel("Price")
                ax.legend()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error while training LSTM model: {e}")

else:
    st.write("Please upload a dataset to proceed.")

# Footer
st.write("Developed by Shardul")
