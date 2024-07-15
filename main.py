import streamlit as st
from datetime import date
import yfinance as yf
from plotly import graph_objs as go
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Prediction App')

stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Prepare data for forecasting
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
df_train['ds'] = pd.to_datetime(df_train['ds'])
df_train.set_index('ds', inplace=True)

# Fit the SARIMAX model
model = SARIMAX(df_train['y'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
model_fit = model.fit(disp=False)

# Make future predictions
future_dates = pd.date_range(start=df_train.index[-1], periods=period + 1)
future_dates = future_dates.to_frame(index=False, name='ds')

forecast = model_fit.get_forecast(steps=period)
forecast_df = forecast.summary_frame()

# Combine forecast with future dates
forecast_df['ds'] = future_dates['ds']
forecast_df.rename(columns={'mean': 'yhat'}, inplace=True)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast_df.tail())

st.write(f'Predict plot for {n_years} years')
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], name='Predicted Close'))
fig1.add_trace(go.Scatter(x=df_train.index, y=df_train['y'], name='Actual Close'))
fig1.layout.update(title_text='Stock Price Forecast', xaxis_rangeslider_visible=True)
st.plotly_chart(fig1)

st.write("Predict components")
# Since SARIMAX does not have built-in components plot, we can show the mean and confidence intervals instead
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], name='Predicted Close'))
fig2.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['mean_ci_lower'], name='Lower Confidence Interval', fill='tonexty'))
fig2.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['mean_ci_upper'], name='Upper Confidence Interval', fill='tonexty'))
st.plotly_chart(fig2)
