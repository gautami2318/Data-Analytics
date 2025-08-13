#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install notebook --upgrade')


# In[2]:


import pandas as pd

nifty50_data = pd.read_csv("nifty50_closing_prices.csv")

nifty50_data.head()


# In[3]:


# check for missing values
missing_values = nifty50_data.isnull().sum()

# check for date column format
date_format_check = pd.to_datetime(nifty50_data['Date'], errors='coerce').notna().all()

# check if the data has sufficient rows for time-series analysis
sufficient_rows = nifty50_data.shape[0] >= 20  # Minimum rows needed for rolling/moving averages

# preparing a summary of the checks
data_preparation_status = {
    "Missing Values in Columns": missing_values[missing_values > 0].to_dict(),
    "Date Column Format Valid": date_format_check,
    "Sufficient Rows for Time-Series Analysis": sufficient_rows
}

data_preparation_status


# In[4]:


# drop the HDFC.NS column since it contains 100% missing values
nifty50_data = nifty50_data.drop(columns=['HDFC.NS'])

# convert the 'Date' column to datetime format
nifty50_data['Date'] = pd.to_datetime(nifty50_data['Date'])

# sort the dataset by date to ensure proper time-series order
nifty50_data = nifty50_data.sort_values(by='Date')

# reset index for a clean dataframe
nifty50_data.reset_index(drop=True, inplace=True)


# In[5]:


# calculate descriptive statistics
descriptive_stats = nifty50_data.describe().T  # Transpose for better readability
descriptive_stats = descriptive_stats[['mean', 'std', 'min', 'max']]
descriptive_stats.columns = ['Mean', 'Std Dev', 'Min', 'Max']
print(descriptive_stats)


# In[6]:


# assign weights to a subset of stocks (example: RELIANCE.NS, HDFCBANK.NS, ICICIBANK.NS)
weights = [0.4, 0.35, 0.25]
portfolio_data = nifty50_data[['RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS']]

# calculate daily returns
daily_returns = portfolio_data.pct_change().dropna()

# calculate portfolio returns
portfolio_returns = (daily_returns * weights).sum(axis=1)

# display portfolio returns
portfolio_returns.head()


# In[7]:


# Calculate standard deviation (volatility)
volatility = daily_returns.std()

# Calculate VaR (95% confidence level)
confidence_level = 0.05
VaR = daily_returns.quantile(confidence_level)

# Display risk metrics
risk_metrics = pd.DataFrame({'Volatility (Std Dev)': volatility, 'Value at Risk (VaR)': VaR})
print(risk_metrics)


# In[9]:


get_ipython().system('pip install plotly')


# In[4]:


import pandas as pd
import plotly.figure_factory as ff

# Example stock price data - replace this with your actual prices DataFrame
prices = pd.DataFrame({
    'AAPL': [150, 152, 153, 155, 154],
    'GOOG': [2700, 2725, 2710, 2730, 2750],
    'MSFT': [300, 305, 310, 315, 312]
})

# Calculate daily returns as percentage change and drop the first NA row
daily_returns = prices.pct_change().dropna()

# Calculate correlation matrix
correlation_matrix = daily_returns.corr()

# Create annotated heatmap of correlation matrix
fig = ff.create_annotated_heatmap(
    z=correlation_matrix.values,
    x=list(correlation_matrix.columns),
    y=list(correlation_matrix.index),
    annotation_text=correlation_matrix.round(2).values,
    colorscale='RdBu', 
    showscale=True
)

fig.update_layout(
    title="Correlation Matrix of Stock Returns",
    title_x=0.5,
    font=dict(size=12),
    plot_bgcolor='white',
    paper_bgcolor='white',
)

fig.show()


# In[9]:


import yfinance as yf
import plotly.graph_objects as go
import pandas as pd

# Download historical data for RELIANCE.NS
nifty50_data = yf.download('RELIANCE.NS', start='2023-01-01', end='2023-08-01')

# Reset index to have 'Date' column
nifty50_data.reset_index(inplace=True)

# Calculate moving averages
nifty50_data['RELIANCE_5d_MA'] = nifty50_data['Close'].rolling(window=5).mean()
nifty50_data['RELIANCE_20d_MA'] = nifty50_data['Close'].rolling(window=20).mean()

# Plotting
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=nifty50_data['Date'],
    y=nifty50_data['Close'],
    mode='lines',
    name='RELIANCE.NS Price'
))

fig.add_trace(go.Scatter(
    x=nifty50_data['Date'],
    y=nifty50_data['RELIANCE_5d_MA'],
    mode='lines',
    name='5-Day MA'
))

fig.add_trace(go.Scatter(
    x=nifty50_data['Date'],
    y=nifty50_data['RELIANCE_20d_MA'],
    mode='lines',
    name='20-Day MA'
))

fig.update_layout(
    title="Moving Averages for RELIANCE.NS",
    xaxis_title="Date",
    yaxis_title="Price",
    template="plotly_white",
    legend=dict(title="Legend")
)

fig.show()


# In[11]:


import yfinance as yf
import plotly.graph_objects as go
import pandas as pd

# Step 1: Download data for RELIANCE.NS
nifty50_data = yf.download('RELIANCE.NS', start='2023-01-01', end='2023-08-01')
nifty50_data.reset_index(inplace=True)  # Make Date a column

# Step 2: Define RSI calculation function
def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Step 3: Calculate RSI for the 'Close' price of RELIANCE.NS
nifty50_data['RELIANCE_RSI'] = calculate_rsi(nifty50_data['Close'])

# Step 4: Plot RSI with overbought and oversold levels
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=nifty50_data['Date'],
    y=nifty50_data['RELIANCE_RSI'],
    mode='lines',
    name='RSI'
))

fig.add_trace(go.Scatter(
    x=nifty50_data['Date'],
    y=[70] * len(nifty50_data),
    mode='lines',
    line=dict(color='red', dash='dash'),
    name='Overbought (70)'
))

fig.add_trace(go.Scatter(
    x=nifty50_data['Date'],
    y=[30] * len(nifty50_data),
    mode='lines',
    line=dict(color='green', dash='dash'),
    name='Oversold (30)'
))

fig.update_layout(
    title="RSI for RELIANCE.NS",
    xaxis_title="Date",
    yaxis_title="RSI",
    template="plotly_white",
    legend=dict(title="Legend")
)

fig.show()


# In[7]:


import numpy as np

# calculate average returns and volatility
mean_returns = daily_returns.mean()
volatility = daily_returns.std()

# assume a risk-free rate
risk_free_rate = 0.04 / 252

# calculate sharpe ratio
sharpe_ratios = (mean_returns - risk_free_rate) / volatility

table_data = pd.DataFrame({
    'Stock': sharpe_ratios.index,
    'Sharpe Ratio': sharpe_ratios.values.round(2)
})

fig = go.Figure(data=[go.Table(
    header=dict(values=['Stock', 'Sharpe Ratio'],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[table_data['Stock'], table_data['Sharpe Ratio']],
               fill_color='lavender',
               align='left')
)])

fig.update_layout(
    title="Sharpe Ratios for Selected Stocks",
    template="plotly_white"
)

fig.show()


# In[13]:


get_ipython().system('pip install yfinance')


# In[12]:


import yfinance as yf
import numpy as np
import plotly.graph_objects as go

# Step 1: Download RELIANCE.NS data
nifty50_data = yf.download('RELIANCE.NS', start='2023-01-01', end='2023-08-01')
nifty50_data.reset_index(inplace=True)

# Step 2: Monte Carlo simulation parameters
num_simulations = 1000
num_days = 252

# Use the last closing price as the starting price
last_price = nifty50_data['Close'].iloc[-1]

# Calculate volatility as the standard deviation of daily returns
volatility = nifty50_data['Close'].pct_change().std()

# Initialize array to store simulated prices
simulated_prices = np.zeros((num_simulations, num_days))

# Step 3: Run simulations
for i in range(num_simulations):
    simulated_prices[i, 0] = last_price
    for j in range(1, num_days):
        simulated_prices[i, j] = simulated_prices[i, j - 1] * np.exp(
            np.random.normal(0, volatility)
        )

# Step 4: Plot simulations
fig = go.Figure()

for i in range(num_simulations):
    fig.add_trace(go.Scatter(
        x=list(range(num_days)),
        y=simulated_prices[i],
        mode='lines',
        line=dict(width=0.5),
        opacity=0.1,
        showlegend=False
    ))

fig.update_layout(
    title="Monte Carlo Simulation for RELIANCE.NS Prices",
    xaxis_title="Days",
    yaxis_title="Simulated Price",
    template="plotly_white"
)

fig.show()


# In[ ]:





# In[ ]:




