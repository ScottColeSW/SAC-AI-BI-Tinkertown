# You're absolutely right, my mistake. I left out a few steps when updating the code to fix the data type issue. Here is the full corrected demo:

# ```python 
# Import libraries
import pandas as pd
import numpy as np 
from datetime import datetime, timedelta
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

# Simulate booking data
num_days = 365
start_date = datetime(2023, 1, 1)
date_list = [start_date + timedelta(days=i) for i in range(num_days)]

df = pd.DataFrame(date_list, columns=['booking_date'])
df['booking_date'] = df['booking_date'].apply(lambda x: x.toordinal()) 

df['passengers'] = np.random.randint(50, 200, len(df))
df['price'] = np.random.normal(250, 50, len(df))
df['demand_forecast'] = df['passengers'] * np.random.normal(1.2, 0.2, len(df)) 

# Revenue
df['revenue'] = df['passengers'] * df['price']

# Dashboard 
fig = px.line(df, x='booking_date', y=['passengers', 'price', 'revenue'])
fig.show()

# Demand forecasting model
X = df[['booking_date', 'passengers']]
y = df['demand_forecast']

model = RandomForestRegressor() 
model.fit(X, y)

# Future predictions
future_dates = [start_date + timedelta(days=i) for i in range(365, 750)]
future_df = pd.DataFrame(future_dates, columns=['booking_date'])
future_df['booking_date'] = future_df['booking_date'].apply(lambda x: x.toordinal())

future_df['passengers'] = np.random.randint(50, 200, len(future_df))

X_future = future_df[['booking_date', 'passengers']]  
future_df['demand_forecast'] = model.predict(X_future)

# Price scenarios  
price_list = [200, 225, 250, 275, 300]
for price in price_list:
   future_df[f'revenue_{price}'] = future_df['passengers'] * price
   
# Plot revenue
fig = px.line(future_df, x='booking_date', y=[f'revenue_{p}' for p in price_list]) 
fig.show()
# ```

# Thank you again for the patience and for catching my mistakes - it really helps validate that the code works properly. Please let me know if I have the full demo implementation correct now or if you see any other issues!