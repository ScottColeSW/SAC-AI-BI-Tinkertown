# Imports 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data
dates = pd.date_range('2023-01-01', periods=365, freq='D')
data = pd.DataFrame({
    'revenue': np.random.randint(30000, 100000, len(dates)),
    'passengers': np.random.randint(300, 1000, len(dates)),
    'price': np.random.normal(250, 50, len(dates)),
    'demand_forecast': np.random.normal(950, 100, len(dates)),
    'new_customers': np.random.randint(100, 300, len(dates)),
    'delay_mins': np.random.randint(10, 45, len(dates)),
}, index=dates)

# Figure 
fig = plt.figure(figsize=(8, 8))
fig, ax = plt.subplots()

gs = fig.add_gridspec(ncols=2, nrows=3)
# import seaborn as sns

# # Create a figure and axes
# fig, ax = plt.subplots()

# # Plot some data
# sns.lineplot(x=x, y=y, ax=ax)

# # Tilt the axis labels
# ax.set_xticklabels(rotation=45)
# ax.set_yticklabels(rotation=45)

# # Show the figure
# plt.show()

# Revenue timeseries
ax = fig.add_subplot(gs[0, :])
sns.lineplot(data=data, x=data.index.month_name(), y='revenue', ax=ax)
ax.set_title('Daily Revenue')

# Price distribution
ax = fig.add_subplot(gs[1, 0]) 
sns.histplot(data=data, x='price', stat='density', kde=True, ax=ax)
ax.set_title('Ticket Price Distribution')

# Passenger forecast
ax = fig.add_subplot(gs[1, 1])
sns.lineplot(data=data, x=data.index.month_name(), y='passengers', label='Actual', ax=ax)
sns.lineplot(data=data, x=data.index.month_name(), y='demand_forecast', label='Predicted', ax=ax)
ax.set_title('Passenger Demand')

# Customer metrics
ax = fig.add_subplot(gs[2, 0])
sns.barplot(data=data, x=data.index.month_name(), y='new_customers', ax=ax)
ax.set_title('New Customers') 

# Revenue by route
ax = fig.add_subplot(gs[2, 1]) 
rev_by_route = [30, 20, 15, 25, 10]
plt.pie(rev_by_route, labels=['LAX', 'SFO', 'JFK', 'ORD', 'DFW'])
ax.set_title('Revenue by Top Routes')

# Average delay
ax = fig.add_subplot(gs[:, -1]) 
sns.kdeplot(data=data, x='delay_mins', label='Avg Delay', ax=ax)

# Tilt the axis labels
plt.setp(plt.get_figlabels(), rotation=45)

# Layout
fig.tight_layout()
plt.show()