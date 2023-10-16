import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

# Create a DateFormatter object
date_formatter = mdates.DateFormatter('%b')

# Sample data
dates = pd.date_range('2023-01-01', periods=365*3, freq='D')
data = pd.DataFrame({
    'revenue': np.random.randint(30000, 100000, len(dates)),
    'passengers': np.random.randint(300, 1000, len(dates)),
    'price': np.random.normal(250, 50, len(dates)),
    'demand_forecast': np.random.normal(950, 100, len(dates)),
    'new_customers': np.random.randint(100, 300, len(dates)),
    'delay_mins': np.random.randint(10, 45, len(dates)),
}, index=dates)

# Figure
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(ncols=2, nrows=3)

# Revenue timeseries
ax = fig.add_subplot(gs[0, 0])
sns.lineplot(data=data, x=data.index.month_name(), y='revenue', ax=ax)
# ax.xaxis.set_major_formatter(date_formatter)
# ax.xaxis.set_major_locator(mticker.MaxNLocator(12))
# ax.set_xticklabels(data.index.month_name(), rotation=45, ha='right')
ax.set_title('Daily Revenue')

# Price distribution
ax = fig.add_subplot(gs[1, 0])
sns.histplot(data=data, x='price', stat='density', kde=True, ax=ax)
ax.set_title('Ticket Price Distribution')

# Passenger forecast
ax = fig.add_subplot(gs[1, 1])
sns.lineplot(data=data, x=data.index.month_name(), y='passengers', label='Actual', ax=ax)
sns.lineplot(data=data, x=data.index.month_name(), y='demand_forecast', label='Predicted', ax=ax)
# ax.xaxis.set_major_formatter(date_formatter)
# ax.set_xticklabels(data.index.month_name(), rotation=45, ha='right')
ax.set_title('Passenger Demand')

# Customer metrics
ax = fig.add_subplot(gs[2, 0])
sns.barplot(data=data, x=data.index.month_name(), y='new_customers', ax=ax)
# ax.xaxis.set_major_formatter(date_formatter)
# ax.set_xticklabels(data.index.month_name(), rotation=45, ha='right')
ax.set_title('New Customers')

# Revenue by route
ax = fig.add_subplot(gs[2, 1])
#fig, ax = plt.subplots(figsize=(10, 6))
rev_by_route = [30, 20, 15, 25, 10]
plt.pie(rev_by_route, labels=['LAX', 'SFO', 'JFK', 'ORD', 'DFW'], autopct='%.1f%%', radius=2)
ax.set_title('Revenue by Top Routes')

# Average delay
ax = fig.add_subplot(gs[0, -1])
sns.kdeplot(data=data, x='delay_mins', label='Avg Delay', ax=ax)
ax.set_title('Average Flight Delay')

# Layout
fig.tight_layout()

# Rotate all x-axis labels
for ax in fig.axes:
    ax.set_xticks(ax.get_xticks(), minor=False)
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')

plt.show()