# Four Types of Inventory Forecasting
# There are four basic approaches you may consider for inventory forecasting.

# Trend forecasting: Project possible trends using changes in demand for your product over time. This doesn’t always account for seasonality or other irregularities in past sales data.
# Graphical forecasting: By graphing historic data, you can identify patterns and add slopped trend lines to identify possible insights that may have been missed without the visual representation.
# Qualitative forecasting: Qualitative forecasting usually involves focus groups and market research. Forecasters then flesh out models from this type of data.
# Quantitative forecasting: This uses past numerical data to predict future demand. The more data gathered, the more accurate the forecast usually is.


import pandas as pd
import matplotlib.pyplot as plt
import gen_data as g
import ai_duties as ad

#data = pd.read_csv('./stock.csv')
data = g.data_simulation_generator(100)
df = pd.DataFrame(data)
#df = pd.DataFrame(data, columns = ['ValueDate', 'Price'])

# Set the Date as Index, but don't drop the ValueDate column
#df = df.set_index('ValueDate', drop=False)
df = df.set_index('Date', drop=False)

# Plot the DataFrame
df.plot(figsize=(5, 3))
plt.show()

ad.identify_patterns(data)