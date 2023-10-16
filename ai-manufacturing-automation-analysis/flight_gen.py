# 1. Data Generation
# - Create synthetic flight sensor data with randomness 
# - Visualize with histograms and scatterplots 

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
import warnings
warnings.filterwarnings("ignore", message="The figure layout has changed to tight")


# Flight conditions
num_flights = 1000
# altitude = np.random.normal(7000, 50, num_flights)
# velocity = np.random.normal(100, 5, num_flights) 

altitude = np.random.normal(3000, 1000, num_flights)  # Mean: 30000 feet, Std: 2000 feet
velocity = np.random.normal(200, 50, num_flights)       # Mean: 250 km/h, Std: 20 km/h

#fuel_level = np.random.uniform(0, 100, num_flights)
#fuel_level = np.random.uniform(0, 101, num_flights)  # Range: 20-80% fuel level
average_fuel = 45
std_dev_fuel = 15
lower_bound = 0
upper_bound = 100

fuel_level = np.random.normal(average_fuel, std_dev_fuel, num_flights)
fuel_level = np.clip(fuel_level, lower_bound, upper_bound)

#engine_temp = np.random.normal(100, 15, num_flights)
engine_temp = np.random.normal(250, 30, num_flights)  # Mean: 250°C, Std: 30°C

energy_consumption = np.random.normal(25, 5, num_flights)

# Cabin conditions  
#velocity = np.random.normal(200, 20, num_flights)       # Mean: 250 km/h, Std: 20 km/h
cabin_temp = np.random.normal(85, 5, num_flights)
cabin_humidity = np.random.normal(90, 5, num_flights)
cabin_pressure = np.random.normal(1, 1, num_flights)
cabin_pressure = np.clip(cabin_pressure, lower_bound, upper_bound)

# Passenger data
passenger_count = np.random.normal(3, 1, num_flights) 

# Comfort levels
comfort_temp = np.random.normal(3, 1, num_flights)
comfort_humidity = np.random.normal(3, 1, num_flights) 

# Additional features
battery_charge = np.random.uniform(0, 100, num_flights)

vibration_level = np.random.normal(0.2, 0.1, num_flights)
vibration_level = np.clip(vibration_level, lower_bound, upper_bound)

noise_level = np.random.normal(62, 5, num_flights)
wind_speed = np.random.normal(10, 3, num_flights)
precipitation = np.random.standard_cauchy(num_flights)

navigation_error = np.random.normal(0, 1, num_flights)
navigation_error = np.clip(navigation_error, lower_bound, upper_bound)

structural_integrity = np.random.normal(0.9, 0.05, num_flights)
component_reliability = np.random.normal(0.95, 0.02, num_flights)

flight_incidents = np.random.standard_t(100, num_flights)
flight_incidents = np.clip(flight_incidents, lower_bound, upper_bound)

maintenance_indicators = np.random.normal(0.7, 0.1, num_flights)
operational_costs = abs(np.random.normal(1500, 100, num_flights))

# Generate flight phases based on random selection
flight_phase = np.random.choice(['takeoff', 'cruise', 'landing'], num_flights, p=[0.2, 0.6, 0.2])


df = pd.DataFrame({
    'altitude': altitude, 
    'velocity': velocity,
    'fuel_level': fuel_level,
    'engine_temp': engine_temp,
    
    'cabin_temp': cabin_temp,
    'cabin_humidity': cabin_humidity,
    'cabin_pressure': cabin_pressure,
    
    'passenger_count': passenger_count,
    
    'comfort_temp': comfort_temp,
    'comfort_humidity': comfort_humidity,

    'battery_charge': battery_charge,
    'energy_consumption': energy_consumption, 
    'vibration_level': vibration_level,
    'noise_level': noise_level,
    'wind_speed': wind_speed,
    'precipitation': precipitation,
    'navigation_error': navigation_error,
    'structural_integrity': structural_integrity,
    'component_reliability': component_reliability,
    'flight_incidents': flight_incidents,
    'maintenance_indicators': maintenance_indicators,
    'operational_costs': operational_costs,
    'flight_phase': flight_phase  # Include flight_phase column
})


# Fuel level and battery charge are positive only
# Use QuantileTransformer to normalize distribution
# QuantileTransformer for non-Gaussian distributions
quant_transformer = QuantileTransformer(output_distribution='normal')
df['fuel_level'] = abs(quant_transformer.fit_transform(df[['fuel_level']]))
df['battery_charge'] = abs(quant_transformer.fit_transform(df[['battery_charge']]))
df['precipitation'] = abs(quant_transformer.fit_transform(df[['precipitation']]))

# PowerTransformer for power law distributions
power_transformer = PowerTransformer()
#df['flight_incidents'] = abs(power_transformer.fit_transform(df[['flight_incidents']]))

print(df.head())


# Calculate summary statistics
summary_stats = df.describe()

# Calculate percentiles
#percentiles = np.percentile(df, [25, 50, 75], axis=0)
# Calculate percentiles for each column
numeric_columns = df.select_dtypes(include=np.number).columns
percentiles = np.percentile(df[numeric_columns], [25, 50, 75], axis=0)

# Create a DataFrame to store the percentiles
percentiles_df = pd.DataFrame(percentiles, columns=numeric_columns, index=[25, 50, 75])

# Print the percentiles DataFrame
print("Percentiles:")
print(percentiles_df)

print(summary_stats)
print("Flight Phase Analysis")
phase_grouped = df.groupby('flight_phase').mean()
print(phase_grouped)

# Calculate the correlation matrix
correlation_matrix = df[numeric_columns].corr()
#correlation_matrix = df.corr()


# # Histogram plots
# df.hist()
# # Apply tight layout
# plt.tight_layout()
# plt.show()

# Scatter plot
plt.scatter(df['altitude'], df['velocity'])
plt.xlabel('Altitude')
plt.ylabel('Velocity')
plt.title("Scatter")
# Apply tight layout
plt.tight_layout()
plt.show()

# Create a correlation heatmap using Seaborn
# Create a larger figure
plt.figure(figsize=(12, 10))

# Define a custom color map with distinct colors for each variable
cmap = sns.color_palette("tab10", as_cmap=True)

# Create the correlation heatmap using the custom color map
sns.heatmap(correlation_matrix, annot=True, cmap=cmap, center=0)
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap of Flight Variables")
# Apply tight layout
plt.tight_layout()
plt.show()

sns.boxplot(data=df[['altitude', 'velocity', 'energy_consumption']])
plt.title("Outliers")
# Apply tight layout
plt.tight_layout()
plt.show()

# # Create the correlation heatmap with vibrant colors
# #sns.heatmap(correlation_matrix, annot=True, cmap=cmap, center=0)
# sns.pairplot(df, vars=['altitude', 'velocity', 'energy_consumption'], palette=cmap)
# plt.suptitle("Relationship Heatmap")
# plt.title("A/V/E Relationships")
# # Apply tight layout
# plt.tight_layout()
# plt.show()
# Create a custom color palette with distinct colors for each variable
# variable_palette = sns.color_palette("tab10", n_colors=len(df.columns))

# # Create a scatter matrix using pandas.plotting
# pd.plotting.scatter_matrix(df[['altitude', 'velocity', 'energy_consumption']], figsize=(10, 8), color=variable_palette)

# plt.suptitle("Relationship Heatmap")
# plt.title("A/V/E Relationships")
# List of variable names
variables = ['altitude', 'velocity', 'energy_consumption']

# Create a custom color palette with distinct colors for each variable
variable_palette = sns.color_palette("tab10", n_colors=len(variables))

# Create a scatter plot matrix using loops
fig, axes = plt.subplots(nrows=len(variables), ncols=len(variables), figsize=(10, 8))

for i, row_var in enumerate(variables):
    for j, col_var in enumerate(variables):
        if i == j:
            # Histograms on the diagonal
            sns.histplot(df[row_var], kde=True, color=variable_palette[i], ax=axes[i, j])
        else:
            # Scatter plots on off-diagonal
            sns.scatterplot(x=row_var, y=col_var, data=df, color=variable_palette[i], ax=axes[i, j])

# Set plot labels and titles
plt.suptitle("Scatter Plot Matrix")
# Apply tight layout
plt.tight_layout()
plt.show()


df.hist(bins=20, figsize=(15, 10))
plt.suptitle("Histogram Data Points")
plt.title("Histogram")
# Apply tight layout
plt.tight_layout()
plt.show()

# Big Gambit - wow?

# List of variable names
variables = df.columns

# Create a custom color palette with distinct colors for each variable
variable_palette = sns.color_palette("tab20", n_colors=len(variables))

# Create a scatter plot heatmap using loops
fig, axes = plt.subplots(nrows=len(variables), ncols=len(variables), figsize=(12, 10))

for i, row_var in enumerate(variables):
    for j, col_var in enumerate(variables):
        ax = axes[i, j]
        if i == j:
            # Histograms on the diagonal
            sns.histplot(df[row_var], kde=True, color=variable_palette[i], ax=ax)
            # ax.set_xlabel('')
            # ax.set_ylabel('')
            # ax.set_yticks([])  # Remove y-axis ticks
            # ax.set_xticklabels([])  # Remove x-axis labels
            ax.set_title(row_var, rotation=0, ha="right")  # Rotate title
        else:
            # Scatter plots on off-diagonal
            sns.scatterplot(x=row_var, y=col_var, data=df, color=variable_palette[i], marker="o", s=10, alpha=0.6, ax=ax)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xticks([])
            ax.set_yticks([])

# Set plot labels and titles
plt.suptitle("Scatter Heatmap with Variable Colors")
#plt.tight_layout()

# Adjust spacing between subplots
#plt.subplots_adjust(wspace=0.1, hspace=0.1)

# Show the plot
plt.show()



# save the data
#df.to_csv("joby_aviation_data.csv", index=False)