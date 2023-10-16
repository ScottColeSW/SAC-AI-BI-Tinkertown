# Imports 
import pandas as pd
import numpy as np

# Customer data
num_customers = 5000  
customer_data = pd.DataFrame({
  'age': np.random.normal(40, 10, num_customers),
  'satisfaction': np.random.randint(1, 10, num_customers),
  'flight_history': np.random.poisson(5, num_customers),
})

# Bookings timeseries
num_days = 365*2
dates = pd.date_range('2020-01-01', periods=num_days)
bookings = pd.DataFrame({
  'passengers': np.random.poisson(950, num_days),
  'price': np.random.normal(250, 50, num_days),
}, index=dates)

# Air traffic data
num_flights = 365*500
flights = pd.DataFrame({
  'airline': np.random.choice(['UA', 'AA', 'DL'], num_flights),
  'origin': np.random.choice(['SFO', 'LAX', 'ORD', 'JFK'], num_flights),
  'destination': np.random.choice(['SFO', 'LAX', 'ORD', 'JFK'], num_flights),
  'departure_time': np.random.randint(0,24, num_flights),
  
  # Add flight path data
  'flight_path': [np.random.choice(['A', 'B', 'C'], 10) for _ in range(num_flights)],
  
  # And runway/landing data
  'runway': np.random.choice([1, 2, 3, 4], num_flights),
  'landing_time': np.random.randint(0,24, num_flights)   
})