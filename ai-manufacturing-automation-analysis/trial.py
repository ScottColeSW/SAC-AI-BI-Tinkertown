# Import libraries
import pandas as pd 
import numpy as np
from scipy.stats import poisson, norm, gamma  

import matplotlib.pyplot as plt

# Set plot style  
plt.style.use('seaborn-v0_8-whitegrid')
# import seaborn as sns
# sns.set_style("whitegrid")

# Set random seed for reproducibility  
np.random.seed(42)

# -----------------
# Data Simulation
# -----------------

# Simulation parameters
num_days = 365*5  

# Create dataframe with date index
dates = pd.date_range(end='2023-12-31', periods=num_days)  
data = pd.DataFrame(index=dates)

# Simulate time series data for each risk indicator

# Cyber risk indicators
data['Attacks'] = np.random.randint(10, 100, num_days)
data['Malware'] = np.random.normal(50, 10, num_days).astype(int)
data['Vulnerabilities'] = np.random.logistic(loc=20, scale=5, size=num_days).astype(int) 
data['SecurityBudget'] = np.random.uniform(15000, 25000, size=num_days)

# Supply chain risk indicators
data['DeliveryDelay'] = np.random.gamma(5, 2, size=num_days)  
data['CommodityPrices'] = np.random.normal(20, 5, num_days)
data['Inventory'] = np.random.randint(100, 500, num_days)
data['TransportCost'] = np.random.lognormal(mean=5, sigma=1, size=num_days)

# Regulatory risk indicators
data['AuditDeficiencies'] = np.random.poisson(3, num_days)
data['ComplianceFailures'] = np.random.binomial(100, 0.05, num_days) 
data['PolicyChanges'] = np.random.choice([0, 1], size=num_days, p=[0.95, 0.05])

# Write simulated data to csv  
data.to_csv('simulated_data.csv')

# ----------------- 
# Risk Modeling
# -----------------

# Load simulated data
data = pd.read_csv('simulated_data.csv', index_col=0) 

# Monte carlo parameters
num_samples = 1000

# Define risk scoring weights  
weights = {'Attacks': 0.2, 'Malware': 0.15, 'Vulnerabilities': 0.15, 'SecurityBudget': 0.5}

# Initialize risk arrays  
cyber_risk = np.empty(num_samples)
supplychain_risk = np.empty(num_samples)
regulatory_risk = np.empty(num_samples)

# Run Monte Carlo simulation
for i in range(num_samples):

    # Simulate cyber risk 
    attacks = poisson.rvs(data['Attacks'].mean())
    malware = norm.rvs(loc=data['Malware'].mean(), scale=data['Malware'].std())
    vuln = np.random.logistic(loc=data['Vulnerabilities'].mean(), scale=data['Vulnerabilities'].std())
    budget = np.random.triangular(left=data['SecurityBudget'].min(), mode=data['SecurityBudget'].mean(), right=data['SecurityBudget'].max())


    # Calculate risk score
    cyber_risk[i] = (weights['Attacks']*attacks + 
                    weights['Malware']*malware +
                    weights['Vulnerabilities']*vuln +
                    weights['SecurityBudget']*budget)

# Simulate other risk scores
# ...

# Clip extreme values    
cyber_risk = np.clip(cyber_risk, -1000, 1000) 

# ------------------
# Analysis & Visualization
# ------------------

# Create risk data frame
risk_data = pd.DataFrame({'Cyber': cyber_risk, 'SupplyChain': supplychain_risk, 'Regulatory': regulatory_risk})  

# Summary statistics
print(risk_data.describe())

# Time series plot 
# fig, axs = plt.subplots(4, 1, figsize=(10, 8))
# data.plot(subplots=True)
fig, axs = plt.subplots(ncols=11, figsize=(16, 8))
data.plot(subplots=True, ax=axs)
plt.tight_layout()
plt.show()

# Risk distribution plot
risk_data.plot.hist(alpha=0.5, bins=50)
plt.show()

# Close plots
#plt.close('all')

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# Load data
df = pd.read_csv('simulated_data.csv') 

app = dash.Dash()

app.layout = html.Div([

    html.H1('Risk Analytics Dashboard'),
    
    dcc.Dropdown(
        id='risk-type', 
        options=[{'label': x, 'value': x} for x in df.columns],
        value='Attacks'
    ),
    
    dcc.Graph(id='time-series'), 
    
    html.P('Risk likelihood:'),
    dcc.Graph(id='dist-plot')
    
])

@app.callback(
    Output('time-series', 'figure'),
    [Input('risk-type', 'value')])
def update_timeseries(risk):
    
    fig = px.line(df, x=df.index, y=risk)
    return fig

@app.callback(
    Output('dist-plot', 'figure'), 
    [Input('risk-type', 'value')])
def update_dist(risk):
   
    fig = px.histogram(df[risk])
    return fig

app.run_server(debug=True)