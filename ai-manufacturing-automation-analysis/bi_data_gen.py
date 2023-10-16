# This generates random time series data simulating metrics for 
# cyber risk, 
# supply chain risk and 
# regulatory risk 
# that can be used for modeling and analysis.Here is some sample Python code to 
# generate simulated historical data for risk analytics:

import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42) 

# Simulate daily data for last 3 years 
num_days = 365*5
dates = pd.date_range(end='2023-12-31', periods=num_days, freq='D')

# Create dataframe
data = pd.DataFrame(index=dates)

# Cyber risk indicators
data['Attacks'] = np.random.randint(low=10, high=100, size=num_days)
data['Malware'] = np.random.normal(loc=50, scale=10, size=num_days).astype(int) 
data['Vulnerabilities'] = np.random.logistic(loc=20, scale=5, size=num_days).astype(int)
data['SecurityBudget'] = np.random.uniform(low=15000, high=25000, size=num_days)

# Supply chain risk indicators
data['DeliveryDelay'] = np.random.gamma(shape=5, scale=2, size=num_days)
data['CommodityPrices'] = np.random.normal(loc=20, scale=5, size=num_days) 
data['Inventory'] = np.random.randint(low=100, high=500, size=num_days)
data['TransportCost'] = np.random.lognormal(mean=5, sigma=1, size=num_days)

# Regulatory risk indicators
data['AuditDeficiencies'] = np.random.poisson(lam=3, size=num_days)
data['ComplianceFailures'] = np.random.binomial(n=100, p=0.05, size=num_days)
data['PolicyChanges'] = np.random.choice([0, 1], size=num_days, p=[0.95, 0.05])

print(data)
data.to_csv('simulated_data.csv')


# Here is an example of using Monte Carlo simulation to build a simple probabilistic risk model and generate risk scores based on the simulated data:


# This runs a Monte Carlo simulation using the input data to generate a distribution 
# of risk scores for each risk type. 
# The summary statistics give insights into potential risk likelihood and impact.
import pandas as pd
import numpy as np
from scipy.stats import norm, gamma, poisson

# Input simulated data
data = pd.read_csv('simulated_data.csv')

# Set weight factors based on risk impact
weights = {'Attacks': 0.2, 
    'Malware': 0.15,
    'Vulnerabilities': 0.15,
    'SecurityBudget': 0.5,
    'DeliveryDelay': 0.3,
    'CommodityPrices': 0.2,
    'Inventory': 0.3, 
    'TransportCost': 0.2,
    'AuditDeficiencies': 0.4,
    'ComplianceFailures': 0.3,
    'PolicyChanges': 0.3}

# Monte carlo simulation          
num_samples = 1000
risk_scores = {'Cyber': [], 'SupplyChain': [], 'Regulatory': []}


for i in range(num_samples):
    # Cyber risk score
    # attacks = np.random.poisson(lam=data['Attacks'].mean())
    # malware = np.random.normal(loc=data['Malware'].mean(), scale=data['Malware'].std())
    attacks = poisson.rvs(data['Attacks'].mean())
    malware = norm.rvs(loc=data['Malware'].mean(), scale=data['Malware'].std()) 
    # etc
    vuln = np.random.logistic(loc=data['Vulnerabilities'].mean(), scale=data['Vulnerabilities'].std())
    budget = np.random.triangular(left=data['SecurityBudget'].min(), mode=data['SecurityBudget'].mean(), right=data['SecurityBudget'].max())
    #cyber_risk = weights['Attacks']*attacks + weights['Malware']*malware + weights['Vulnerabilities']*vuln + weights['SecurityBudget']*budget
    attacks = poisson.rvs(data['Attacks'].mean(), size=num_samples)
    malware = norm.rvs(loc=data['Malware'].mean(), scale=data['Malware'].std(), size=num_samples)

    cyber_risk = (weights['Attacks'] * attacks + 
                weights['Malware'] * malware +
                weights['Vulnerabilities'] * vuln + 
                weights['SecurityBudget'] * budget)
    risk_scores['Cyber'].append(cyber_risk)
    
    # Supply chain risk score
    delay = gamma.rvs(a=data['DeliveryDelay'].mean(), scale=data['DeliveryDelay'].std())
    prices = norm.rvs(loc=data['CommodityPrices'].mean(), scale=data['CommodityPrices'].std())
    inventory = np.random.triangular(left=data['Inventory'].min(), mode=data['Inventory'].mean(), right=data['Inventory'].max()) 
    transport = np.random.lognormal(mean=data['TransportCost'].mean(), sigma=data['TransportCost'].std())
    supplychain_risk = weights['DeliveryDelay']*delay + weights['CommodityPrices']*prices + weights['Inventory']*inventory + weights['TransportCost']*transport
    risk_scores['SupplyChain'].append(supplychain_risk)

    # cyber_risk = cyber_risk.clip(upper=1000)
    # supplychain_risk = supplychain_risk.clip(upper=1000) 

    cyber_risk = cyber_risk.clip(min=-1000, max=1000)


    risk_data = pd.DataFrame({'Cyber': cyber_risk})
    # Copy dataframe 
    risk_data_clipped = risk_data.copy()

    # Clip each column separately
    for col in risk_data:
        risk_data_clipped[col] = risk_data[col].clip(upper=1000)

    # Now plot  
    risk_data_clipped.plot.hist()


    supplychain_risk = supplychain_risk.clip(min=-1000, max=1000)
    # etc
    #risk_data = pd.DataFrame({'Cyber': cyber_risk, 'SupplyChain': supplychain_risk})

    # Regulatory risk score
    audit_def = poisson.rvs(mu=data['AuditDeficiencies'].mean())
    comp_fail = np.random.binomial(n=100, p=data['ComplianceFailures'].mean()/100)
    pol_change = np.random.choice([0, 1], p=[1-data['PolicyChanges'].mean(), data['PolicyChanges'].mean()])
    regulatory_risk = weights['AuditDeficiencies']*audit_def + weights['ComplianceFailures']*comp_fail + weights['PolicyChanges']*pol_change
    risk_scores['Regulatory'].append(regulatory_risk)
    
# Calculate summary statistics   
risk_data = pd.DataFrame(risk_scores)
risk_data.describe()


data.plot(subplots=True, figsize=(8,12))

#risk_data.plot.hist(alpha=0.5)
risk_data_clipped = risk_data.clip(upper=1000)
risk_data_clipped.plot.hist()

cyber_risk = (weights['Attacks']*poisson.rvs(data['Attacks'].mean(), size=num_samples) + 
              weights['Malware']*norm.rvs(loc=data['Malware'].mean(), scale=data['Malware'].std(), size=num_samples)
              + ...)

risk_scores['Cyber'] = cyber_risk

import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm 
import matplotlib.patches as mpatches

palette = cm.tab20(range(20))
lines = ['-', '--', '-.', ':']

for i, col in enumerate(data.columns):
    data[col].plot(color=palette[i], linestyle=lines[i%4])

data['Attacks'].plot()

max_attacks = data['Attacks'].argmax() 
plt.scatter(max_attacks, data['Attacks'].max(), color='red')

plt.annotate('Major Attack', xy=(max_attacks, data['Attacks'].max()),  
             xytext=(max_attacks, data['Attacks'].max()+5),
             arrowprops=dict(facecolor='black'))

# Plot each indicator in a separate subplot
fig, axs = plt.subplots(4, 1, figsize=(10, 8))

data['Attacks'].plot(ax=axs[0], title='Attacks')
data['Malware'].plot(ax=axs[1], title='Malware')  
data['Vulnerabilities'].plot(ax=axs[2], title='Vulnerabilities')
data['SecurityBudget'].plot(ax=axs[3], title='Security Budget')

plt.tight_layout()
plt.show()

# Plot supply chain KPIs
fig, axs = plt.subplots(4, 1, figsize=(10, 8))

data['DeliveryDelay'].plot(ax=axs[0], title='Delivery Delay')
data['CommodityPrices'].plot(ax=axs[1], title='Commodity Prices')
data['Inventory'].plot(ax=axs[2], title='Inventory Levels')  
data['TransportCost'].plot(ax=axs[3], title='Transport Cost')

plt.tight_layout()
plt.show()

ax = data['Inventory'].plot()
ax.set_ylim(0, 500) 
ax.set_xticklabels(data.index.strftime('%Y-%m'))
ax.grid(axis='y')

fig, axs = plt.subplots(2, 1, figsize=(10,6))
data['Attacks'].plot(ax=axs[0], title='Cyber Risk')
data['Inventory'].plot(ax=axs[1], title='Inventory Levels')

# Plot regulatory KPIs
fig, axs = plt.subplots(3, 1, figsize=(10, 6))

data['AuditDeficiencies'].plot(ax=axs[0], title='Audit Deficiencies')
data['ComplianceFailures'].plot(ax=axs[1], title='Compliance Failures') 
data['PolicyChanges'].plot(ax=axs[2], title='Policy Changes')

plt.tight_layout()
plt.show()