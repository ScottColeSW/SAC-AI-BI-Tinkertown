# Here is an example of using Monte Carlo simulation to build a simple probabilistic risk model and generate risk scores based on the simulated data:


# This runs a Monte Carlo simulation using the input data to generate a distribution 
# of risk scores for each risk type. 
# The summary statistics give insights into potential risk likelihood and impact.
import pandas as pd
import numpy as np
from scipy.stats import norm, gamma, poisson

# Input simulated data
data = pd.read_csv('simulated_data.csv', index_col='Date')

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
    attacks = np.random.poisson(lam=data['Attacks'].mean())
    malware = np.random.normal(loc=data['Malware'].mean(), scale=data['Malware'].std())
    vuln = np.random.logistic(loc=data['Vulnerabilities'].mean(), scale=data['Vulnerabilities'].std())
    budget = np.random.triangular(left=data['SecurityBudget'].min(), mode=data['SecurityBudget'].mean(), right=data['SecurityBudget'].max())
    cyber_risk = weights['Attacks']*attacks + weights['Malware']*malware + weights['Vulnerabilities']*vuln + weights['SecurityBudget']*budget
    risk_scores['Cyber'].append(cyber_risk)
    
    # Supply chain risk score
    delay = gamma.rvs(a=data['DeliveryDelay'].mean(), scale=data['DeliveryDelay'].std())
    prices = norm.rvs(loc=data['CommodityPrices'].mean(), scale=data['CommodityPrices'].std())
    inventory = np.random.triangular(left=data['Inventory'].min(), mode=data['Inventory'].mean(), right=data['Inventory'].max()) 
    transport = np.random.lognormal(mean=data['TransportCost'].mean(), sigma=data['TransportCost'].std())
    supplychain_risk = weights['DeliveryDelay']*delay + weights['CommodityPrices']*prices + weights['Inventory']*inventory + weights['TransportCost']*transport
    risk_scores['SupplyChain'].append(supplychain_risk)

    # Regulatory risk score
    audit_def = poisson.rvs(mu=data['AuditDeficiencies'].mean())
    comp_fail = np.random.binomial(n=100, p=data['ComplianceFailures'].mean()/100)
    pol_change = np.random.choice([0, 1], p=[1-data['PolicyChanges'].mean(), data['PolicyChanges'].mean()])
    regulatory_risk = weights['AuditDeficiencies']*audit_def + weights['ComplianceFailures']*comp_fail + weights['PolicyChanges']*pol_change
    risk_scores['Regulatory'].append(regulatory_risk)
    
# Calculate summary statistics   
risk_data = pd.DataFrame(risk_scores)
risk_data.describe()