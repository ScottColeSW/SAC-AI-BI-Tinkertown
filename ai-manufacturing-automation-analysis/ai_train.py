import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingClassifier

def train_gradient_boosting(dataframe, target_column):
    # features = ["part_temperature", "machine_speed", "measurement_accuracy", "passenger_count", "time_savings", "emergency_signals", "fuel_efficiency", "emission_reduction", "operational_costs"]
    model = GradientBoostingClassifier()
    model.fit(dataframe[features], dataframe[target_column])
    return model

def train_xgboost(dataframe, target_column):
    # features = ["part_temperature", "machine_speed", "measurement_accuracy", "passenger_count", "time_savings", "emergency_signals", "fuel_efficiency", "emission_reduction", "operational_costs"]
    model = xgb.XGBClassifier()
    model.fit(dataframe[features], dataframe[target_column])
    return model

def train_lightgbm(dataframe, target_column):
    # features = ["part_temperature", "machine_speed", "measurement_accuracy", "passenger_count", "time_savings", "emergency_signals", "fuel_efficiency", "emission_reduction", "operational_costs"]
    model = lgb.LGBMClassifier()
    model.fit(dataframe[features], dataframe[target_column])
    return model