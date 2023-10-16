import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
#from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import ai_train as at
import ai_eval as ae
import ai_generate as ag
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
import seaborn as sns
import optuna
from optuna.visualization import plot_optimization_history, plot_parallel_coordinate, plot_slice, plot_param_importances



def generate_synthetic_data(num_points):
    # setup the default tolerances
    #mechanical performance issues
    #Mechanical_Warning_Feature
    parts_temp_tolerance_max = 1000
    machine_speed_tolerance = 3400
    fuel_efficiency_min = 50
    battery_health_min = .2

    #cost performance issues
    #Cost_Inefficiencies_Warning_Feature
    operational_cost_max = 240
    time_savings_min = 100

    #too many warnings
    #Emergency_Signals_Warning_Feature
    emergency_signals_alert = 350
    #uses batter_health_min also

    # setup the data to be used
    data = {
        "passenger_count": [],
        "time_savings": [],
        "emergency_signals": [],
        "fuel_efficiency": [],
        "emission_reduction": [],
        "operational_costs": [],
        "part_temperature": [],
        "machine_speed": [],
        "measurement_accuracy": [],
        "battery_health": [],
        "motor_efficiency": [],
        "flight_time": [],
        "flight_range": [],
        "noise_level": [],
        "charging_time": [],
        "maintenance_intervals": [],
        "avionics_performance": [],
        "component_reliability": [],
        "energy_consumption": [],
        "regulatory_compliance": [],
        "passenger_load_factors": [],
        "weather_impact": [],
        "material_usage_efficiency": [],
        "carbon_emissions": [],
        "time_savings_as_compared_driving": [],
        "payload_capacity": [],
        "emergency_landing_success_rate": [],
        "time_to_emergency_landing": [],
        "safety_margin_for_emergency_landing": [],
        "terrain_analysis": [],
        "weather_impact_on_emergency_landing": [],
        "system_redundancy": [],
        "pilot_training_and_response": [],
        "passenger_safety_during_emergency": [],
        "communication_and_ground_support": [],
        "cabin_alertness": [],
        "flight_problems": [],
        "Mechanical_Warning_Feature": [],
        "Cost_Inefficiencies_Warning_Feature": [],
        "Emergency_Signals_Warning_Feature": []
    }

    for _ in range(num_points):
        time_savings = np.random.randint(30, 2000)
        emergency_signals = np.random.randint(1, 500)
        fuel_efficiency = np.random.randint(10, 150)
        emission_reduction = np.random.randint(100, 200)
        operational_costs = np.random.randint(10, 400)
        part_temperature = np.random.randint(100, 1200)
        machine_speed = np.random.randint(1000, 4000)
        measurement_accuracy = np.random.randint(1, 10)
        battery_health = np.random.uniform(0.001, 1.0)
        motor_efficiency = np.random.uniform(0.85, 1.0)
        flight_time = np.random.uniform(20, 60)
        flight_range = np.random.uniform(.1, 200)
        noise_level = np.random.uniform(60, 80)
        charging_time = np.random.uniform(1, 4)
        maintenance_intervals = np.random.randint(100, 500)
        avionics_performance = np.random.uniform(0.9, 1.0)
        component_reliability = np.random.uniform(0.95, 1.0)
        energy_consumption = np.random.uniform(500, 800)
        regulatory_compliance = np.random.uniform(0.8, 1.0)
        passenger_load_factors = np.random.uniform(0.6, 0.9)
        weather_impact = np.random.uniform(0.7, 1.0)
        material_usage_efficiency = np.random.uniform(0.85, 1.0)
        carbon_emissions = np.random.uniform(30, 70)
        operational_costs = np.random.uniform(1500, 3000)
        passenger_count = np.random.randint(1, 4)
        time_savings_as_compared_driving = np.random.uniform(20, 120)
        payload_capacity = np.random.uniform(100, 500)
        emergency_landing_success_rate = np.random.uniform(0.7, 1.0)
        time_to_emergency_landing = np.random.uniform(1, 5)
        safety_margin_for_emergency_landing = np.random.uniform(100, 500)
        terrain_analysis = np.random.uniform(0.5, 1.0)
        weather_impact_on_emergency_landing = np.random.uniform(0.6, 1.0)
        system_redundancy = np.random.uniform(0.8, 1.0)
        pilot_training_and_response = np.random.uniform(0.7, 1.0)
        passenger_safety_during_emergency = np.random.uniform(0.8, 1.0)
        communication_and_ground_support = np.random.uniform(0.9, 1.0)
        cabin_alertness = np.random.uniform(0.8, 1.0)

        # claude suggestion
        # Sample column values from normal distributions
        time_savings = np.random.normal(loc=100, scale=30, size=num_points) 
        battery_health = np.random.normal(loc=0.8, scale=0.1, size=num_points)

        # Sample column values from uniform distributions
        flight_range = np.random.uniform(low=10, high=300, size=num_points)

        # Add random noise
        noise_level = machine_speed + np.random.normal(scale=10, size=num_points)

        # Introduce some outliers
        fuel_efficiency = np.random.randint(10, 150, size=num_points)
        fuel_efficiency[0:10] = fuel_efficiency[0:10] * 3

        # Clip extreme values
        battery_health = np.clip(battery_health, 0, 1)

        Cost_Inefficiencies_Warning_Feature = (
            (fuel_efficiency <= fuel_efficiency_min).any() | 
            (operational_costs >= operational_cost_max)
        )
        
#        temp_high = (part_temperature >= parts_temp_tolerance_max).any() 
        temp_high = part_temperature >= parts_temp_tolerance_max
        speed_high = machine_speed >= machine_speed_tolerance
        fuel_low = fuel_efficiency <= fuel_efficiency_min
        battery_low = battery_health <= battery_health_min

        #Mechanical_Warning_Feature = (temp_high or speed_high or fuel_low or battery_low)
        #Mechanical_Warning_Feature = np.logical_or(temp_high, speed_high, fuel_low, battery_low)
        temp_speed = np.logical_or(temp_high, speed_high)
        fuel_battery = np.logical_or(fuel_low, battery_low)

        Mechanical_Warning_Feature = np.logical_or(temp_speed, fuel_battery)

        # Engineer a General Flight Warning
        #Mechanical_Warning_Feature = int(part_temperature >= parts_temp_tolerance_max or machine_speed >= machine_speed_tolerance or fuel_efficiency <= fuel_efficiency_min or battery_health <= battery_health_min)
        #Cost_Inefficiencies_Warning_Feature2 = int(time_savings <= time_savings_min or fuel_efficiency <= fuel_efficiency_min or operational_costs >= operational_cost_max)
        Emergency_Signals_Warning_Feature = (emergency_signals >= emergency_signals_alert or battery_health <= battery_health_min)
        
        #flight_problems = (Emergency_Signals_Warning_Feature and Cost_Inefficiencies_Warning_Feature and Mechanical_Warning_Feature)
        flight_problems = np.logical_and(
            Emergency_Signals_Warning_Feature, 
            Cost_Inefficiencies_Warning_Feature,
            Mechanical_Warning_Feature
        )

        data["time_savings"].append(time_savings)
        data["emergency_signals"].append(emergency_signals)
        data["fuel_efficiency"].append(fuel_efficiency)
        data["emission_reduction"].append(emission_reduction)
        data["part_temperature"].append(part_temperature)
        data["machine_speed"].append(machine_speed)
        data["measurement_accuracy"].append(measurement_accuracy)
        data["battery_health"].append(battery_health)
        data["motor_efficiency"].append(motor_efficiency)
        data["flight_time"].append(flight_time)
        data["flight_range"].append(flight_range)
        data["noise_level"].append(noise_level)
        data["charging_time"].append(charging_time)
        data["maintenance_intervals"].append(maintenance_intervals)
        data["avionics_performance"].append(avionics_performance)
        data["component_reliability"].append(component_reliability)
        data["energy_consumption"].append(energy_consumption)
        data["regulatory_compliance"].append(regulatory_compliance)
        data["passenger_load_factors"].append(passenger_load_factors)
        data["weather_impact"].append(weather_impact)
        data["material_usage_efficiency"].append(material_usage_efficiency)
        data["carbon_emissions"].append(carbon_emissions)
        data["operational_costs"].append(operational_costs)
        data["passenger_count"].append(passenger_count)
        data["time_savings_as_compared_driving"].append(time_savings_as_compared_driving)
        data["payload_capacity"].append(payload_capacity)
        data["emergency_landing_success_rate"].append(emergency_landing_success_rate)
        data["time_to_emergency_landing"].append(time_to_emergency_landing)
        data["safety_margin_for_emergency_landing"].append(safety_margin_for_emergency_landing)
        data["terrain_analysis"].append(terrain_analysis)
        data["weather_impact_on_emergency_landing"].append(weather_impact_on_emergency_landing)
        data["system_redundancy"].append(system_redundancy)
        data["pilot_training_and_response"].append(pilot_training_and_response)
        data["passenger_safety_during_emergency"].append(passenger_safety_during_emergency)
        data["communication_and_ground_support"].append(communication_and_ground_support)
        data["cabin_alertness"].append(cabin_alertness)
        data["flight_problems"].append(flight_problems)
        data["Emergency_Signals_Warning_Feature"].append(Emergency_Signals_Warning_Feature)
        data["Cost_Inefficiencies_Warning_Feature"].append(Cost_Inefficiencies_Warning_Feature)
        data["Mechanical_Warning_Feature"].append(Mechanical_Warning_Feature)
        
    return pd.DataFrame(data)

def ai_identify_problems_randforest(dataframe, features, target):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(dataframe[features], dataframe[target], test_size=0.2, random_state=42)
    
    # Initialize the Random Forest Regressor model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the model on the training data
    model.fit(X_train, y_train)
    
    # Make predictions on the testing data
    predictions = model.predict(X_test)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)

    # Assuming y_test and predictions are your true labels and predicted labels
    report = classification_report(y_test, predictions)

    print("Classification Report:\n", report)

    return model


def automated_hyperparameter_tuning(data, features, target_column):
    X_train, X_test, y_train, y_test = train_test_split(data[features], data[target_column], test_size=0.2, random_state=42)
    
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 3, 15)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        
        return cross_val_score(model, X_train, y_train, n_jobs=-1, cv=3).mean()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    
    return study

def hyperparameter_tuning(data, features, target_column, model_class, param_ranges, direction='maximize', n_trials=100):
    X_train, X_test, y_train, y_test = split_data(data, features, target_column)
    
    def objective(trial):
        params = {}
        for param_name, (param_type, param_range) in param_ranges.items():
            if param_type == 'int':
                params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
            elif param_type == 'float':
                params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
        
        model = model_class(**params, random_state=42)
        return cross_val_score(model, X_train, y_train, n_jobs=-1, cv=3).mean()

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    return best_params


def grid_search_gradient_boosting(dataframe, target_column, param_grid):
#    features = ["part_temperature", "machine_speed", "measurement_accuracy", "passenger_count", "time_savings", "emergency_signals", "fuel_efficiency", "emission_reduction", "operational_costs"]
    model = GradientBoostingClassifier()
    
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=3)
    grid_search.fit(dataframe[features], dataframe[target_column])
    
    return grid_search.best_estimator_



def randomized_search_xgboost(dataframe, target_column, param_dist):
#    features = ["part_temperature", "machine_speed", "measurement_accuracy", "passenger_count", "time_savings", "emergency_signals", "fuel_efficiency", "emission_reduction", "operational_costs"]
    model = xgb.XGBClassifier()
    
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, scoring='accuracy', cv=3, n_iter=10)
    random_search.fit(dataframe[features], dataframe[target_column])
    
    return random_search.best_estimator_





# Solution suggestion function
def suggest_solutions(problematic_pairs):
    solutions = []

    # Suggest solutions...

    return solutions


def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 15)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    
    return cross_val_score(model, X_train, y_train, n_jobs=-1, cv=3).mean()

def automated_hyperparameter_tuning(data, features, target_column):
    X_train, X_test, y_train, y_test = train_test_split(data[features], data[target_column], test_size=0.2, random_state=42)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    
    return study

def generate_hyperparameter_tuning_report(study):
    # Visualizations
    plot_optimization_history(study)
    plt.show()

    plot_parallel_coordinate(study)
    plt.show()

    plot_slice(study)
    plt.show()

    plot_param_importances(study)
    plt.show()

def split_data(data, features, target_column, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(data[features], data[target_column], test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

def print_evaluation_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)


def main():
    features = ["Mechanical_Warning_Feature", "Cost_Inefficiencies_Warning_Feature", "Emergency_Signals_Warning_Feature"]
    target_column = "flight_problems"
    num_data_points = 10000

    data = generate_synthetic_data(num_data_points)

    # Train a Random Forest Classifier and print evaluation metrics
    X_train, X_test, y_train, y_test = split_data(data, features, target_column)
    model = RandomForestClassifier(n_estimators=num_data_points, random_state=42)

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()

    model.fit(X_train, y_train)

#    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print_evaluation_metrics(y_test, y_pred)

    # Perform automated hyperparameter tuning with Optuna
    param_ranges = {
        'n_estimators': ('int', (50, num_data_points)),
        'max_depth': ('int', (3, 15)),
        'min_samples_split': ('int', (2, int(.2 * num_data_points)))
    }
    best_params = hyperparameter_tuning(data, features, target_column, RandomForestClassifier, param_ranges)
    print("Best Hyperparameters:", best_params)

 # Use the trained model for predictions
    new_data = pd.DataFrame({
        "Mechanical_Warning_Feature": [1, 0, 1],
        "Cost_Inefficiencies_Warning_Feature": [0, 1, 0],
        "Emergency_Signals_Warning_Feature": [1, 1, 0]
    })
    
    predictions = model.predict(new_data)
    print("Predictions:", predictions)

    # Define your thresholds
    mechanical_threshold = 0.5
    cost_threshold = 0.5
    emergency_threshold = 0.5

    # Make predictions and issue warnings
    for idx, row in new_data.iterrows():
        mechanical_prob = model.predict_proba(row[features].values.reshape(1, -1))[0, 0]
        cost_prob = model.predict_proba(row[features].values.reshape(1, -1))[0, 1]
        emergency_prob = model.predict_proba(row[features].values.reshape(1, -1))[0, 2]

        if mechanical_prob > mechanical_threshold:
            print("Mechanical Warning Issued!")
        if cost_prob > cost_threshold:
            print("Cost Warning Issued!")
        if emergency_prob > emergency_threshold:
            print("Emergency Warning Issued!")


if __name__ == "__main__":
    main()






# #features = ["part_temperature", "machine_speed", "measurement_accuracy", "passenger_count", "time_savings", "emergency_signals", "fuel_efficiency", "emission_reduction", "operational_costs"]

# # Main function
# def main():

#     # List of features
#     features = [
#         "Mechanical_Warning_Feature", 
#         "Cost_Inefficiencies_Warning_Feature", 
#         "Emergency_Signals_Warning_Feature"
#     ]

#     target_column = "flight_problems"

#     # Synthetic data points 
#     num_data_points = 10000
# #    threshold = 0.3
    
#     #pd.set_option('display.max_columns', None)  # Set to display all columns

#     print("Generate Simulated Data for Demonstration")
#     data = generate_synthetic_data(num_data_points)

#     model = ai_identify_problems_randforest(data, features, target_column)
    
#     # Automated hyperparameter tuning with Optuna
#     best_params = automated_hyperparameter_tuning(data, features, target_column)
#     print("Best Hyperparameters:", best_params)

#     # Automated hyperparameter tuning with Optuna
#     study = automated_hyperparameter_tuning(data, features, target_column)
    
#     # Generate hyperparameter tuning report
#     generate_hyperparameter_tuning_report(study)


    


#     # You can now use the trained model for making predictions on new data
#     # For example:
#     new_data = pd.DataFrame({
#         "Mechanical_Warning_Feature": [1, 0, 1],
#         "Cost_Inefficiencies_Warning_Feature": [0, 1, 0],
#         "Emergency_Signals_Warning_Feature": [1, 1, 0]
#     })
    
#     predictions = model.predict(new_data)
#     print("Predictions:", predictions)



    # # ai_identify_problems_randforest(dataframe, features, target)
    # ai_problematic_rand_rows = ai_identify_problems_randforest(data, features, target_column)
    # if not ai_problematic_rand_rows.empty:
    #     print("AI Potential rand Warnings:")
    #     print(ai_problematic_rand_rows)

    # # Usage
    # gb_model = at.train_gradient_boosting(data, target_column)
    # ae.evaluate_model(gb_model, data[features], data[target_column])

    # # Usage
    # xgb_model = at.train_xgboost(data, target_column)
    # ae.evaluate_model(xgb_model, data[features], data[target_column])

    # # Usage
    # lgb_model = at.train_lightgbm(data, target_column)
    # ae.evaluate_model(lgb_model, data[features], data[target_column])

    # # Usage
    # param_grid = {
    #     'n_estimators': [100, 200, 300],
    #     'max_depth': [3, 4, 5],
    #     'learning_rate': [0.1, 0.01]
    # }
    # best_gb_model = grid_search_gradient_boosting(data, target_column, param_grid)

    # # Usage
    # param_dist = {
    #     'n_estimators': [100, 200, 300],
    #     'max_depth': [3, 4, 5],
    #     'learning_rate': [0.1, 0.01],
    #     'subsample': [0.8, 0.9, 1.0],
    #     'colsample_bytree': [0.8, 0.9, 1.0]
    # }
    # best_xgb_model = randomized_search_xgboost(data, target_column, param_dist)

    # # Example evaluation usage
    # y_test = [10, 20, 30, 40, 50]
    # y_pred = [12, 18, 28, 38, 48]

    # evaluate_regression(y_test, y_pred)


    # # Display solutions...
    # solutions = suggest_solutions(problematic_rows)

# Run the main function
# if __name__ == "__main__":
#     main()